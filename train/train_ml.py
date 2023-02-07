import time
from options.train_options import TrainOptions
from models.networks import ResUnetGenerator, VGGLoss, save_checkpoint, load_checkpoint_parallel
from models.afwm import TVLoss, AFWM
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
import cv2
import datetime
from util import flow_util
from util import util


def de_offset(s_grid):
    [b, _, h, w] = s_grid.size()

    x = torch.arange(w).view(1, -1).expand(h, -1).float()
    y = torch.arange(h).view(-1, 1).expand(-1, w).float()
    x = 2 * x / (w - 1) - 1
    y = 2 * y / (h - 1) - 1
    grid = torch.stack([x, y], dim=0).float().cuda()
    grid = grid.unsqueeze(0).expand(b, -1, -1, -1)

    offset = grid - s_grid

    offset_x = offset[:, 0, :, :] * (w - 1) / 2
    offset_y = offset[:, 1, :, :] * (h - 1) / 2

    offset = torch.cat((offset_y, offset_x), 0)

    return offset

def eval_delta_loss(delta_x, epsilon):
    delta_loss = (delta_x.pow(2) + epsilon * epsilon).pow(0.45)
    return delta_loss
    
f2c = flow_util.flow2color()
opt = TrainOptions().parse()
path = 'runs/' + opt.name
os.makedirs(path, exist_ok=True)
os.makedirs(opt.checkpoints_dir, exist_ok=True)
torch.backends.cudnn.benchmark = True


def CreateDataset(opt):
    # training with augumentation
    # from data.aligned_dataset import AlignedDataset_aug
    # dataset = AlignedDataset_aug()
    from data.aligned_dataset import AlignedDataset
    dataset = AlignedDataset()
    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset


os.makedirs('sample_fs', exist_ok=True)
iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')

torch.cuda.set_device(opt.local_rank)
torch.distributed.init_process_group(
    'nccl',
    init_method='env://'
)
device = torch.device(f'cuda:{opt.local_rank}')

start_epoch, epoch_iter = 1, 0

train_data = CreateDataset(opt)
train_sampler = DistributedSampler(train_data)
train_loader = DataLoader(train_data, batch_size=opt.batchSize, shuffle=False,
                          num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)
dataset_size = len(train_loader)

PBTN = AFWM(opt, 3)
PFSN = AFWM(opt, 3)

PBTN.train()
PBTN.cuda()
PBTN = torch.nn.SyncBatchNorm.convert_sync_batchnorm(PBTN).to(device)
if opt.load_pretrain:
    load_checkpoint_parallel(PBTN, opt.PBTN_checkpoint)

PFSN.train()
PFSN.cuda()
PFSN = torch.nn.SyncBatchNorm.convert_sync_batchnorm(PFSN).to(device)
if opt.load_pretrain:
    load_checkpoint_parallel(PFSN, opt.PFSN_checkpoint)

gen_model = ResUnetGenerator(7, 4, 5, ngf=64, norm_layer=nn.BatchNorm2d)
gen_model.eval()
gen_model.cuda()
load_checkpoint_parallel(gen_model, opt.gen_checkpoint)

if opt.isTrain and len(opt.gpu_ids):
    teacher_model = torch.nn.parallel.DistributedDataParallel(PBTN, device_ids=[opt.local_rank])
    stud_model = torch.nn.parallel.DistributedDataParallel(PFSN, device_ids=[opt.local_rank])
    gen_model = torch.nn.parallel.DistributedDataParallel(gen_model, device_ids=[opt.local_rank])

criterionL1 = nn.L1Loss()
criterionVGG = VGGLoss()

params_warp_tea = [p for p in teacher_model.parameters()]
params_warp_stud = [p for p in stud_model.parameters()]
optimizer_warp_tea = torch.optim.Adam(params_warp_tea, lr=opt.lr, betas=(opt.beta1, 0.999))
optimizer_warp_stud = torch.optim.Adam(params_warp_stud, lr=opt.lr, betas=(opt.beta1, 0.999))

total_steps = (start_epoch - 1) * dataset_size + epoch_iter
step = 1
step_per_batch = dataset_size

if opt.local_rank == 0:
    writer = SummaryWriter(path)

for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    if epoch != start_epoch:
        epoch_iter = epoch_iter % dataset_size
    start_time = time.time()
    train_sampler.set_epoch(epoch) # to make shuffling work properly across multiple epochs

    for i, data in enumerate(train_loader):
        iter_start_time = time.time()

        total_steps += 1
        epoch_iter += 1
        save_fake = True

        t_mask = torch.FloatTensor((data['label'].cpu().numpy() == 7).astype(np.float64))
        data['label'] = data['label'] * (1 - t_mask) + t_mask * 4
        parsing_result,person_clothes,densepose_fore,clothes_un,pre_clothes_edge_un, clothes,pre_clothes_edge,real_image,person_clothes_edge  = util.prepare_data(data)
        # import ipdb; ipdb.set_trace()
        with torch.no_grad():
            flow_out_int = teacher_model(real_image.cuda(), clothes_un.cuda(), pre_clothes_edge_un.cuda())
            warped_cloth_int, last_flow_int, _1, _2, _3, x_all_int, x_edge_all_int, _5, _6 = flow_out_int
            warped_prod_edge_int = x_edge_all_int[4]

            gen_inputs_int = torch.cat([real_image.cuda(), warped_cloth_int, warped_prod_edge_int], 1)
            gen_outputs_int = gen_model(gen_inputs_int)
            p_rendered_int, m_composite_int = torch.split(gen_outputs_int, [3, 1], 1)
            p_rendered_int = torch.tanh(p_rendered_int)
            m_composite_int = torch.sigmoid(m_composite_int)
            m_composite_int = m_composite_int * warped_prod_edge_int
            p_tryon_int = warped_cloth_int * m_composite_int + p_rendered_int * (1 - m_composite_int)

        flow_out = teacher_model(real_image.cuda(), clothes.cuda(), pre_clothes_edge.cuda())
        warped_cloth, last_flow, cond_all, flow_all, delta_list, x_all, x_edge_all, delta_x_all, delta_y_all = flow_out
        warped_prod_edge = x_edge_all[4]

        flow_out_stud = stud_model(p_tryon_int.detach(), clothes.cuda(), pre_clothes_edge.cuda())
        warped_cloth_stud, last_flow_stud, cond_all_stud, flow_all_stud, delta_list_stud, x_all_stud, x_edge_all_stud, delta_x_all_stud, delta_y_all_stud = flow_out_stud
        warped_prod_edge_stud = x_edge_all_stud[4]

        epsilon = 0.001
        loss_smooth_tea = sum([TVLoss(x) for x in delta_list])
        loss_smooth_stud = sum([TVLoss(x) for x in delta_list_stud])
        loss_smooth_total = loss_smooth_tea + loss_smooth_stud
        # print('Loss smooth total: ',loss_smooth_total)
        loss_all = 0
        loss_tea = 0
        loss_stud = 0
        l1_loss_batch = torch.abs(warped_cloth.detach() - person_clothes.cuda())
        l1_loss_batch = l1_loss_batch.reshape(opt.batchSize, 3 * 256 * 192)
        l1_loss_batch = l1_loss_batch.sum(dim=1) / (3 * 256 * 192)
        l1_loss_batch_pred = torch.abs(warped_cloth_stud.detach() - person_clothes.cuda())
        l1_loss_batch_pred = l1_loss_batch_pred.reshape(opt.batchSize, 3 * 256 * 192)
        l1_loss_batch_pred = l1_loss_batch_pred.sum(dim=1) / (3 * 256 * 192)
        weight = (l1_loss_batch < l1_loss_batch_pred).float()
        num_all = len(np.where(weight.cpu().numpy() > 0)[0])
       
        # print('Num all: ',num_all)
        if num_all == 0:
            num_all = 1
    
        t_feature = 0
        s_feature = 0
        for num in range(5):
            cur_person_clothes = F.interpolate(person_clothes, scale_factor=0.5 ** (4 - num), mode='bilinear')
            cur_person_clothes_edge = F.interpolate(person_clothes_edge, scale_factor=0.5 ** (4 - num), mode='bilinear')
            loss_l1 = criterionL1(x_all[num], cur_person_clothes.cuda())
            loss_vgg = criterionVGG(x_all[num], cur_person_clothes.cuda())
            loss_edge = criterionL1(x_edge_all[num], cur_person_clothes_edge.cuda())
            b, c, h, w = delta_x_all[num].shape
            loss_flow_x = eval_delta_loss(delta_x_all[num], epsilon)
            loss_flow_x = torch.sum(loss_flow_x) / (b * c * h * w)
            loss_flow_y = eval_delta_loss(delta_y_all[num], epsilon)
            loss_flow_y = torch.sum(loss_flow_y) / (b * c * h * w)
            loss_second_smooth = loss_flow_x + loss_flow_y
            b1, c1, h1, w1 = cond_all[num].shape
            
            loss_tea = loss_tea + (num + 1) * loss_l1 + (num + 1) * 0.2 * loss_vgg + (num + 1) * 2 * loss_edge + (
                        num + 1) * 6 * loss_second_smooth 

            loss_l1_stud = criterionL1(x_all_stud[num], cur_person_clothes.cuda())
            loss_vgg_stud = criterionVGG(x_all_stud[num], cur_person_clothes.cuda())
            loss_edge_stud = criterionL1(x_edge_all_stud[num], cur_person_clothes_edge.cuda())

            loss_flow_x_stud = eval_delta_loss(delta_x_all_stud[num], epsilon)
            loss_flow_x_stud = torch.sum(loss_flow_x_stud) / (b * c * h * w)
            loss_flow_y_stud = eval_delta_loss(delta_y_all_stud[num], epsilon)
            loss_flow_y_stud = torch.sum(loss_flow_y_stud) / (b * c * h * w)
            loss_second_smooth_stud = loss_flow_x_stud + loss_flow_y_stud
            
            weight_all = weight.reshape(-1, 1, 1, 1).repeat(1, 256, h1, w1)
            cond_sup_loss = ((cond_all[num].detach() - cond_all_stud[num]) ** 2 * weight_all).sum() / (256 * h1 * w1 * num_all)

            loss_stud = loss_stud + (num + 1) * loss_l1_stud + (num + 1) * 0.2 * loss_vgg_stud + (
                        num + 1) * 2 * loss_edge_stud + (num + 1) * 6 * loss_second_smooth_stud + cond_sup_loss


            if num >= 2:
                b1, c1, h1, w1 = flow_all_stud[num].shape
                weight_flow_stud = weight.reshape(-1, 1, 1).repeat(1, h1, w1)
                flow_loss_stud = (torch.norm(flow_all[num].detach() - flow_all_stud[num], p=2,
                                             dim=1) * weight_flow_stud).sum() / (h1 * w1 * num_all)
                loss_stud = loss_stud + (num + 1) * 1 * flow_loss_stud
                # print('Flow loss-{}: {}'.format(num,flow_loss_stud))

        loss_all = 0.01 * loss_smooth_total + loss_tea + loss_stud
        t_feature = cond_all[4].sum()
        s_feature = cond_all_stud[4].sum()
        if opt.local_rank == 0:
            writer.add_scalar("t_feature",t_feature,step)
            writer.add_scalar("s_feature",s_feature,step)
            writer.add_scalar("feature_loss",cond_sup_loss/num_all,step)
            writer.add_scalar('loss_all', loss_all, step)

        optimizer_warp_tea.zero_grad(set_to_none=True)
        optimizer_warp_stud.zero_grad(set_to_none=True)
        loss_all.backward()
        optimizer_warp_tea.step()
        optimizer_warp_stud.step()
        ############## Display results and errors ##########

        path = opt.name
        os.makedirs(path, exist_ok=True)
        if step % 300 == 0:
            if opt.local_rank == 0:
                a = real_image.float().cuda()
                b = person_clothes.cuda()  # GT
                c = clothes.cuda()
                c_un = clothes_un.cuda()
                d = torch.cat([densepose_fore.cuda(), densepose_fore.cuda(), densepose_fore.cuda()], 1)
                e = warped_cloth
                e_stu = warped_cloth_stud
                e_un = warped_cloth_int
                p = p_tryon_int
                flow_offset = de_offset(last_flow[0].unsqueeze(0))
                flow_color = f2c(flow_offset).cuda()
                flow_offset_int = de_offset(last_flow_int[0].unsqueeze(0))
                flow_color_int = f2c(flow_offset_int).cuda()
                flow_offset_stud = de_offset(last_flow_stud[0].unsqueeze(0))
                flow_color_stud = f2c(flow_offset_stud).cuda()

                combine = torch.cat(
                    [a[0], b[0], c[0], c_un[0], d[0], e[0], e_stu[0], e_un[0], p[0], flow_color, flow_color_stud,
                     flow_color_int], 2).squeeze()
                cv_img = (combine.permute(1, 2, 0).detach().cpu().numpy() + 1) / 2
                writer.add_image('combine', (combine.data + 1) / 2.0, step)
                rgb = (cv_img * 255).astype(np.uint8)
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite(path + '/' + str(step) + '.jpg', bgr)

        step += 1
        iter_end_time = time.time()
        iter_delta_time = iter_end_time - iter_start_time
        step_delta = (step_per_batch - step % step_per_batch) + step_per_batch * (opt.niter + opt.niter_decay - epoch)
        eta = iter_delta_time * step_delta
        eta = str(datetime.timedelta(seconds=int(eta)))
        time_stamp = datetime.datetime.now()
        now = time_stamp.strftime('%Y.%m.%d-%H:%M:%S')
        if step % 100 == 0:
            if opt.local_rank == 0:
                end_time = time.time()
                elapsed_time = end_time - start_time
                print(f"num_all-{num_all}-t_feature: {t_feature} - s_feature: {s_feature}")
                print('{}:{}:[step-{}]--[loss total-{:.4f}]--[loss teacher-{}]--[loss_student-{}]--[ETA-{}]'.format(now, epoch_iter,step, loss_all,loss_tea,loss_stud,eta))

        if epoch_iter >= dataset_size:
            break

    # end of epoch
    iter_end_time = time.time()
    if opt.local_rank == 0:
        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    ### save model for this epoch
    if epoch % opt.save_epoch_freq == 0:
        if opt.local_rank == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
            save_checkpoint(teacher_model.module,
                            os.path.join(opt.checkpoints_dir, opt.name, 'teacher_warp_epoch_%03d.pth' % (epoch + 1)))
            save_checkpoint(stud_model.module,
                            os.path.join(opt.checkpoints_dir, opt.name, 'student_warp_epoch_%03d.pth' % (epoch + 1)))

    if epoch > opt.niter:
        teacher_model.module.update_learning_rate(optimizer_warp_tea)
        stud_model.module.update_learning_rate(optimizer_warp_stud)


































