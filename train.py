import numpy as np
import torch
import os
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim
from utils import calculate_rmse, calculate_ssim, calculate_psnr, save_output_exr, FilmicTonemap
import logging
import time
import random
from dataset import ExrDataset
from model import Network
import settings
from losses import rae_loss
from distutils.dir_util import copy_tree
from torch.utils.tensorboard import SummaryWriter

logger = settings.logger
val_logger = settings.val_logger

np.random.seed(settings.manual_random_seed)
torch.manual_seed(settings.manual_random_seed)
torch.cuda.manual_seed(settings.manual_random_seed)
torch.cuda.manual_seed_all(settings.manual_random_seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
# torch.use_deterministic_algorithms(True)
g = torch.Generator()
g.manual_seed(settings.manual_random_seed)


def save_model(epoch, network, optimizer, checkpoint_root_dir, save_model_path=None):
    if save_model_path is None:
        save_model_path = os.path.join(checkpoint_root_dir, settings.model_name + '_' + "{:04d}".format(epoch)  + '.pth')
    logger.info('SAVING MODEL AT EPOCH %s' % (epoch + 1))
    torch.save({
        'epoch': epoch,
        'state_dict': network.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, save_model_path)

def train_clip(outputs, data):
    return rae_loss(outputs, data)

def train(cur_epoch, network, optimizer, training_generator, writer=None):
    network.train()
    network.initial_est()
    network.change_mode('train')

    tot_loss=tot_ls=tot_lt=tot_lp=0

    start_time = time.time()
    for i, data in enumerate(training_generator):
        for k in data.keys():
            data[k] = data[k].float().cuda()

        optimizer.zero_grad()
        outputs = network(data)
        loss, ls, lt, lp = train_clip(outputs, data)
        loss.backward()
        optimizer.step()

        tot_loss += loss.item()
        tot_ls += ls.item()
        tot_lt += lt.item()
        tot_lp += lp.item()

        logger.debug('[TRAIN] epoch: %d  [%d / %d], average tot_loss: %f, L1 spatial loss: %f, L1 temporal loss: %f '
                         'L1 HFEN loss: %f' % (cur_epoch + 1, i + 1, len(training_generator), tot_loss / (i + 1), ls.item(), 
                         lt.item(), lp.item()))

        if writer is not None and i % settings.tb_record_interval == 0:
            cur_iter = cur_epoch * len(training_generator) + i + 1
            writer.add_scalar('Loss/L1_spatial', ls.item(), cur_iter)
            writer.add_scalar('Loss/L1_temporal', lt.item(), cur_iter)
            writer.add_scalar('Loss/L1_HFEN', lp.item(), cur_iter)
            writer.add_scalar('Loss/Total', loss.item(), cur_iter)
    
    logger.info('[TRAIN] epoch: %d, lr: %.4e, tot_loss: %.4e time: %.4f'
                % (cur_epoch + 1, optimizer.param_groups[0]['lr'],
                   tot_loss / (len(training_generator)), time.time() - start_time))

def val(cur_epoch, network, testing_generator, save_root_dir, writer=None):
    network.eval()
    network.initial_est()
    network.change_mode('test')

    fh = logging.FileHandler(os.path.join(save_root_dir, 'validation.log'), mode='w')
    fh.setLevel(logging.DEBUG)
    val_logger.addHandler(fh)
    psnr = ssim = rmse = 0.
    output_count = 0
    total_time = 0
    with torch.no_grad():
        for i, data in enumerate(testing_generator):
            for k in data.keys():
               data[k] = data[k].float().cuda()

            final_gt = data['gt'][..., -1]
            albedo = data['albedo'][..., -1]
            gt_albedo = data['gt_albedo'][..., -1]
            start_time = time.time()
            out = network(data)
            temp_time = time.time() - start_time
            total_time += temp_time
            output = out[..., -1]
            output = torch.expm1(output)
            output *= albedo
            final_gt = torch.expm1(final_gt)
            final_gt *= gt_albedo
            save_output_exr(output, save_root_dir, '{:04d}'.format(i+1))

            out = output.cpu().detach().numpy()
            out = np.transpose(out, (0, 2, 3, 1))
            gt = final_gt.cpu().detach().numpy()
            gt = np.transpose(gt, (0, 2, 3, 1))

            temp_psnr = 0
            temp_ssim = 0
            temp_rmse = 0
            temp_count = 0

            for bn in range(gt.shape[0]):
                out_mapping = FilmicTonemap(out[bn]) * 255.
                gt_mapping = FilmicTonemap(gt[bn]) * 255.
                temp_psnr += calculate_psnr(out_mapping, gt_mapping)
                temp_ssim += calculate_ssim(out_mapping, gt_mapping)
                temp_rmse += calculate_rmse(out[bn], gt[bn])
                temp_count += 1
            val_logger.debug("[Validation {:04d}], count {:d}, psnr: {:.4f}, ssim: {:.4f}, rmse: {:.4f}, time: {:.4f}".format(
                i, temp_count, temp_psnr / temp_count, temp_ssim / temp_count, temp_rmse / temp_count, temp_time / temp_count
            ))
            psnr += temp_psnr
            ssim += temp_ssim
            rmse += temp_rmse
            output_count += temp_count

        mean_psnr = psnr / output_count
        mean_ssim = ssim / output_count
        mean_rmse = rmse / output_count

        logger.info('[Validation {}] epoch: {:04d} psnr: {:.4f} ssim: {:.4f} rmse: {:.4f} time: {:.4f}'.format(
            settings.model_name, cur_epoch + 1, mean_psnr, mean_ssim, mean_rmse, total_time / output_count))
        val_logger.info('[Validation {}] epoch: {:04d} psnr: {:.4f} ssim: {:.4f} rmse: {:.4f} time: {:.4f}'.format(
            settings.model_name, cur_epoch + 1, mean_psnr, mean_ssim, mean_rmse, total_time / output_count))
        if writer is not None:
            writer.add_scalar('Val/PSNR', mean_psnr, cur_epoch + 1)
            writer.add_scalar('Val/SSIM', mean_ssim, cur_epoch + 1)
            writer.add_scalar('Val/RMSE', mean_rmse, cur_epoch + 1)
    val_logger.removeHandler(fh)
    return mean_psnr, mean_ssim, mean_rmse


def seed_worker(worker_id):
    worker_seed = settings.manual_random_seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def main():
    network = Network()
    if torch.cuda.is_available():
        network.cuda()

    optimizer = torch.optim.Adam(network.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(settings.epoch_num / 6), int(settings.epoch_num / 3 * 2)], gamma=0.5)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200,800], gamma=0.5)
    if settings.resume:
        checkpoint = torch.load(settings.resume_model_path)
        start_epoch = checkpoint['epoch'] + 1
        network.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()
        print('Resume training! start from epoch: ', start_epoch)
        logger.info('Resume training! start from epoch: ' + str(start_epoch))
    else:
        start_epoch = 0

    checkpoint_root_dir = os.path.join(settings.save_model_dir, settings.model_name)
    if os.path.exists(checkpoint_root_dir):
        os.rename(checkpoint_root_dir, checkpoint_root_dir + '_archived_' + time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()))

    os.makedirs(checkpoint_root_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=checkpoint_root_dir)
    copy_tree('/jizhi/jizhi2/worker/trainer/bistro', os.path.join(checkpoint_root_dir, 'ssd'))
    
    fh = logging.FileHandler(os.path.join(checkpoint_root_dir, 'training.log'))
    # fh.setLevel(logging.INFO)
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    val_root_dir = os.path.join(checkpoint_root_dir, "val_results_latest")
    if not os.path.exists(val_root_dir):
        os.makedirs(val_root_dir)

    logger.info(">>>>> Sparse Sampling Denoising {} <<<<<".format(settings.model_name))
    logger.info(network)

    train_params = {'batch_size': settings.batch_size,
                    'shuffle': True,
                    'pin_memory': True,
                    'num_workers': settings.num_workers,
                    'drop_last': True,
                    'worker_init_fn': seed_worker,
                    'prefetch_factor': 2,
                    'generator': g}

    val_params = {'batch_size': 1,
                   'shuffle': False,
                   'pin_memory': True,
                   'num_workers': 10,
                   'worker_init_fn': seed_worker,
                    'generator': g}

    training_set = ExrDataset(settings.train_root_dir, is_train=True)
    training_generator = DataLoader(training_set, **train_params)

    validation_set = ExrDataset(settings.val_root_dir, is_train=False)
    val_generator = DataLoader(validation_set, **val_params)

    best_psnr = best_ssim = -1
    best_rmse = float('inf')
    for epoch in range(start_epoch, settings.epoch_num):
        train(epoch, network, optimizer, training_generator)
        val_psnr, val_ssim, val_rmse = val(epoch, network, val_generator, save_root_dir=val_root_dir)

        if val_psnr >= best_psnr:
            logger.info(
                ">>>>> At epoch {:04d}, Save Best PSNR Model: PSNR {:.4f}, SSIM {:.4f}, RMSE {:.4f}. <<<<<".format(
                    epoch + 1, val_psnr, val_ssim, val_rmse))
            copy_tree(os.path.join(settings.save_image_dir, settings.model_name, "val_results_latest"),
                      os.path.join(settings.save_image_dir, settings.model_name, "val_results_best_psnr"))
            save_model(epoch, network, optimizer, checkpoint_root_dir,
                       save_model_path=os.path.join(checkpoint_root_dir, settings.model_name + '_best-psnr.pth'))
            best_psnr = val_psnr

        if val_ssim >= best_ssim:
            logger.info(
                ">>>>> At epoch {:04d}, Save Best SSIM Model: PSNR {:.4f}, SSIM {:.4f}, RMSE {:.4f}. <<<<<".format(
                    epoch + 1, val_psnr, val_ssim, val_rmse))
            copy_tree(os.path.join(settings.save_image_dir, settings.model_name, "val_results_latest"),
                      os.path.join(settings.save_image_dir, settings.model_name, "val_results_best_ssim"))
            save_model(epoch, network, optimizer, checkpoint_root_dir,
                       save_model_path=os.path.join(checkpoint_root_dir, settings.model_name + '_best-ssim.pth'))
            best_ssim = val_ssim

        if val_rmse < best_rmse:
            logger.info(
                ">>>>> At epoch {:04d}, Save Best RMSE Model: PSNR {:.4f}, SSIM {:.4f}, RMSE {:.4f}. <<<<<".format(
                    epoch + 1, val_psnr, val_ssim, val_rmse))
            copy_tree(os.path.join(settings.save_image_dir, settings.model_name, "val_results_latest"),
                      os.path.join(settings.save_image_dir, settings.model_name, "val_results_best_rmse"))
            save_model(epoch, network, optimizer, checkpoint_root_dir,
                       save_model_path=os.path.join(checkpoint_root_dir, settings.model_name + '_best-rmse.pth'))
            best_rmse = val_rmse

        latest_save_model_path = os.path.join(checkpoint_root_dir, settings.model_name + '_latest.pth')
        save_model(epoch, network, optimizer, checkpoint_root_dir, save_model_path=latest_save_model_path)
        scheduler.step()

    logger.info("Synthesis val best results images to video...")
    os.system("ffmpeg -apply_trc iec61966_2_1 -i {} {}".format(
        os.path.join(settings.save_image_dir, settings.model_name, "val_results_best_psnr", "%04d.exr"),
        os.path.join(settings.save_image_dir, settings.model_name, "val_results_best_psnr", "val_best_psnr.mp4")
    ))
    os.system("ffmpeg -apply_trc iec61966_2_1 -i {} {}".format(
        os.path.join(settings.save_image_dir, settings.model_name, "val_results_best_ssim", "%04d.exr"),
        os.path.join(settings.save_image_dir, settings.model_name, "val_results_best_ssim", "val_best_ssim.mp4")
    ))
    os.system("ffmpeg -apply_trc iec61966_2_1 -i {} {}".format(
        os.path.join(settings.save_image_dir, settings.model_name, "val_results_best_rmse", "%04d.exr"),
        os.path.join(settings.save_image_dir, settings.model_name, "val_results_best_rmse", "val_best_rmse.mp4")
    ))


if __name__ == '__main__':
    main()
