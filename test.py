import numpy as np
import torch
import os
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
import torch.optim
import logging
from utils import calculate_rmse, calculate_psnr, calculate_ssim, tonemap, save_output_exr, save_output_png, FilmicTonemap
import time
from dataset import ExrDataset
from model import Network
import settings

logger = settings.logger

np.random.seed(settings.manual_random_seed)
torch.manual_seed(settings.manual_random_seed)
torch.cuda.manual_seed(settings.manual_random_seed)
torch.cuda.manual_seed_all(settings.manual_random_seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
# torch.use_deterministic_algorithms(True)


def load_checkpoints(network, ckp_path):
    try:
        logger.info('Load checkpoint %s' % ckp_path)
        obj = torch.load(ckp_path)
    except FileNotFoundError:
        logger.error('No checkpoint %s!!' % ckp_path)
        raise RuntimeError("No checkpoint %s !! " % ckp_path)
    network.load_state_dict(obj['state_dict'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='last', type=str)
    args = parser.parse_args()
    if args.checkpoint == 'psnr':
        checkpoint_path = os.path.join(settings.save_model_dir, settings.model_name,
                                       settings.model_name + '_best-psnr.pth')
    elif args.checkpoint == 'ssim':
        checkpoint_path = os.path.join(settings.save_model_dir, settings.model_name,
                                       settings.model_name + '_best-ssim.pth')
    elif args.checkpoint == 'rmse':
        checkpoint_path = os.path.join(settings.save_model_dir, settings.model_name,
                                       settings.model_name + '_best-rmse.pth')
    else:
        checkpoint_path = os.path.join(settings.save_model_dir, settings.model_name,
                                       settings.model_name + '_latest.pth')

    network = Network()
    logger.info(network)
    if torch.cuda.is_available():
        network.cuda()
    load_checkpoints(network, checkpoint_path)

    save_root_dir = os.path.join(settings.save_image_dir, settings.model_name, 'test_results_' + args.checkpoint)
    if os.path.exists(save_root_dir):
        os.rename(save_root_dir, save_root_dir + '_archived_' + time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()))
    os.makedirs(save_root_dir, exist_ok=True)

    fh = logging.FileHandler(os.path.join(save_root_dir, 'test.log'))
    # fh.setLevel(logging.INFO)
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    test_params = {'batch_size': 1,
                   'shuffle': False,
                   'pin_memory': True,
                   'num_workers': 10}
    testing_set = ExrDataset(settings.test_image_dir, is_train=False)
    testing_generator = DataLoader(testing_set, **test_params)

    network.eval()
    network.initial_est()
    network.change_mode('test')
    psnr, ssim, output_count = 0., 0., 0
    rmse = 0.
    total_time = 0.
    with torch.no_grad():
        for i, data in enumerate(testing_generator):
            for k in data.keys():
                data[k] = data[k].float().cuda()

            final_gt = data['gt'][..., -1]
            albedo = data['albedo'][..., -1]
            gt_albedo = data['gt_albedo'][..., -1]
            start_time = time.time()
            output = network(data)
            temp_time = time.time() - start_time
            total_time += temp_time
            output = output[...,-1]
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
            logger.info("[Test {:04d}], count {:d}, psnr: {:.4f}, ssim: {:.4f}, rmse: {:.4f}, time: {:.4f}".format(
                i, temp_count, temp_psnr / temp_count, temp_ssim / temp_count, temp_rmse / temp_count, temp_time / temp_count
            ))
            psnr += temp_psnr
            ssim += temp_ssim
            rmse += temp_rmse
            output_count += temp_count

            mean_psnr = psnr / output_count
            mean_ssim = ssim / output_count
            mean_rmse = rmse / output_count

    logger.info('[Test {}] psnr: {:.4f} ssim: {:.4f} rmse: {:.4f} time: {:.4f}'.format(
                settings.model_name, mean_psnr, mean_ssim, mean_rmse, total_time / output_count))
    logger.info("Synthesis test results images to video...")
    os.system("ffmpeg -apply_trc iec61966_2_1 -i {} {}".format(
        os.path.join(save_root_dir, "%04d.exr"),
        os.path.join(save_root_dir, "test_{}.mp4".format(args.checkpoint))
    ))


if __name__ == '__main__':
    main()
