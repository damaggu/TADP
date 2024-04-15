# ------------------------------------------------------------------------------
#
# Mostly copied and adapted from VPD.
# https://github.com/wl-zhao/VPD/blob/main/depth/test.py
#
# The code is from GLPDepth (https://github.com/vinvino02/GLPDepth).
# For non-commercial purpose only (research, evaluation etc).
# -----------------------------------------------------------------------------

import os
from collections import OrderedDict

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from TADP.tadp_depth import TADPDepth
import models.depth.utils_depth.metrics as metrics
import models.depth.utils_depth.logging as logging

from datasets.depth.base_dataset import get_dataset
from models.depth.configs.test_options import TestOptions
import models.depth.utils_depth.distributed as dist_utils

metric_name = ['d1', 'd2', 'd3', 'abs_rel', 'sq_rel', 'rmse', 'rmse_log',
               'log10', 'silog']


def main():
    opt = TestOptions()
    args = opt.initialize().parse_args()
    print(args)

    if dist_utils.is_launched_with_torch_distributed():
        print("Running on distributed.")
        dist_utils.init_distributed_mode_simple(args)
        device = torch.device(args.gpu)
    else:
        print("Running on single GPU.")
        device = torch.device('cuda')
        args.rank = 0

    args.shift_window_test = True  # TODO test/validate does not work if this is off

    model = TADPDepth(args=args)

    # CPU-GPU agnostic settings

    cudnn.benchmark = True
    model.to(device)

    if args.ckpt_dir is None:
        raise ValueError('--ckpt_dir is required.')

    model_weight = torch.load(args.ckpt_dir)['model']
    if 'module' in next(iter(model_weight.items()))[0]:
        model_weight = OrderedDict((k[7:], v) for k, v in model_weight.items())
    model.load_state_dict(model_weight, strict=False)
    model.eval()

    if dist_utils.is_launched_with_torch_distributed():
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)

    # Dataset setting
    dataset_kwargs = {'dataset_name': args.dataset, 'data_path': args.data_path}
    dataset_kwargs['crop_size'] = (args.crop_h, args.crop_w)

    val_dataset = get_dataset(**dataset_kwargs, is_train=False)

    sampler_val = torch.utils.data.DistributedSampler(
        val_dataset, num_replicas=dist_utils.get_world_size(), rank=args.rank, shuffle=False)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, sampler=sampler_val,
                                             pin_memory=True)

    # Perform experiment

    results_dict = validate(val_loader, model,
                            device=device, args=args)
    if args.rank == 0:
        result_lines = logging.display_result(results_dict)
        print(result_lines)


def validate(val_loader, model, device, args):
    if args.save_eval_pngs or args.save_visualize:
        result_path = os.path.join(args.result_dir, args.exp_name)
        if args.rank == 0:
            logging.check_and_make_dirs(result_path)
        print("Saving result images in to %s" % result_path)

    if args.rank == 0:
        depth_loss = logging.AverageMeter()
    model.eval()

    ddp_logger = logging.MetricLogger()

    result_metrics = {}
    for metric in metric_name:
        result_metrics[metric] = 0.0

    for batch_idx, batch in enumerate(val_loader):
        # if batch['filename'][0] != 'bathroom_rgb_00743.jpg':
        #     continue
        print(f'{batch_idx} / {len(val_loader)}')
        input_RGB = batch['image'].to(device)
        depth_gt = batch['depth'].to(device)
        filename = batch['filename'][0]
        class_id = batch['class_id']
        metas = {'img_paths': batch['ori_img_path']}

        with torch.no_grad():
            if args.shift_window_test:
                bs, _, h, w = input_RGB.shape
                assert w > h and bs == 1
                interval_all = w - h
                interval = interval_all // (args.shift_size - 1)
                sliding_images = []
                sliding_masks = torch.zeros((bs, 1, h, w), device=input_RGB.device)
                class_ids = []
                for i in range(args.shift_size):
                    sliding_images.append(input_RGB[..., :, i * interval:i * interval + h])
                    sliding_masks[..., :, i * interval:i * interval + h] += 1
                    class_ids.append(class_id)
                input_RGB = torch.cat(sliding_images, dim=0)
                class_ids = torch.cat(class_ids, dim=0)
            if args.flip_test:
                input_RGB = torch.cat((input_RGB, torch.flip(input_RGB, [3])), dim=0)
                class_ids = torch.cat((class_ids, class_ids), dim=0)
            metas['img_paths'] = metas['img_paths'] * input_RGB.shape[0]
            pred = model(input_RGB, metas, class_ids=class_ids)
        pred_d = pred['pred_d']
        if args.flip_test:
            batch_s = pred_d.shape[0] // 2
            pred_d = (pred_d[:batch_s] + torch.flip(pred_d[batch_s:], [3])) / 2.0
        if args.shift_window_test:
            pred_s = torch.zeros((bs, 1, h, w), device=pred_d.device)
            for i in range(args.shift_size):
                pred_s[..., :, i * interval:i * interval + h] += pred_d[i:i + 1]
            pred_d = pred_s / sliding_masks

        pred_d = pred_d.squeeze()
        depth_gt = depth_gt.squeeze()

        pred_crop, gt_crop = metrics.cropping_img(args, pred_d, depth_gt)
        computed_result = metrics.eval_depth(pred_crop, gt_crop)

        if args.save_eval_pngs:
            save_path = os.path.join(result_path, filename)
            if save_path.split('.')[-1] == 'jpg':
                save_path = save_path.replace('jpg', 'png')
            pred_d = pred_d.squeeze()
            if args.dataset == 'nyudepthv2':
                pred_d = pred_d.cpu().numpy() * 1000.0
                cv2.imwrite(save_path, pred_d.astype(np.uint16),
                            [cv2.IMWRITE_PNG_COMPRESSION, 0])
            else:
                pred_d = pred_d.cpu().numpy() * 256.0
                cv2.imwrite(save_path, pred_d.astype(np.uint16),
                            [cv2.IMWRITE_PNG_COMPRESSION, 0])

        if args.save_visualize:
            save_path = os.path.join(result_path, filename)
            os.makedirs('corresponding_gt_for_results', exist_ok=True)
            save_path_for_gt = os.path.join('corresponding_gt_for_results', filename)
            pred_d_numpy = pred_d.squeeze().cpu().numpy()

            if args.trim_edges:
                pred_d_numpy = pred_d_numpy[60:-20][:, 20:-20]
            pred_d_numpy = (pred_d_numpy / pred_d_numpy.max()) * 255
            pred_d_numpy = pred_d_numpy.astype(np.uint8)

            pred_d_color = cv2.applyColorMap(pred_d_numpy, cv2.COLORMAP_MAGMA)
            cv2.imwrite(save_path, pred_d_color)

            depth_gt = depth_gt.squeeze().cpu().numpy()
            depth_gt = depth_gt[60:-20][:, 20:-20]
            depth_gt = (depth_gt / depth_gt.max()) * 255
            depth_gt = depth_gt.astype(np.uint8)
            depth_gt = cv2.applyColorMap(depth_gt, cv2.COLORMAP_MAGMA)
            cv2.imwrite(save_path_for_gt, depth_gt)

        ddp_logger.update(**computed_result)
        for key in result_metrics.keys():
            result_metrics[key] += computed_result[key]

    ddp_logger.synchronize_between_processes()

    for key in result_metrics.keys():
        result_metrics[key] = ddp_logger.meters[key].global_avg

    return result_metrics


if __name__ == '__main__':
    main()
