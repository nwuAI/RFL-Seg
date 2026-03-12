
import argparse
from torch.autograd import Variable
import torch
import torch.nn.functional as F
import utils.metrics as metrics
from hausdorff import hausdorff_distance
import numpy as np
from tqdm import tqdm

from utils.metric2Excel import save_metrics_to_excel
from utils.visualization import visual_segmentation, \
    visual_segmentation_sets_with_pt, visualize_predict_and_seg, visualize_preseg
from utils.generate_prompts import get_click_prompt
import time
from medpy.metric.binary import hd95

def preseg_eval_mask_slice2(valloader, model, criterion, opt, cfg, isInfer):
    model.eval()
    val_losses, mean_dice = 0, 0
    max_slice_number = cfg.batch_size * (len(valloader) + 1)
    dices = np.zeros((max_slice_number, opt.classes))

    hds_hd95 = np.zeros((max_slice_number, opt.classes))
    ious, accs, ses, sps = np.zeros((max_slice_number, opt.classes)), np.zeros(
        (max_slice_number, opt.classes)), np.zeros((max_slice_number, opt.classes)), np.zeros(
        (max_slice_number, opt.classes))
    eval_number = 0
    sum_time = 0
    for batch_idx, (datapack) in enumerate(valloader):
        imgs = Variable(datapack['image'].to(dtype=torch.float32, device=opt.device))
        label = Variable(datapack['label'].to(dtype=torch.float32, device=opt.device))

        with torch.no_grad():
            start_time = time.time()
            pred = model(imgs)
            sum_time = sum_time + (time.time() - start_time)
        if not isInfer:
            val_loss = criterion(pred, label)
            val_losses += val_loss.item()

        gt = label.detach().cpu().numpy()
        gt = gt[:, 0, :, :]
        predict = torch.sigmoid(pred)
        predict = predict.detach().cpu().numpy()  # (b, c, h, w)
        seg = predict[:, 0, :, :] > 0.5  # (b, h, w)


        visualize_preseg(seg, datapack['image_name'], cfg.save_dir)
        if not isInfer:
            b, h, w = seg.shape
            for j in range(0, b):
                pred_i = np.zeros((1, h, w))
                pred_i[seg[j:j + 1, :, :] == 1] = 255
                gt_i = np.zeros((1, h, w))
                gt_i[gt[j:j + 1, :, :] == 1] = 255
                dice_i = metrics.dice_coefficient(pred_i, gt_i)
                dices[eval_number + j, 1] += dice_i
                iou, acc, se, sp = metrics.sespiou_coefficient2(pred_i, gt_i, all=False)
                ious[eval_number + j, 1] += iou
                accs[eval_number + j, 1] += acc
                ses[eval_number + j, 1] += se
                sps[eval_number + j, 1] += sp
                try:
                    hds_hd95[eval_number + j, 1] += hd95(pred_i[0, :, :], gt_i[0, :, :], voxelspacing=None, connectivity=1)
                except RuntimeError as e:
                    if "does not contain any binary object" in str(e):
                        hds_hd95[eval_number + j, 1] += np.nan
                    else:
                        raise e
                del pred_i, gt_i
            eval_number = eval_number + b

    if not isInfer:
        dices = dices[:eval_number, :]
        hds_hd95 = hds_hd95[:eval_number, :]
        ious, accs, ses, sps = ious[:eval_number, :], accs[:eval_number, :], ses[:eval_number, :], sps[:eval_number, :]
        val_losses = val_losses / (batch_idx + 1)

        dice_mean = np.nanmean(dices, axis=0)
        hd_mean = np.nanmean(hds_hd95, axis=0)

        mean_dice = np.nanmean(dice_mean[1:])
        mean_hdis = np.nanmean(hd_mean[1:])
        if opt.mode == "train":
            return dices, mean_dice, mean_hdis, val_losses
        else:
            dice_mean, hd_mean, iou_mean, acc_mean, se_mean, sp_mean, dices_std, hd_std, iou_std, acc_std, se_std, sp_std = save_metrics_to_excel(
                cfg, dices, hds_hd95, ious, accs, ses, sps)
            return dice_mean, hd_mean, iou_mean, acc_mean, se_mean, sp_mean, dices_std, hd_std, iou_std, acc_std, se_std, sp_std

def eval_mask_slice2(valloader, model, criterion, opt, args, isFineIterate):
    model.eval()
    val_losses, mean_dice = 0, 0
    max_slice_number = opt.batch_size * (len(valloader) + 1)
    dices = np.zeros((max_slice_number, opt.classes))
    hds = np.zeros((max_slice_number, opt.classes))
    ious, accs, ses, sps = np.zeros((max_slice_number, opt.classes)), np.zeros(
        (max_slice_number, opt.classes)), np.zeros((max_slice_number, opt.classes)), np.zeros(
        (max_slice_number, opt.classes))
    eval_number = 0
    sum_time = 0

    pbar = tqdm(enumerate(valloader), total=len(valloader), desc="Evaluating")
    for batch_idx, (datapack) in pbar:
        imgs = Variable(datapack['image'].to(dtype=torch.float32, device=opt.device))

        gen_imgs = Variable(datapack['generated_image'].to(dtype=torch.float32, device=opt.device))
        masks = Variable(datapack['low_mask'].to(dtype=torch.float32, device=opt.device))  # 比较评估是根据mask来的
        low_coarse_masks = Variable(datapack['low_coarse_mask'].to(dtype=torch.float32, device=opt.device))
        label = Variable(datapack['label'].to(dtype=torch.float32, device=opt.device))
        image_filename = datapack['image_name']
        pt = get_click_prompt(datapack, opt)
        bbox = torch.as_tensor(datapack['bbox'], dtype=torch.float32, device=opt.device)
        with torch.no_grad():
            start_time = time.time()
            pred = model(imgs, gen_imgs, pt, bbox, low_coarse_masks)
            sum_time = sum_time + (time.time() - start_time)

        if not isFineIterate:
            val_loss = criterion(pred, masks)
            val_losses += val_loss.item()

        if args.modelname == 'MSA' or args.modelname == 'SAM':
            gt = masks.detach().cpu().numpy()
        else:
            gt = label.detach().cpu().numpy()
        gt = gt[:, 0, :, :]
        predict = torch.sigmoid(pred['masks'])
        predict = predict.detach().cpu().numpy()  # (b, c, h, w)
        seg = predict[:, 0, :, :] > 0.5  # (b, h, w)


        visualize_predict_and_seg(seg, datapack['image_name'], opt.result_path)
        if not isFineIterate:
            b, h, w = seg.shape
            for j in range(0, b):
                pred_i = np.zeros((1, h, w))
                pred_i[seg[j:j + 1, :, :] == 1] = 255
                gt_i = np.zeros((1, h, w))
                gt_i[gt[j:j + 1, :, :] == 1] = 255
                dice_i = metrics.dice_coefficient(pred_i, gt_i)

                dices[eval_number + j, 1] += dice_i
                iou, acc, se, sp = metrics.sespiou_coefficient2(pred_i, gt_i, all=False)
                ious[eval_number + j, 1] += iou
                accs[eval_number + j, 1] += acc
                ses[eval_number + j, 1] += se
                sps[eval_number + j, 1] += sp
                try:
                    hds[eval_number + j, 1] += hd95(pred_i[0, :, :], gt_i[0, :, :], voxelspacing=None, connectivity=1)
                except RuntimeError as e:

                    if "does not contain any binary object" in str(e):
                        hds[eval_number + j, 1] += np.nan
                    else:
                        raise e
                del pred_i, gt_i
                if opt.visual:
                    visual_segmentation_sets_with_pt(seg[j:j + 1, :, :], image_filename[j], opt, pt[0][j, :, :])
            eval_number = eval_number + b


        if eval_number > 0:
            current_dice = np.mean(dices[:eval_number, 1]) / eval_number * opt.classes if eval_number > 0 else 0
            pbar.set_postfix({
                'Loss': f'{val_losses / (batch_idx + 1):.4f}',
                'Dice': f'{current_dice:.4f}',
                'Samples': eval_number
            })

    if not isFineIterate:
        dices = dices[:eval_number, :]
        hds = hds[:eval_number, :]
        ious, accs, ses, sps = ious[:eval_number, :], accs[:eval_number, :], ses[:eval_number, :], sps[:eval_number, :]
        val_losses = val_losses / (batch_idx + 1)
        dice_mean = np.mean(dices, axis=0)
        hd_mean = np.nanmean(hds, axis=0)
        mean_dice = np.mean(dice_mean[1:])
        mean_hdis = np.nanmean(hd_mean[1:])
        if opt.mode == "train":
            return dices, mean_dice, mean_hdis, val_losses
        else:
            cfg = argparse.Namespace()
            cfg.load_model_dict_name = args.load_cp
            cfg.save_dir = args.preseg_path
            dice_mean, hd_mean, iou_mean, acc_mean, se_mean, sp_mean, dices_std, hd_std, iou_std, acc_std, se_std, sp_std = save_metrics_to_excel(
                cfg, dices, hds, ious, accs, ses, sps)
            return dice_mean, hd_mean, iou_mean, acc_mean, se_mean, sp_mean, dices_std, hd_std, iou_std, acc_std, se_std, sp_std

def to_device(batch_input, device):
    device_input = []
    for one_dict in batch_input:
        one_dict_device = {}
        for key, value in one_dict.items():
            if key == 'image' or key == 'labels' or key == 'boxes' or key == 'point_coords' or key == 'point_labels':
                one_dict_device[key] = value.float().to(device)
            else:
                one_dict_device[key] = value
        device_input.append(one_dict_device)
    return device_input
def get_eval(valloader, model, criterion, opt, args, isFineIterate):
    if opt.eval_mode == "mask_slice":
        return eval_mask_slice2(valloader, model, criterion, opt, args, isFineIterate)
    else:
        raise RuntimeError("Could not find the eval mode:", opt.eval_mode)

def preseg_eval(valloader, model, criterion, opt, cfg, isInfer):
    return preseg_eval_mask_slice2(valloader, model, criterion, opt, cfg, isInfer)