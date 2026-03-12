from ast import arg
import os
import argparse
import torch
import numpy as np
import random
import time
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, SequentialLR, LambdaLR

from utils.config import get_config
from utils.evaluation import get_eval
from models.model_dict import get_model
from utils.loss_functions.sam_loss import get_criterion
from utils.generate_prompts import get_click_prompt
from utils.print import print_arguments
from utils.split_dataset import prepare_data_with_split


def parse_arguments():
    """Parse command line arguments and configuration file"""
    parser = argparse.ArgumentParser(description='Networks')
    parser.add_argument('--modelname', default='RFLSeg', type=str, help='Model type')
    parser.add_argument('-encoder_input_size', type=int, default=256, help='Encoder input size')
    parser.add_argument('-low_image_size', type=int, default=128, help='Image embedding size')
    parser.add_argument('--task', default='BUSI', help='Task name')
    parser.add_argument('--vit_name', type=str, default='vit_b', help='ViT model name')
    parser.add_argument('--sam_ckpt', type=str, default='checkpoints/sam_vit_b_01ec64.pth', help='SAM pretrained weights')
    parser.add_argument('--batch_size', type=int, default=8, help='GPU batch size')
    parser.add_argument('--n_gpu', type=int, default=1, help='Number of GPUs')
    parser.add_argument('--base_lr', type=float, default=5e-4, help='Base learning rate')
    parser.add_argument('--warmup', type=bool, default=False, help='Whether to use learning rate warmup')
    parser.add_argument('--warmup_period', type=int, default=250, help='Warmup iterations')
    parser.add_argument('--min_lr', type=float, default=5e-5, help='Minimum learning rate')
    parser.add_argument('--patience', type=int, default=5, help='Learning rate decay patience')
    parser.add_argument('--early_stopping_patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--reduce_factor', type=float, default=0.5, help='Learning rate decay factor')
    parser.add_argument('--cosine_epochs', type=int, default=30, help='Cosine annealing period')
    parser.add_argument('-keep_log', type=bool, default=True, help='Whether to save training logs')
    parser.add_argument('--is_active_lr', type=bool, default=True, help='Whether to enable dynamic learning rate')
    parser.add_argument('--use_coarse_mask', type=bool, default=True, help='True:coarse_mask; False:fine_mask')
    parser.add_argument('--preseg_path', type=str, default='./output/BUSI/deeplabv3+_resnet50/')
    parser.add_argument('--gen_name', type=str, default='coarse_gen')
    parser.add_argument('--auto_split', type=bool, default=False, help='Whether to use auto split, True: use split_dataset.py to split; False: use default split')
    parser.add_argument('--split_name', type=str, default='0')

    args = parser.parse_args()
    opt = get_config(args.task)
    opt.save_path = args.preseg_path + "sam_checkpoints/"
    return args, opt


def setup_environment(opt):
    """Setup training environment and random seeds"""
    print(f"Using {'cuda' if torch.cuda.is_available() else 'cpu'} device for training...")
    device = torch.device(opt.device)

    # Set random seeds
    seed_value = 1234
    np.random.seed(seed_value)
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True

    return device


def prepare_model(args, opt, device):
    """Prepare model and load pretrained weights"""
    model = get_model(args.modelname, args=args, opt=opt)
    model.to(device)

    if opt.pre_trained:
        print(f"Loading pretrained model: {opt.load_path}")
        checkpoint = torch.load(opt.load_path)
        model.load_state_dict(checkpoint)

    if args.n_gpu > 1:
        model = nn.DataParallel(model)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {pytorch_total_params}")

    return model

def prepare_optimizer_scheduler(args, model, train_loader):
    """Prepare optimizer and learning rate scheduler"""
    if args.warmup:
        b_lr = args.base_lr / args.warmup_period
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=b_lr,
            betas=(0.9, 0.999),
            weight_decay=0.1
        )
    else:
        b_lr = args.base_lr
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.base_lr,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=0,
            amsgrad=False
        )

    scheduler = None
    if args.is_active_lr:
        warmup_scheduler = LambdaLR(
            optimizer,
            lr_lambda=lambda step: min(1.0, (step + 1) / args.warmup_period)
        ) if args.warmup else None

        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=args.cosine_epochs * len(train_loader),
            eta_min=args.min_lr
        )

        plateau_scheduler = ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=args.reduce_factor,
            patience=args.patience,
            verbose=True,
            min_lr=args.min_lr
        )

        if args.warmup:
            schedulers = [warmup_scheduler, cosine_scheduler]
            milestones = [args.warmup_period]
            scheduler = SequentialLR(optimizer, schedulers, milestones)
        else:
            scheduler = cosine_scheduler
    else:
        print("Using fixed learning rate, disabling dynamic learning rate adjustment")

    return optimizer, scheduler, plateau_scheduler


def train_epoch(model, train_loader, optimizer, criterion, scheduler, device, args, opt, epoch, TensorWriter=None):
    """Train a single epoch"""
    model.train()
    train_losses = 0
    lr_history = []

    with tqdm(train_loader, desc=f'Epoch {epoch + 1} Training', unit='batch') as tepoch:
        for batch_idx, datapack in enumerate(tepoch):
            # Prepare data
            imgs = datapack['image'].to(dtype=torch.float32, device=device)
            gen_imgs = datapack['generated_image'].to(dtype=torch.float32, device=device)
            coarse_masks = datapack['low_coarse_mask'].to(dtype=torch.float32, device=device)
            masks = datapack['low_mask'].to(dtype=torch.float32, device=device)
            bbox = torch.as_tensor(datapack['bbox'], dtype=torch.float32, device=device)
            pt = get_click_prompt(datapack, opt)

            pred = model(imgs, gen_imgs, pt, bbox, coarse_masks)
            train_loss = criterion(pred, masks)

            # Backward propagation
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            # Update learning rate
            current_lr = optimizer.param_groups[0]['lr']
            if args.is_active_lr and args.warmup:
                scheduler.step()

            # Record loss and learning rate
            train_losses += train_loss.item()
            lr_history.append(current_lr)
            tepoch.set_postfix(loss=train_loss.item(), lr=current_lr)

    # Calculate average loss
    train_loss_avg = train_losses / len(train_loader)
    print(f'Epoch {epoch + 1} training completed, average loss: {train_loss_avg:.4f}')

    # Record TensorBoard
    if TensorWriter:
        TensorWriter.add_scalar('train_loss', train_loss_avg, epoch)
        TensorWriter.add_scalar('learning_rate', current_lr, epoch)

    return train_loss_avg, lr_history


def validate_model(model, val_loader, criterion, device, args, opt, epoch, TensorWriter=None):
    """Validate model performance"""
    model.eval()
    dices, mean_dice, mean_hdis, val_loss = get_eval(
        val_loader, model, criterion, opt, args, isFineIterate=False
    )

    print(f'Epoch {epoch + 1} validation results - Dice: {mean_dice:.4f}, HD95: {mean_hdis:.4f}, Loss: {val_loss:.4f}')

    # Record TensorBoard
    if TensorWriter:
        TensorWriter.add_scalar('val_loss', val_loss, epoch)
        TensorWriter.add_scalar('dices', mean_dice, epoch)
        TensorWriter.add_scalar('hdis', mean_hdis, epoch)

    return mean_dice, mean_hdis, val_loss


def save_model(model, save_path, epoch, metric_name, metric_value):
    """Save model and record metrics"""
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    torch.save(model.state_dict(), save_path)
    print(f"Saving model: {save_path}, {metric_name}: {metric_value:.4f}")
    return save_path


def main():
    # Parse arguments and configuration
    args, opt = parse_arguments()
    print_arguments(args, opt)
    device = setup_environment(opt)


    # Prepare TensorBoard
    TensorWriter = None
    if args.keep_log:
        logtimestr = time.strftime('%m%d%H%M')
        boardpath = f"{opt.tensorboard_path}{args.modelname}{opt.save_path_code}{logtimestr}"
        os.makedirs(boardpath, exist_ok=True)
        TensorWriter = SummaryWriter(boardpath)

    # Prepare model and data
    model = prepare_model(args, opt, device)
    # Modify here to call the new data preparation function
    if args.auto_split:
        train_loader, val_loader = prepare_data_with_split(args, opt, os.path.join(opt.data_subpath, "splits", args.split_name))
    else:
        train_loader, val_loader = prepare_data_with_split(args, opt, os.path.join(opt.data_subpath, "splits"))
    # Prepare optimizer and loss function
    optimizer, scheduler, plateau_scheduler = prepare_optimizer_scheduler(args, model, train_loader)
    criterion = get_criterion(modelname=args.modelname, opt=opt)

    # Initialize training status
    best_dice = 0.0
    best_dice_file = None
    min_val_loss = float('inf')
    min_val_loss_file = None

    best_dice_after_30 = 0.0
    best_dice_after_40 = 0.0
    best_dice_after_30_file = None
    best_dice_after_40_file = None
    best_dice_epoch = -1
    best_dice_after_30_epoch = -1
    best_dice_after_40_epoch = -1

    min_val_loss_after_30 = float('inf')
    min_val_loss_after_40 = float('inf')
    min_val_loss_after_30_file = None
    min_val_loss_after_40_file = None

    min_val_loss_epoch = -1
    min_val_loss_after_30_epoch = -1
    min_val_loss_after_40_epoch = -1

    no_improvement_epochs = 0
    early_stopping = False

    # Training loop
    for epoch in range(opt.epochs):
        # Display current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch + 1}/{opt.epochs}, Current learning rate: {current_lr:.6f}')
        if early_stopping:
            print("Early stopping triggered, stopping training")
            break

        # Training phase
        train_loss_avg, lr_history = train_epoch(
            model, train_loader, optimizer, criterion, scheduler, device, args, opt, epoch, TensorWriter
        )

        # Update learning rate (non-warmup mode)
        if args.is_active_lr and not args.warmup:
            scheduler.step()

        # Validation phase
        if epoch % opt.eval_freq == 0:
            mean_dice, mean_hdis, val_loss = validate_model(
                model, val_loader, criterion, device, args, opt, epoch, TensorWriter
            )

            # Update learning rate (plateau decay)
            if args.is_active_lr and epoch >= args.cosine_epochs:
                plateau_scheduler.step(mean_dice)

            # Save best Dice model
            # 保存全局最佳Dice模型
            if mean_dice > best_dice:
                if best_dice_file:
                    os.remove(best_dice_file)
                timestr = time.strftime('%m%d%H%M')
                best_dice_file = save_model(
                    model,
                    os.path.join(opt.save_path,
                                 f"{args.modelname}{opt.save_path_code}_{timestr}_epoch{epoch}_dicemax_{mean_dice:.4f}.pth"),
                    epoch,
                    "Dice",
                    mean_dice
                )
                best_dice = mean_dice
                best_dice_epoch = epoch
                no_improvement_epochs = 0
            else:
                no_improvement_epochs += 1

            # 保存全局最小loss模型
            if val_loss < min_val_loss:
                if min_val_loss_file:
                    os.remove(min_val_loss_file)
                timestr = time.strftime('%m%d%H%M')
                min_val_loss_file = save_model(
                    model,
                    os.path.join(opt.save_path,
                                 f"{args.modelname}{opt.save_path_code}_{timestr}_epoch{epoch}_lossmin_{val_loss:.4f}.pth"),
                    epoch,
                    "Loss",
                    val_loss
                )
                min_val_loss = val_loss
                min_val_loss_epoch = epoch

            # 检查并保存第30轮后的最佳Dice模型
            if epoch >= 30 and mean_dice > best_dice_after_30:
                # 只有当不是同一个模型时才保存
                if epoch != best_dice_epoch:
                    if best_dice_after_30_file and os.path.exists(best_dice_after_30_file):
                        os.remove(best_dice_after_30_file)
                    timestr = time.strftime('%m%d%H%M')
                    best_dice_after_30_file = save_model(
                        model,
                        os.path.join(opt.save_path,
                                     f"{args.modelname}{opt.save_path_code}_{timestr}_epoch{epoch}_after30_dicemax_{mean_dice:.4f}.pth"),
                        epoch,
                        "Dice after 30 epochs",
                        mean_dice
                    )
                best_dice_after_30 = mean_dice
                best_dice_after_30_epoch = epoch

            # 检查并保存第40轮后的最佳Dice模型
            if epoch >= 40 and mean_dice > best_dice_after_40:
                # 只有当不是同一个模型时才保存
                if epoch != best_dice_epoch and epoch != best_dice_after_30_epoch:
                    if best_dice_after_40_file and os.path.exists(best_dice_after_40_file):
                        os.remove(best_dice_after_40_file)
                    timestr = time.strftime('%m%d%H%M')
                    best_dice_after_40_file = save_model(
                        model,
                        os.path.join(opt.save_path,
                                     f"{args.modelname}{opt.save_path_code}_{timestr}_epoch{epoch}_after40_dicemax_{mean_dice:.4f}.pth"),
                        epoch,
                        "Dice after 40 epochs",
                        mean_dice
                    )
                best_dice_after_40 = mean_dice
                best_dice_after_40_epoch = epoch

            # 检查并保存第30轮后的最小loss模型
            if epoch >= 30 and val_loss < min_val_loss_after_30:
                # 只有当不是同一个模型时才保存
                if epoch != min_val_loss_epoch:
                    if min_val_loss_after_30_file and os.path.exists(min_val_loss_after_30_file):
                        os.remove(min_val_loss_after_30_file)
                    timestr = time.strftime('%m%d%H%M')
                    min_val_loss_after_30_file = save_model(
                        model,
                        os.path.join(opt.save_path,
                                     f"{args.modelname}{opt.save_path_code}_{timestr}_epoch{epoch}_after30_lossmin_{val_loss:.4f}.pth"),
                        epoch,
                        "Loss after 30 epochs",
                        val_loss
                    )
                min_val_loss_after_30 = val_loss
                min_val_loss_after_30_epoch = epoch

            # 检查并保存第40轮后的最小loss模型
            if epoch >= 40 and val_loss < min_val_loss_after_40:
                # 只有当不是同一个模型时才保存
                if epoch != min_val_loss_epoch and epoch != min_val_loss_after_30_epoch:
                    if min_val_loss_after_40_file and os.path.exists(min_val_loss_after_40_file):
                        os.remove(min_val_loss_after_40_file)
                    timestr = time.strftime('%m%d%H%M')
                    min_val_loss_after_40_file = save_model(
                        model,
                        os.path.join(opt.save_path,
                                     f"{args.modelname}{opt.save_path_code}_{timestr}_epoch{epoch}_after40_lossmin_{val_loss:.4f}.pth"),
                        epoch,
                        "Loss after 40 epochs",
                        val_loss
                    )
                min_val_loss_after_40 = val_loss
                min_val_loss_after_40_epoch = epoch
            # Early stopping check
            current_lr = optimizer.param_groups[0]['lr']
            if current_lr <= args.min_lr and no_improvement_epochs >= args.early_stopping_patience:
                early_stopping = True
        print("-------------------------------------------")
    print("Training completed")


if __name__ == '__main__':
    main()
