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
from utils.evaluation import preseg_eval
from utils.loss_functions.sam_loss import get_pre_seg_criterion

from segmentation_models_pytorch import Unet, UnetPlusPlus, DeepLabV3, DeepLabV3Plus
from colorama import init, Fore, Style
from segmentation_models_pytorch.encoders import get_preprocessing_params

from utils.split_dataset import coarse_seg_prepare_data, coarse_seg_prepare_test_data

class Config:
    def __init__(self):

        self.model_choices = {
            'unet': Unet,
            'unet++': UnetPlusPlus,
            'deeplabv3': DeepLabV3,
            'deeplabv3+': DeepLabV3Plus
        }
        self.encoder_choices = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'efficientnet-b0', 'efficientnet-b1',
                                'efficientnet-b2']

        self.model_name = "deeplabv3+"
        self.encoder_name = "resnet50"
        self.encoder_weights = "imagenet"
        self.in_channels = 3
        self.num_classes = 1
        self.task = "BUSI"
        self.batch_size = 32
        self.n_gpu = 1
        self.base_lr = 5e-4
        self.min_lr = 5e-5
        self.patience = 5
        self.early_stopping_patience = 10
        self.reduce_factor = 0.5
        self.cosine_epochs = 30
        self.keep_log = True
        self.is_active_lr = True
        self.auto_split = False
        self.split_output_dir = 'BUSI_1'
        self.save_dir = f"./output/{self.task}/{self.model_name}_{self.encoder_name}"
        self.load_model_dict_name = "deeplabv3+__02101048_epoch49_dicemax_0.7869.pth" # None
        self.test_model_path = os.path.join(self.save_dir, "checkpoints", self.load_model_dict_name)
        os.makedirs(os.path.join(self.save_dir, "checkpoints"), exist_ok=True)

        if self.model_name not in self.model_choices:
            raise ValueError(f"Error Model")
        if self.encoder_name not in self.encoder_choices:
            raise ValueError(f"Error Encoder")
    def get_model_class(self):
        return self.model_choices[self.model_name]



def custom_preprocess_input(x, encoder_name):
    params = get_preprocessing_params(encoder_name)
    mean = np.array(params['mean'])
    std = np.array(params['std'])
    mean = np.expand_dims(np.expand_dims(mean, -1), -1)
    std = np.expand_dims(np.expand_dims(std, -1), -1)
    x = x - mean
    x = x / std
    return x



init(autoreset=True)


def parse_opt(config):
    opt = get_config(config.task)
    return opt


def setup_environment(opt):
    device = torch.device(opt.device)

    seed_value = 42
    np.random.seed(seed_value)
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True

    return device


def prepare_model(cfg, opt, device):
    model_class = cfg.get_model_class()
    model = model_class(
        encoder_name=cfg.encoder_name,
        encoder_weights=cfg.encoder_weights,
        in_channels=cfg.in_channels,
        classes=cfg.num_classes
    ).to(opt.device)

    if cfg.n_gpu > 1:
        model = nn.DataParallel(model)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"total parameters count: {pytorch_total_params}")

    return model

def prepare_optimizer_scheduler(cfg, model, train_loader):

    b_lr = cfg.base_lr
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.base_lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0,
        amsgrad=False
    )

    scheduler = None
    if cfg.is_active_lr:

        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=cfg.cosine_epochs * len(train_loader),
            eta_min=cfg.min_lr
        )

        plateau_scheduler = ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=cfg.reduce_factor,
            patience=cfg.patience,
            verbose=True,
            min_lr=cfg.min_lr
        )
        scheduler = cosine_scheduler
    else:
        plateau_scheduler = None

    return optimizer, scheduler, plateau_scheduler


def train_epoch(model, train_loader, optimizer, criterion, device, epoch, TensorWriter=None):
    model.train()
    train_losses = 0
    lr_history = []

    with tqdm(train_loader, desc=f'Epoch {epoch + 1} Training', unit='batch') as tepoch:
        for batch_idx, datapack in enumerate(tepoch):

            imgs = datapack['image'].to(dtype=torch.float32, device=device)
            masks = datapack['label'].to(dtype=torch.float32, device=device)

            pred = model(imgs)
            train_loss = criterion(pred, masks)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            current_lr = optimizer.param_groups[0]['lr']

            train_losses += train_loss.item()
            lr_history.append(current_lr)
            tepoch.set_postfix(loss=train_loss.item(), lr=current_lr)


    train_loss_avg = train_losses / len(train_loader)
    print(f'Epoch {epoch + 1} complete, average loss: {train_loss_avg:.4f}')

    if TensorWriter:
        TensorWriter.add_scalar('train_loss', train_loss_avg, epoch)
        TensorWriter.add_scalar('learning_rate', current_lr, epoch)

    return train_loss_avg, lr_history


def validate_model(model, val_loader, criterion, opt, cfg, epoch, TensorWriter=None):
    model.eval()
    dices, mean_dice, mean_hdis, val_loss = preseg_eval(val_loader, model, criterion, opt, cfg, isInfer=False)

    print(f'Epoch {epoch + 1} val result - Dice: {mean_dice:.4f}, HD: {mean_hdis:.4f}, loss: {val_loss:.4f}')

    if TensorWriter:
        TensorWriter.add_scalar('val_loss', val_loss, epoch)
        TensorWriter.add_scalar('dices', mean_dice, epoch)
        TensorWriter.add_scalar('hdis', mean_hdis, epoch)

    return mean_dice, mean_hdis, val_loss


def save_model(model, save_path, epoch, metric_name, metric_value):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    torch.save(model.state_dict(), save_path)
    print(f"save model: {save_path}, {metric_name}: {metric_value:.4f}")
    return save_path


def test_model(model, test_loader, cfg, opt, criterion, isInfer):
    opt.mode = "test"
    model.train()
    checkpoint = torch.load(cfg.test_model_path)
    # ------when the load model is saved under multiple GPU
    new_state_dict = {}
    for k, v in checkpoint.items():
        if k[:7] == 'module.':
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    #  ========================================================================= begin to evaluate the model ============================================================================

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if not isInfer:
        print(Fore.CYAN + f"Total_params: {pytorch_total_params}")
    model.eval()

    if not isInfer:
        mean_dice, mean_hdis, mean_iou, mean_acc, mean_se, mean_sp, std_dice, std_hdis, std_iou, std_acc, std_se, std_sp = preseg_eval(
            test_loader, model, criterion, opt, cfg, isInfer)
        printMethod(cfg, opt, mean_dice, mean_hdis, mean_iou, mean_acc, mean_se, mean_sp, std_dice, std_hdis, std_iou,
                    std_acc, std_se, std_sp)
    else:
        preseg_eval(test_loader, model, criterion, opt, cfg, isInfer)


def printMethod(cfg, opt, mean_dice, mean_hdis, mean_iou, mean_acc, mean_se, mean_sp, std_dice, std_hdis, std_iou,
                std_acc, std_se, std_sp):
    print(Fore.MAGENTA + "=" * 80)
    print(Fore.MAGENTA + f"Dataset: {cfg.task} | Model: {cfg.model_name}")
    print(Fore.MAGENTA + f"Checkpoint: {cfg.test_model_path}")
    print(Fore.MAGENTA + "=" * 80)
    print(Fore.BLUE + "{:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
        "metric", "Dice", "HD", "IoU", "Acc", "Se", "Sp"
    ))
    for i in range(1, len(mean_dice)):
        print(Fore.BLUE + "{:<10} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f}".format(
            f"Class {i}",
            mean_dice[i],
            mean_hdis[i],
            mean_iou[i],
            mean_acc[i],
            mean_se[i],
            mean_sp[i]
        ))
    print(Fore.WHITE + "-" * 80)
    print(Fore.GREEN + "{:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
        "std", "Dice", "HD", "IoU", "Acc", "Se", "Sp"
    ))

    for i in range(1, len(std_dice)):
        print(Fore.GREEN + "{:<10} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f}".format(
            f"Class {i}",
            std_dice[i],
            std_hdis[i],
            std_iou[i],
            std_acc[i],
            std_se[i],
            std_sp[i]
        ))
    print(Fore.WHITE + "-" * 80)
    print(Fore.YELLOW + "{:<10} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f}".format(
        "mean",
        np.mean(mean_dice[1:]),
        np.mean(mean_hdis[1:]),
        np.mean(mean_iou[1:]),
        np.mean(mean_acc[1:]),
        np.mean(mean_se[1:]),
        np.mean(mean_sp[1:])
    ))

    print(Fore.YELLOW + "{:<10} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f}".format(
        "avg std",
        np.mean(std_dice[1:]),
        np.mean(std_hdis[1:]),
        np.mean(std_iou[1:]),
        np.mean(std_acc[1:]),
        np.mean(std_se[1:]),
        np.mean(std_sp[1:])
    ))
    print(Fore.MAGENTA + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description='mode')
    parser.add_argument('--mode', default='train', type=str, help='mode of model, e.g., train, test')
    parser.add_argument('--isInfer', default=False, type=bool, help='is infer model')
    args = parser.parse_args()
    cfg = Config()
    opt = parse_opt(cfg)
    device = setup_environment(opt)

    model = prepare_model(cfg, opt, device)
    criterion = get_pre_seg_criterion(dice_weight=0.75)  # 0.7,0.75,0.8
    # criterion = CombinedLoss(alpha=0.5, beta=0.2, gamma=0.3)
    if args.mode == 'train':

        TensorWriter = None
        if cfg.keep_log:
            logtimestr = time.strftime('%m%d%H%M')
            boardpath = f"{cfg.save_dir}/logs/{logtimestr}"
            os.makedirs(boardpath, exist_ok=True)
            TensorWriter = SummaryWriter(boardpath)

        if cfg.auto_split:
            split_path = os.path.join(opt.data_subpath, "splits", cfg.split_output_dir)
        else:
            split_path = os.path.join(opt.data_subpath, "splits")

        train_loader, val_loader = coarse_seg_prepare_data(cfg, opt, split_path)

        optimizer, scheduler, plateau_scheduler = prepare_optimizer_scheduler(cfg, model, train_loader)
        best_dice = 0.0
        best_dice_file = None
        min_val_loss = float('inf')
        min_val_loss_file = None
        no_improvement_epochs = 0
        early_stopping = False

        for epoch in range(opt.epochs):

            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch {epoch + 1}/{opt.epochs}, lr: {current_lr:.6f}')
            if early_stopping:
                print("early_stopping")
                break

            train_loss_avg, lr_history = train_epoch(
                model, train_loader, optimizer, criterion, device, epoch, TensorWriter
            )

            if cfg.is_active_lr:
                scheduler.step()

            if epoch % opt.eval_freq == 0:
                mean_dice, mean_hdis, val_loss = validate_model(model, val_loader, criterion, opt, cfg, epoch,
                                                                TensorWriter)

                if cfg.is_active_lr and epoch >= cfg.cosine_epochs:
                    plateau_scheduler.step(mean_dice)

                if mean_dice > best_dice:
                    if best_dice_file:
                        os.remove(best_dice_file)
                    timestr = time.strftime('%m%d%H%M')
                    best_dice_file = save_model(
                        model,
                        os.path.join(cfg.save_dir, 'checkpoints',
                                     f"{cfg.model_name}{opt.save_path_code}_{timestr}_epoch{epoch}_dicemax_{mean_dice:.4f}.pth"),
                        epoch,
                        "Dice",
                        mean_dice
                    )
                    best_dice = mean_dice
                    no_improvement_epochs = 0
                else:
                    no_improvement_epochs += 1

                current_lr = optimizer.param_groups[0]['lr']
                if current_lr <= cfg.min_lr and no_improvement_epochs >= cfg.early_stopping_patience:
                    early_stopping = True
            print("-------------------------------------------")
        print("Complete!")
        print("--------------------------------------------------------------")
    if args.mode == 'test':
        print("====================test==========================")
        if cfg.auto_split:
            split_path = os.path.join(opt.data_subpath, "splits", cfg.split_output_dir)
        else:
            split_path = os.path.join(opt.data_subpath, "splits")
        if args.isInfer:
            test_loader, train_loader, val_loader = coarse_seg_prepare_test_data(cfg, opt, args.isInfer, split_path)
            test_model(model, train_loader, cfg, opt, criterion, isInfer=args.isInfer)
            test_model(model, val_loader, cfg, opt, criterion, isInfer=args.isInfer)
        else:
            test_loader = coarse_seg_prepare_test_data(cfg, opt, args.isInfer, split_path)
        test_model(model, test_loader, cfg, opt, criterion, isInfer=False)


if __name__ == '__main__':
    main()
