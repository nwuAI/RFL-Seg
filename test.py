import os
from utils.split_dataset import prepare_data_test
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import argparse
import torch.optim as optim
import numpy as np
import torch
import random
from utils.config import get_config
from utils.evaluation import get_eval
from models.model_dict import get_model
from utils.loss_functions.sam_loss import get_criterion
from thop import profile
from colorama import init, Fore, Style
from utils.print import printMethod
init(autoreset=True)

def parse_arguments():

    parser = argparse.ArgumentParser(description='Networks')
    parser.add_argument('--modelname', default='RFLSeg', type=str )
    parser.add_argument('-encoder_input_size', type=int, default=256)
    parser.add_argument('-low_image_size', type=int, default=128)
    parser.add_argument('--task', default='BUSI', help='task or dataset name')
    parser.add_argument('--vit_name', type=str, default='vit_b',
                        help='select the vit model for the image encoder of sam')
    parser.add_argument('--sam_ckpt', type=str, default='checkpoints/sam_vit_b_01ec64.pth',
                        help='Pretrained checkpoint of SAM')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='batch_size per gpu')
    parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
    parser.add_argument('--base_lr', type=float, default=0.0005,
                        help='segmentation network learning rate, 0.005 for SAMed, 0.0001 for MSA')
    parser.add_argument('--warmup', type=bool, default=False,
                        help='If activated, warp up the learning from a lower lr to the base_lr')
    parser.add_argument('--warmup_period', type=int, default=250,
                        help='Warp up iterations, only valid whrn warmup is activated')
    parser.add_argument('-keep_log', type=bool, default=False, help='keep the loss&lr&dice during training or not')
    parser.add_argument('--isFineIterate', type=bool, default=False, help='False:test fine mask; True: all fine mask')
    parser.add_argument('--use_coarse_mask', type=bool, default=True, help='True:coarse_mask; False:fine_mask')
    parser.add_argument('--preseg_path', type=str, default='./output/BUSI/deeplabv3+_resnet50/')
    parser.add_argument('--gen_name', type=str, default='coarse_gen')
    parser.add_argument('--load_cp', type=str, default='')
    parser.add_argument('--auto_split', type=bool, default=False)
    parser.add_argument('--split_name', type=str, default='0')

    return parser.parse_args()


def set_seed(seed_value=300):

    np.random.seed(seed_value)  # set random seed for numpy
    random.seed(seed_value)  # set random seed for python
    os.environ['PYTHONHASHSEED'] = str(seed_value)  # avoid hash random
    torch.manual_seed(seed_value)  # set random seed for CPU
    torch.cuda.manual_seed(seed_value)  # set random seed for one GPU
    torch.cuda.manual_seed_all(seed_value)  # set random seed for all GPU
    torch.backends.cudnn.deterministic = True  # set random seed for convolution

def prepare_model(args, opt):
    model = get_model(args.modelname, args=args, opt=opt)
    device = torch.device(opt.device)
    model.to(device)
    model.train()

    checkpoint = torch.load(opt.load_path)
    new_state_dict = {}
    for k, v in checkpoint.items():
        if k[:7] == 'module.':
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)

    optimizer = optim.Adam(model.parameters(), lr=args.base_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0,
                           amsgrad=False)
    criterion = get_criterion(modelname=args.modelname, opt=opt)

    return model, optimizer, criterion, device


def evaluate_model(model, testloader, trainloader, valloader, criterion, opt, args):

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(Fore.CYAN + f"Total_params: {pytorch_total_params}")

    input = torch.randn(1, 1, args.encoder_input_size, args.encoder_input_size).cuda()
    low_coarse_masks = torch.randn(1, 1, args.low_image_size, args.low_image_size).cuda()
    gen_input = torch.randn(1, 1, args.encoder_input_size, args.encoder_input_size).cuda()
    points = (torch.tensor([[[1, 2]]]).float().cuda(), torch.tensor([[1]]).float().cuda())
    boxes = torch.tensor([[[10, 100, 20, 200]]]).float().cuda()

    # flops, params = profile(model, inputs=(input, gen_input, points, boxes, low_coarse_masks), )
    # print(Fore.CYAN + f'Gflops: {flops / 1000000000}, params: {params}')

    model.eval()

    if opt.mode == "train":
        dices, mean_dice, _, test_losses = get_eval(testloader, model, criterion=criterion, opt=opt, args=args)
        print(Fore.YELLOW + f"mean dice: {mean_dice}")
    else:
        if args.isFineIterate:
            get_eval(trainloader, model, criterion=criterion, opt=opt, args=args, isFineIterate=True)
            get_eval(valloader, model, criterion=criterion, opt=opt, args=args, isFineIterate=True)
            mean_dice, mean_hdis, mean_iou, mean_acc, mean_se, mean_sp, std_dice, std_hdis, std_iou, std_acc, std_se, std_sp = get_eval(
                testloader, model, criterion=criterion, opt=opt, args=args, isFineIterate=False)
        else:
            mean_dice, mean_hdis, mean_iou, mean_acc, mean_se, mean_sp, std_dice, std_hdis, std_iou, std_acc, std_se, std_sp = get_eval(
                testloader, model, criterion=criterion, opt=opt, args=args, isFineIterate=False)

        printMethod(args, opt, mean_dice, mean_hdis, mean_iou, mean_acc, mean_se, mean_sp,
                    std_dice, std_hdis, std_iou, std_acc, std_se, std_sp)

def main():
    args = parse_arguments()
    opt = get_config(args.task)  # please configure your hyper-parameter
    opt.mode = "test"
    opt.visual = True
    opt.result_path = args.preseg_path
    opt.load_path = args.preseg_path + "sam_checkpoints/" + args.load_cp
    opt.modelname = args.modelname
    opt.isFineIterate = args.isFineIterate

    set_seed()
    if args.auto_split:
        split_path = os.path.join(opt.data_subpath, "splits", args.split_name)
    else:
        split_path = os.path.join(opt.data_subpath, "splits")
    trainloader, valloader, testloader = prepare_data_test(args, opt, split_path)
    model, optimizer, criterion, device = prepare_model(args, opt)
    evaluate_model(model, testloader, trainloader, valloader, criterion, opt, args)


if __name__ == '__main__':
    main()