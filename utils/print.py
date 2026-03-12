from termcolor import colored
from colorama import init, Fore, Style  # Import colorama library
import numpy as np
# Initialize colorama
init(autoreset=True)


def print_arguments(args, opt=None):
    """Print parameter information in a formatted table"""
    # Define title and separator
    title = "Parameter Information"
    line_length = 60
    separator = "=" * line_length

    # Create parameter list
    param_list = [
        ("Model Name", args.modelname),
        ("Encoder Input Size", args.encoder_input_size),
        ("Low Resolution Image Size", args.low_image_size),
        ("Task Name", args.task),
        ("ViT Model Name", args.vit_name),
        ("Batch Size", args.batch_size),
        ("Number of GPUs", args.n_gpu),
        ("Base Learning Rate", args.base_lr),
        ("Minimum Learning Rate", args.min_lr),
        ("Learning Rate Decay Factor", args.reduce_factor),
        ("Plateau Decay Patience", args.patience),
        ("Cosine Annealing Period", f"{args.cosine_epochs} epochs"),
        ("Use Warmup", args.warmup),
        ("Warmup Period", args.warmup_period),
        ("Save Logs", args.keep_log),
        ("Early Stopping Patience", args.early_stopping_patience),
        ("Enable Dynamic Learning Rate", args.is_active_lr),
        ("Use Coarse Mask", args.use_coarse_mask),
        ("Pre-segmentation Result Path", args.preseg_path),
        ("Generation Result Name", args.gen_name)
    ]
    param_list.insert(5, ("Pretrained Checkpoint Path", args.sam_ckpt))

    # Calculate maximum key length for alignment
    max_key_length = max(len(key) for key, _ in param_list)

    # Print formatted parameter information
    print(colored(separator.center(line_length), 'cyan'))
    print(colored(title.center(line_length), 'cyan', attrs=['bold']))
    print(colored(separator.center(line_length), 'cyan'))

    for key, value in param_list:
        # Align with colon
        line = f"{key.ljust(max_key_length)} : {value}"
        # Color boolean and special values
        if isinstance(value, bool):
            line = colored(line, 'yellow' if value else 'red')
        elif key in ["Base Learning Rate", "Minimum Learning Rate", "Learning Rate Decay Factor"]:
            line = colored(line, 'green')
        print(line)

    print(colored(separator.center(line_length), 'cyan'))


def printMethod(args, opt, mean_dice, mean_hdis, mean_iou, mean_acc, mean_se, mean_sp,
                std_dice, std_hdis, std_iou, std_acc, std_se, std_sp):
    """
    Print evaluation results and save to file, add FPS and memory usage display
    """
    print(Fore.MAGENTA + "=" * 80)
    print(Fore.MAGENTA + f"Dataset: {args.task} | Model: {args.modelname}")
    print(Fore.MAGENTA + f"Checkpoint: {opt.load_path}")

    # Print class metrics
    print(Fore.BLUE + "{:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
        "Metric Value", "Dice", "HD", "IoU", "Acc", "Se", "Sp"
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

    # Print standard deviation
    print(Fore.GREEN + "{:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
        "Std Dev", "Dice", "HD", "IoU", "Acc", "Se", "Sp"
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

    # Print averages
    print(Fore.YELLOW + "{:<10} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f}".format(
        "Average Metrics",
        np.mean(mean_dice[1:]),
        np.mean(mean_hdis[1:]),
        np.mean(mean_iou[1:]),
        np.mean(mean_acc[1:]),
        np.mean(mean_se[1:]),
        np.mean(mean_sp[1:])
    ))

    # Print average standard deviation
    print(Fore.YELLOW + "{:<10} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f}".format(
        "Average Std Dev",
        np.mean(std_dice[1:]),
        np.mean(std_hdis[1:]),
        np.mean(std_iou[1:]),
        np.mean(std_acc[1:]),
        np.mean(std_se[1:]),
        np.mean(std_sp[1:])
    ))

    # Add performance information to file output
    with open("experiments.txt", "a+") as file:
        performance_info = f"{args.task} {args.modelname}-pt10 "
        performance_info += f"{'%.2f' % (mean_dice[1])}±{'%.2f' % std_dice[1]} "
        performance_info += f"{'%.2f' % mean_hdis[1]}±{'%.2f' % std_hdis[1]} "
        performance_info += f"{'%.2f' % (mean_iou[1])}±{'%.2f' % std_iou[1]} "
        performance_info += f"{'%.2f' % (mean_acc[1])}±{'%.2f' % std_acc[1]} "
        performance_info += f"{'%.2f' % (mean_se[1])}±{'%.2f' % std_se[1]} "
        performance_info += f"{'%.2f' % (mean_sp[1])}±{'%.2f' % std_sp[1]}"

        file.write(performance_info + "\n")

    print(Fore.MAGENTA + "=" * 80)
