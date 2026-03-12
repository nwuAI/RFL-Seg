# split_dataset.py
from utils.data_us import JointTransform2D, ImageToImage2D, PreSegImageToImage2D
from torch.utils.data import DataLoader
import os

def split_dataset_by_dataset_type(dataset_path, dataset_name, output_dir='./splits/'):
    """
    Split dataset into training, validation, and test sets based on dataset type

    Parameters:
    - dataset_path: Root path of the dataset
    - dataset_name: Name of the dataset
    - output_dir: Path to save the split results

    Returns:
    - train_file, val_file, test_file: Paths of the split files
    """
    import os
    import random
    from glob import glob

    # Get all image file names (without extension)
    img_dir = os.path.join(dataset_path, 'img')
    label_dir = os.path.join(dataset_path, 'label')
    img_files = glob(os.path.join(img_dir, '*.png'))
    label_files = glob(os.path.join(label_dir, '*.png'))
    img_names = [os.path.splitext(os.path.basename(f))[0] for f in label_files]

    total_count = len(img_names)
    print(f"Total samples in dataset: {total_count}")

    # Set different splitting strategies based on dataset type
    if dataset_name == 'TN3K':
        # BUSI dataset split: 400 training, 100 validation, remaining for testing
        train_count = 2303
        val_count = 576
        test_count = 614
    else:
        # Default strategy: 70% training, 10% validation, 20% testing
        test_count = int(total_count * 0.2)
        val_count = int(total_count * 0.1)
        train_count = total_count - test_count - val_count

    # Ensure counts do not exceed total
    train_count = min(train_count, total_count)
    val_count = min(val_count, total_count - train_count)
    test_count = min(test_count, total_count - train_count - val_count)

    # Shuffle data randomly
    random.shuffle(img_names)

    # Split datasets
    train_names = img_names[:train_count]
    val_names = img_names[train_count:train_count + val_count]
    test_names = img_names[train_count + val_count:train_count + val_count + test_count]

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save training set
    train_file = f'train-{dataset_name}.txt'
    train_file_path = os.path.join(output_dir, train_file)
    with open(train_file_path, 'w') as f:
        for name in train_names:
            f.write(f'1/{dataset_name}/{name}\n')

    # Save validation set
    val_file = f'val-{dataset_name}.txt'
    val_file_path = os.path.join(output_dir, val_file)
    with open(val_file_path, 'w') as f:
        for name in val_names:
            f.write(f'1/{dataset_name}/{name}\n')

    # Save test set
    test_file = f'test-{dataset_name}.txt'
    test_file_path = os.path.join(output_dir, test_file)
    with open(test_file_path, 'w') as f:
        for name in test_names:
            f.write(f'{dataset_name}/{name}\n')

    # Output statistics
    actual_total = len(train_names) + len(val_names) + len(test_names)
    print(f"Dataset split completed ({dataset_name}):")
    print(f"  Training set: {len(train_names)} samples")
    print(f"  Validation set: {len(val_names)} samples")
    print(f"  Test set: {len(test_names)} samples")
    print(f"  Total: {actual_total} samples")
    print(f"  Split files saved to: {output_dir}")

    return train_file, val_file, test_file


def prepare_data_with_split(args, opt, split_output_dir='./splits/'):
    """
    Prepare training and validation datasets with dynamic splitting

    Parameters:
    - args: Command line arguments
    - opt: Configuration object
    - split_output_dir: Path to save split files
    """
    # If automatic dataset splitting is enabled (should check existing files during inference)
    if args.auto_split:
        # Check if pre-generated split files exist
        dataset_name = args.task
        common(opt, dataset_name, split_output_dir)

    tf_train = JointTransform2D(
        img_size=args.encoder_input_size,
        low_img_size=args.low_image_size,
        ori_size=opt.img_size,
        crop=opt.crop,
        p_flip=0.0,  # Horizontal flip
        p_rota=0.5,  # Rotation
        p_scale=0.5,  # Scaling
        p_gaussn=0.0,  # Gaussian noise
        p_contr=0.5,  # Contrast
        p_gama=0.5,  # Gamma correction
        p_distor=0.0,  # Distortion transform
        color_jitter_params=None,
        long_mask=True
    )

    tf_val = JointTransform2D(
        img_size=args.encoder_input_size,
        low_img_size=args.low_image_size,
        ori_size=opt.img_size,
        crop=opt.crop,
        p_flip=0,
        color_jitter_params=None,
        long_mask=True
    )

    train_dataset = ImageToImage2D(
        opt.data_path,
        split_output_dir,
        opt.train_split,
        tf_train,
        img_size=args.encoder_input_size,
        use_coarse_mask=args.use_coarse_mask,
        preseg_path=args.preseg_path,
        gen_name=args.gen_name,
        max_prompts=10
    )

    val_dataset = ImageToImage2D(
        opt.data_path,
        split_output_dir,
        opt.val_split,
        tf_val,
        img_size=args.encoder_input_size,
        use_coarse_mask=args.use_coarse_mask,
        preseg_path=args.preseg_path,
        gen_name=args.gen_name,
        max_prompts=10
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    print(f"Training set sample count: {len(train_dataset)}")
    print(f"Validation set sample count: {len(val_dataset)}")
    print(f"Iterations per epoch: {len(train_loader)}")

    return train_loader, val_loader


def prepare_data_test(args, opt, split_output_dir='./splits/'):
    """
    Prepare dataset and data loader for testing/inference phase
    """

    opt.batch_size = args.batch_size * args.n_gpu

    # If automatic dataset splitting is enabled (should check existing files during inference)
    if args.auto_split:
        # Check if pre-generated split files exist
        dataset_name = args.task
        common(opt, dataset_name, split_output_dir)

    tf_test = JointTransform2D(img_size=args.encoder_input_size, low_img_size=args.low_image_size,
                               ori_size=opt.img_size,
                               crop=opt.crop, p_flip=0, color_jitter_params=None, long_mask=True)
    test_dataset = ImageToImage2D(opt.data_path, split_output_dir, opt.test_split, tf_test, img_size=args.encoder_input_size,
                                  use_coarse_mask=args.use_coarse_mask, preseg_path=args.preseg_path,
                                  gen_name=args.gen_name, class_id=1, max_prompts=3)
    testloader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    if args.isFineIterate:
        tf_train = JointTransform2D(img_size=args.encoder_input_size, low_img_size=args.low_image_size,
                                    ori_size=opt.img_size, crop=opt.crop, p_flip=0, color_jitter_params=None,
                                    long_mask=True)
        train_dataset = ImageToImage2D(opt.data_path, split_output_dir, opt.train_split, tf_train, img_size=args.encoder_input_size,
                                       use_coarse_mask=args.use_coarse_mask, preseg_path=args.preseg_path,
                                       gen_name=args.gen_name, class_id=1, max_prompts=3)
        trainloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=8,
                                 pin_memory=True)
        tf_val = JointTransform2D(img_size=args.encoder_input_size, low_img_size=args.low_image_size,
                                  ori_size=opt.img_size, crop=opt.crop, p_flip=0, color_jitter_params=None,
                                  long_mask=True)
        val_dataset = ImageToImage2D(opt.data_path, split_output_dir, opt.val_split, tf_val, img_size=args.encoder_input_size,
                                     use_coarse_mask=args.use_coarse_mask, preseg_path=args.preseg_path,
                                     gen_name=args.gen_name, class_id=1, max_prompts=3)
        valloader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=8, pin_memory=True)
        return trainloader, valloader, testloader
    return None, None, testloader


def coarse_seg_prepare_data(cfg, opt, split_output_dir='./splits/'):
    """Prepare training and validation datasets with dynamic splitting"""
    # If automatic dataset splitting is enabled (should check existing files during inference)
    if cfg.auto_split:
        # Check if pre-generated split files exist
        dataset_name = cfg.task
        common(opt, dataset_name, split_output_dir)

    tf_train = JointTransform2D(
        img_size=opt.img_size,
        low_img_size=opt.img_size // 2,
        ori_size=opt.img_size,
        crop=opt.crop,
        p_flip=0.5,
        p_rota=0.5,
        p_scale=0.5,
        p_gaussn=0.0,
        p_contr=0.5,
        p_gama=0.5,
        p_distor=0.5,
        color_jitter_params=None,
        long_mask=True
    )

    tf_val = JointTransform2D(
        img_size=opt.img_size,
        low_img_size=opt.img_size // 2,
        ori_size=opt.img_size,
        crop=opt.crop,
        p_flip=0,
        color_jitter_params=None,
        long_mask=True
    )

    train_dataset = PreSegImageToImage2D(
        opt.data_path,
        split_output_dir,
        opt.train_split,
        tf_train,
        img_size=opt.img_size
    )

    val_dataset = PreSegImageToImage2D(
        opt.data_path,
        split_output_dir,
        opt.val_split,
        tf_val,
        img_size=opt.img_size
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    print(f"Training set sample count: {len(train_dataset)}")
    print(f"Validation set sample count: {len(val_dataset)}")
    print(f"Iterations per epoch: {len(train_loader)}")

    return train_loader, val_loader


def coarse_seg_prepare_test_data(cfg, opt, isInfer, split_output_dir='./splits/'):
    """Prepare test dataset with dynamic splitting"""
    # If automatic dataset splitting is enabled (should check existing files during inference)
    if cfg.auto_split:
        # Check if pre-generated split files exist
        dataset_name = cfg.task
        common(opt, dataset_name, split_output_dir)

    tf_test = JointTransform2D(img_size=opt.img_size, low_img_size=opt.img_size // 2, ori_size=opt.img_size,
                               crop=opt.crop, p_flip=0, color_jitter_params=None, long_mask=True)
    test_dataset = PreSegImageToImage2D(opt.data_path, split_output_dir, opt.test_split, tf_test, img_size=opt.img_size)
    testloader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    if isInfer:
        tf_train = JointTransform2D(img_size=opt.img_size, low_img_size=opt.img_size // 2, ori_size=opt.img_size,
                                    crop=opt.crop, p_flip=0, color_jitter_params=None, long_mask=True)
        train_dataset = PreSegImageToImage2D(opt.data_path, split_output_dir, opt.train_split, tf_train, img_size=opt.img_size)
        trainloader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=8,
                                 pin_memory=True)
        tf_val = JointTransform2D(img_size=opt.img_size, low_img_size=opt.img_size // 2, ori_size=opt.img_size,
                                  crop=opt.crop, p_flip=0, color_jitter_params=None, long_mask=True)
        val_dataset = PreSegImageToImage2D(opt.data_path, split_output_dir, opt.val_split, tf_val, img_size=opt.img_size)
        valloader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=8, pin_memory=True)
        return testloader, trainloader, valloader
    else:
        print(f"Test set sample count: {len(test_dataset)}")
        return testloader

def common(opt, dataset_name, split_output_dir):
    expected_train_file = f'train-{dataset_name}'
    expected_val_file = f'val-{dataset_name}'
    expected_test_file = f'test-{dataset_name}'
    expected_train_file_path = os.path.join(split_output_dir, f'train-{dataset_name}.txt')
    expected_val_file_path = os.path.join(split_output_dir, f'val-{dataset_name}.txt')
    expected_test_file_path = os.path.join(split_output_dir, f'test-{dataset_name}.txt')

    # Check if all required split files exist
    missing_files = []
    if not os.path.exists(expected_train_file_path):
        missing_files.append(expected_train_file_path)
    if not os.path.exists(expected_val_file_path):
        missing_files.append(expected_val_file_path)
    if not os.path.exists(expected_test_file_path):
        missing_files.append(expected_test_file_path)

    if missing_files:
        raise FileNotFoundError(
            f"Required dataset split files not found: {missing_files}.\n"
            f"Please run the dataset splitting function first to generate these files, or disable the auto_split feature."
        )

    # Use existing split files
    opt.train_split = expected_train_file
    opt.val_split = expected_val_file
    opt.test_split = expected_test_file

    print(f"Using existing split files:")
    print(f"  Training set: {opt.train_split}")
    print(f"  Validation set: {opt.val_split}")
    print(f"  Test set: {opt.test_split}")


