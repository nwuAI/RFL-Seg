import torch

data_root = "./USData/"

# -------------------------------------------------------------------------------------------------
class Config_TN3K:
    data_path = data_root
    data_subpath = data_root + "ThyroidNodule-TN3K/"
    save_path = "./checkpoints/TN3K/"
    result_path = "./result/TN3K/"
    tensorboard_path = "./tensorboard/TN3K/"
    load_path = save_path + "US__07071711_epoch61_lossmin_0.0757.pth"
    save_path_code = "_"

    workers = 1                  # number of data loading workers (default: 8)
    epochs = 200                 # number of total epochs to run (default: 400)
    batch_size = 4               # batch size (default: 4)
    learning_rate = 1e-4         # initial learning rate (default: 0.001)
    momentum = 0.9               # momentum
    classes = 2                  # the number of classes (background + foreground)
    img_size = 256               # the input size of model
    train_split = "train-ThyroidNodule-TN3K"  # the file name of training set
    val_split = "val-ThyroidNodule-TN3K"     # the file name of testing set
    test_split = "test-ThyroidNodule-TN3K"     # the file name of testing set
    crop = None                  # the cropped image size
    eval_freq = 1                # the frequency of evaluate the model
    save_freq = 2000               # the frequency of saving the model
    device = "cuda" if torch.cuda.is_available() else "cpu"              # training device, cpu or cuda
    cuda = "on"                  # switch on/off cuda option (default: off)
    gray = "yes"                 # the type of input image
    img_channel = 1              # the channel of input image
    eval_mode = "mask_slice"        # the mode when evaluate the model, slice level or patient level
    pre_trained = False
    mode = "train"
    visual = False
    modelname = "SAM"
    isFineIterate = False


class Config_DDTI:
    data_path = data_root
    data_subpath = data_root + "ThyroidNodule-DDTI/"
    save_path = "./checkpoints/DDTI/"
    result_path = "./result/DDTI/"
    tensorboard_path = "./tensorboard/DDTI/"
    load_path = save_path + "US__07071711_epoch61_lossmin_0.0757.pth"
    save_path_code = "_"

    workers = 1                  # number of data loading workers (default: 8)
    epochs = 200                 # number of total epochs to run (default: 400)
    batch_size = 4               # batch size (default: 4)
    learning_rate = 1e-4         # initial learning rate (default: 0.001)
    momentum = 0.9               # momentum
    classes = 2                  # the number of classes (background + foreground)
    img_size = 256               # the input size of model
    train_split = "train-ThyroidNodule-DDTI"  # the file name of training set
    val_split = "val-ThyroidNodule-DDTI"     # the file name of testing set
    test_split = "test-ThyroidNodule-DDTI"     # the file name of testing set
    crop = None                  # the cropped image size
    eval_freq = 1                # the frequency of evaluate the model
    save_freq = 2000               # the frequency of saving the model
    device = "cuda" if torch.cuda.is_available() else "cpu"              # training device, cpu or cuda
    cuda = "on"                  # switch on/off cuda option (default: off)
    gray = "yes"                 # the type of input image
    img_channel = 1              # the channel of input image
    eval_mode = "mask_slice"        # the mode when evaluate the model, slice level or patient level
    pre_trained = False
    mode = "train"
    visual = False
    modelname = "SAM"
    isFineIterate = False

class Config_BUSI:
    # This dataset is for breast cancer segmentation

    data_path = data_root
    data_subpath = data_root + "Breast-BUSI/"
    save_path = "./checkpoints/BUSI/"
    result_path = "./result/BUSI/"
    tensorboard_path = "./tensorboard/BUSI/"
    load_path = "./checkpoints/sam_vit_b_01ec64.pth"
    save_path_code = "_"

    workers = 1                         # number of data loading workers (default: 8)
    epochs = 200                        # number of total epochs to run (default: 400)
    batch_size = 4                      # batch size (default: 4)
    learning_rate = 1e-4                # iniial learning rate (default: 0.001)
    momentum = 0.9                      # momntum
    classes = 2                         # thenumber of classes (background + foreground)
    img_size = 256                      # theinput size of model
    train_split = "train-Breast-BUSI"   # the file name of training set
    val_split = "val-Breast-BUSI"       # the file name of testing set
    test_split = "test-Breast-BUSI"     # the file name of testing set
    crop = None                         # the cropped image size
    eval_freq = 1                       # the frequency of evaluate the model
    save_freq = 2000                    # the frequency of saving the model
    device = "cuda" if torch.cuda.is_available() else "cpu"                     # training device, cpu or cuda
    cuda = "on"                         # switch on/off cuda option (default: off)
    gray = "yes"                        # the type of input image
    img_channel = 1                     # the channel of input image
    eval_mode = "mask_slice"                 # the mode when evaluate the model, slice level or patient level
    pre_trained = False
    mode = "train"
    visual = False
    modelname = "SAM"
    isFineIterate = False

class Config_UDIAT:
    # This dataset is for breast cancer segmentation

    data_path = data_root
    data_subpath = data_root + "Breast-UDIAT/"
    save_path = "./checkpoints/UDIAT/"
    result_path = "./result/Breast-UDIAT/"
    tensorboard_path = "./tensorboard/Breast-UDIAT/"
    load_path = "./checkpoints/sam_vit_b_01ec64.pth"
    save_path_code = "_"

    workers = 1                         # number of data loading workers (default: 8)
    epochs = 200                     # number of total epochs to run (default: 400)
    batch_size = 4                     # batch size (default: 4)
    learning_rate = 1e-4                # iniial learning rate (default: 0.001)
    momentum = 0.9                      # momntum
    classes = 2                         # thenumber of classes (background + foreground)
    img_size = 256                      # theinput size of model
    train_split = "train-Breast-UDIAT"   # the file name of training set
    val_split = "val-Breast-UDIAT"       # the file name of testing set
    test_split = "test-Breast-UDIAT"     # the file name of testing set
    crop = None                         # the cropped image size
    eval_freq = 1                       # the frequency of evaluate the model
    save_freq = 2000                    # the frequency of saving the model
    device = "cuda" if torch.cuda.is_available() else "cpu"                     # training device, cpu or cuda
    cuda = "on"                         # switch on/off cuda option (default: off)
    gray = "yes"                        # the type of input image
    img_channel = 1                     # the channel of input image
    eval_mode = "mask_slice"                 # the mode when evaluate the model, slice level or patient level
    pre_trained = False
    mode = "train"
    visual = False
    modelname = "SAM"
    isFineIterate = False


# ==================================================================================================
def get_config(task="BUSI"):
    if task == "TN3K":
        return Config_TN3K()
    elif task == "DDTI":
        return Config_DDTI()
    elif task == "BUSI":
        return Config_BUSI()
    elif task == "UDIAT":
        return Config_UDIAT()
    else:
        assert("We do not have the related dataset, please choose another task.")