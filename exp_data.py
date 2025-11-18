import torch
import torch.nn as nn
from torchvision.utils import save_image
from PIL import Image
from timm.data import Dataset, resolve_data_config
from timm.optim import create_optimizer

import os
from types import SimpleNamespace
from contextlib import suppress

from vig import Stem, DeepGCN
from visualizer_utils import pre_transforms
from data.myloader import create_loader

# A big dict of "parsed args" with defaults.
args = {
    "distributed" : False,

    # Config / data
    "config": "",                          # -c / --config
    "data": "./data/",       # positional "data" (you must set this)

    # Dataset / model
    "model": "vig_ti_224_gelu",
    "pretrained": False,
    "initial_checkpoint": "",
    "resume": "",
    "no_resume_opt": False,
    "num_classes": 1000,
    "gp": None,
    "img_size": None,
    "crop_pct": None,
    "mean": None,
    "std": None,
    "interpolation": "",
    "batch_size": 2,
    "validation_batch_size_multiplier": 1,

    # Optimizer
    "opt": "sgd",
    "opt_eps": None,
    "opt_betas": None,
    "momentum": 0.9,
    "weight_decay": 0.0001,
    "clip_grad": None,

    # LR schedule
    "sched": "step",
    "lr": 0.01,
    "lr_noise": None,
    "lr_noise_pct": 0.67,
    "lr_noise_std": 1.0,
    "lr_cycle_mul": 1.0,
    "lr_cycle_limit": 1,
    "warmup_lr": 0.0001,
    "min_lr": 1e-5,
    "epochs": 200,
    "start_epoch": None,
    "decay_epochs": 30.0,
    "warmup_epochs": 3,
    "cooldown_epochs": 10,
    "patience_epochs": 10,
    "decay_rate": 0.1,

    # Augmentation & regularization
    "no_aug": True,
    "repeated_aug": False,
    "scale": [1.0, 1.0],
    "ratio": [1.0, 1.0],
    "hflip": 0.0,
    "vflip": 0.0,
    "color_jitter": 0.0,
    "aa": None,
    "aug_splits": 0,
    "jsd": False,
    "reprob": 0.0,
    "remode": "const",
    "recount": 1,
    "resplit": False,
    "mixup": 0.0,
    "cutmix": 0.0,
    "cutmix_minmax": None,
    "mixup_prob": 0.0,
    "mixup_switch_prob": 0.0,
    "mixup_mode": "batch",
    "mixup_off_epoch": 0,
    "smoothing": 0.0,
    "train_interpolation": None,
    "drop": 0.0,
    "drop_connect": None,
    "drop_path": None,
    "drop_block": None,

    # BatchNorm
    "bn_tf": False,
    "bn_momentum": None,
    "bn_eps": None,
    "sync_bn": False,
    "dist_bn": "",
    "split_bn": False,

    # EMA
    "model_ema": False,
    "model_ema_force_cpu": False,
    "model_ema_decay": 0.9998,

    # Misc
    "seed": 42,
    "log_interval": 50,
    "recovery_interval": 0,
    "workers": 1,
    "num_gpu": 1,
    "save_images": False,
    "amp": False,
    "apex_amp": False,
    "native_amp": False,
    "channels_last": False,
    "pin_mem": False,
    "no_prefetcher": False,
    "output": "",
    "eval_metric": "top1",
    "tta": 0,
    "local_rank": 0,
    "use_multi_epochs_loader": False,
    "init_method": "env://",
    "train_url": None,

    # Newly added
    "attn_ratio": 1.0,
    "pretrain_path": None,
    "evaluate": False,

    # Wandb
    "log_wandb": True,
}

class OptInit:
        def __init__(self, num_classes=1000, drop_path_rate=0.0, drop_rate=0.0, num_knn=9, **kwargs):
            self.k = num_knn # neighbor num (default:9)
            self.conv = 'mr' # graph conv layer {edge, mr}
            self.act = 'gelu' # activation layer {relu, prelu, leakyrelu, gelu, hswish}
            self.norm = 'batch' # batch or instance normalization {batch, instance}
            self.bias = True # bias of conv layer True or False
            self.n_blocks = 12 # number of basic blocks in the backbone
            self.n_filters = 192 # number of channels of deep features
            self.n_classes = num_classes # Dimension of out_channels
            self.dropout = drop_rate # dropout rate
            self.use_dilation = False # use dilated knn or not
            self.epsilon = 0.2 # stochastic epsilon for gcn
            self.use_stochastic = False # stochastic for gcn, True or False
            self.drop_path = drop_path_rate



def load_val_data(out_dir="./outputs/data/val"):
    # Device Setting
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Model Initialization
    opt = OptInit()
    model = DeepGCN(opt=opt)
    model.to(device)

    # Validation Data 
    data_config = resolve_data_config(args, model=model, verbose=args["local_rank"] == 0)
    val_dir = "./data/val/"    # /stratch/dataset/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/val/
    if not os.path.exists(val_dir):
        print('Validation folder does not exist at: {}'.format(val_dir))
        exit(1)

    dataset_val = Dataset(val_dir)
    
    loader_val  = create_loader(
        dataset_val,
        input_size=data_config['input_size'],
        batch_size=args["batch_size"],
        is_training=False,
        use_prefetcher= not args["no_prefetcher"],
        interpolation=data_config['interpolation'],
        mean= data_config['mean'],
        std=data_config['std'],
        num_workers=args["workers"],
        distributed=args["distributed"],
        crop_pct=data_config['crop_pct'],
        pin_memory=args["pin_mem"],
    )

    # Save 5 validation images 
    os.makedirs(out_dir, exist_ok=True)
    for batch_idx, (input, target) in enumerate(loader_val):  
        input.cpu()
        target.cpu()
        print("Val image label: ", target)
        save_image(input[0], f"{out_dir}/image_{batch_idx}.jpg")
        if batch_idx == 5:
            break

def load_train_data(out_dir="./outputs/data/train"):
    # Device Setting
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Model Initialization
    opt = OptInit()
    model = DeepGCN(opt=opt)
    model.to(device)

    # Training Data 
    data_config = resolve_data_config(args, model=model, verbose=args["local_rank"] == 0)
    train_dir = "./data/train/"    
    #train_dir = /stratch/dataset/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train/
    if not os.path.exists(train_dir):
        print('Training folder does not exist at: {}'.format(train_dir))
        exit(1)
    dataset_train = Dataset(train_dir)

    num_aug_splits = 0
    collate_fn = None
    loader_train = create_loader(
        dataset_train,
        input_size=data_config['input_size'],
        batch_size=args["batch_size"],
        is_training=True,
        use_prefetcher= not args["no_prefetcher"],
        no_aug=args["no_aug"],
        re_prob=args["reprob"],
        re_mode=args["remode"],
        re_count=args["recount"],
        re_split=args["resplit"],
        scale=args["scale"],
        ratio=args["ratio"],
        hflip=args["hflip"],
        vflip=args["vflip"],
        color_jitter=args["color_jitter"],
        auto_augment=args["aa"],
        num_aug_splits=num_aug_splits,
        interpolation=args["train_interpolation"],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args["workers"],
        distributed=args["distributed"],
        collate_fn=collate_fn,
        pin_memory=args["pin_mem"],
        use_multi_epochs_loader=args["use_multi_epochs_loader"],
        repeated_aug=args["repeated_aug"]
    )

    # Save 5 training images 
    os.makedirs(out_dir, exist_ok=True)
    for batch_idx, (input, target) in enumerate(loader_train):  
        input.cpu()
        save_image(input[0], f"{out_dir}/image_{batch_idx}.jpg")
        if batch_idx == 5:
            break

def train_data(train_dir="/stratch/dataset/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train/"):
    # Device Setting
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Model Initialization
    opt = OptInit()
    model = DeepGCN(opt=opt)
    model.to(device)

    # Training Data 
    data_config = resolve_data_config(args, model=model, verbose=args["local_rank"] == 0)
       
    if not os.path.exists(train_dir):
        print('Training folder does not exist at: {}'.format(train_dir))
        exit(1)
    dataset_train = Dataset(train_dir)

    num_aug_splits = 0
    collate_fn = None
    loader_train = create_loader(
        dataset_train,
        input_size=data_config['input_size'],
        batch_size=args["batch_size"],
        is_training=True,
        use_prefetcher= not args["no_prefetcher"],
        no_aug=args["no_aug"],
        re_prob=args["reprob"],
        re_mode=args["remode"],
        re_count=args["recount"],
        re_split=args["resplit"],
        scale=args["scale"],
        ratio=args["ratio"],
        hflip=args["hflip"],
        vflip=args["vflip"],
        color_jitter=args["color_jitter"],
        auto_augment=args["aa"],
        num_aug_splits=num_aug_splits,
        interpolation=args["train_interpolation"],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args["workers"],
        distributed=args["distributed"],
        collate_fn=collate_fn,
        pin_memory=args["pin_mem"],
        use_multi_epochs_loader=args["use_multi_epochs_loader"],
        repeated_aug=args["repeated_aug"]
    )


    # Loss formulation & Optimizer
    train_loss_fn = nn.CrossEntropyLoss().cuda()
    args_ns = SimpleNamespace(**args)
    optimizer = create_optimizer(args_ns, model)

    # Training
    EPOCH_MAX = 0
    for epoch in range(0, EPOCH_MAX):
        for batch_idx, (input, target) in enumerate(loader_train):    
            with suppress():
                output = model(input)
                loss = train_loss_fn(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Test Data Preparation
    test_data = []
    test_targets = [] 

    for batch_idx, (input, target) in enumerate(loader_train):  
        input.cpu()
        target.cpu()
        test_data.append(input) 
        test_targets.append(target)

    test_data = torch.cat(test_data, dim=0)
    test_targets = torch.cat(test_targets, dim=0)
    test_data.to(device) 
    test_targets.to(device) 

    # Test Inference
    model.eval()
    output = model(test_data)
    print("Output shape:", output.shape)
    indices = torch.argmax(output, dim=1)  
    print("Predicted class indices per image:", indices)
    print("Targets:", test_targets)


if __name__ == "__main__":
    #load_train_data()
    load_val_data()
    pass