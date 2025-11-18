import torch
import torch.nn as nn
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
    "data": "./data/train/",       # positional "data" (you must set this)

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
    "no_aug": False,
    "repeated_aug": False,
    "scale": [0.08, 1.0],
    "ratio": [3.0 / 4.0, 4.0 / 3.0],
    "hflip": 0.5,
    "vflip": 0.0,
    "color_jitter": 0.4,
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
    "mixup_prob": 1.0,
    "mixup_switch_prob": 0.5,
    "mixup_mode": "batch",
    "mixup_off_epoch": 0,
    "smoothing": 0.1,
    "train_interpolation": "random",
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

def train_data():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    opt = OptInit()
    model = DeepGCN(opt=opt)
    model.to(device)

    data_config = resolve_data_config(args, model=model, verbose=args["local_rank"] == 0)
    print("Input size data_config: ", data_config['input_size'])
    train_dir = "./data/train/"    
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

    train_loss_fn = nn.CrossEntropyLoss().cuda()

    args_ns = SimpleNamespace(**args)
    optimizer = create_optimizer(args_ns, model)

    # Training
    EPOCH_MAX = 10
    for epoch in range(0, EPOCH_MAX):
        for batch_idx, (input, target) in enumerate(loader_train):      
            with suppress():
                output = model(input)
                loss = train_loss_fn(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Test
    model.eval()
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

    output = model(test_data)
    print("Output shape:", output.shape)
    indices = torch.argmax(output, dim=1)  
    print("Predicted class indices per image:", indices)
    print("Targets:", test_targets)

def inference():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    img_path = 'fig/ILSVRC2012_val_00000075.JPEG'
    img = Image.open(img_path).resize((224, 224))    
    img_data = torch.unsqueeze(pre_transforms(img), 0)
    #img_data = img_data.repeat(5, 1, 1, 1)
    img_data = img_data.to(device)
    #print(img_data.shape)
    
    #img_data = torch.rand(5, 3, 224, 224).to(device)
    #stem = Stem().to(device)
    #a = stem(img_data)
    #print(a.shape)

    opt = OptInit()
    model = DeepGCN(opt=opt)
    #print(model)
    model.to(device)
    model.eval()
    output = model(img_data)

    print("Output shape:", output.shape)
    indices = torch.argmax(output, dim=1)  
    print("Predicted class indices per image:", indices)

if __name__ == "__main__":

    train_data()
    exit(1)





