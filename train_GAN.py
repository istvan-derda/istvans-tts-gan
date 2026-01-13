from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cfg
from dataLoader import *
from GANModels import * 
from functions import train, LinearLrDecay, copy_params, cur_stages
from utils.utils import set_log_dir

import torch
from torch.utils import data
import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy
from adamw import AdamW
import random 
from torchvision.transforms import ToTensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    train_data = load_train_data()

    train_tts_gan(train_data, patch_size=15, seq_len=150, in_channels=3)


def load_train_data():
    args = cfg.parse_args()
    data = unimib_load_dataset(incl_xyz_accel = True, incl_rms_accel = False, incl_val_group = False, is_normalize = True, one_hot_encode = False, data_mode = 'Train', single_class = True, class_name = args.class_name, augment_times=args.augment_times)
    data = [datapoint for datapoint, _ in data]
    return np.array(data)

def train_tts_gan(train_data, patch_size, seq_len, in_channels):
    args = cfg.parse_args()
    print(train_data.shape)
    
    if args.seed is not None:
        torch.manual_seed(args.random_seed)
        torch.cuda.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)
        np.random.seed(args.random_seed)
        random.seed(args.random_seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # import network
    
    gen_net = Generator(patch_size=patch_size, seq_len=seq_len, channels=in_channels, latent_dim=args.latent_dim)
    dis_net = Discriminator(patch_size=patch_size, seq_length=seq_len, in_channels=in_channels)
    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    else:
        gen_net = torch.nn.DataParallel(gen_net).cuda()
        dis_net = torch.nn.DataParallel(dis_net).cuda()
        

    # set optimizer
    if args.optimizer == "adam":
        gen_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, gen_net.parameters()),
                                        args.g_lr, (args.beta1, args.beta2))
        dis_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, dis_net.parameters()),
                                        args.d_lr, (args.beta1, args.beta2))
    elif args.optimizer == "adamw":
        gen_optimizer = AdamW(filter(lambda p: p.requires_grad, gen_net.parameters()),
                                        args.g_lr, weight_decay=args.wd)
        dis_optimizer = AdamW(filter(lambda p: p.requires_grad, dis_net.parameters()),
                                         args.g_lr, weight_decay=args.wd)
        
    gen_scheduler = LinearLrDecay(gen_optimizer, args.g_lr, 0.0, 0, args.max_iter)
    dis_scheduler = LinearLrDecay(dis_optimizer, args.d_lr, 0.0, 0, args.max_iter)

    train_loader = data.DataLoader(train_data, batch_size=args.batch_size, num_workers=2, shuffle = True)
 
    args.max_epoch = np.ceil(args.max_iter / len(train_loader))

    # initial
    avg_gen_net = deepcopy(gen_net).cpu()
    gen_avg_param = copy_params(avg_gen_net)
    del avg_gen_net
    start_epoch = 0
    best_fid = 1e4

    # set writer
    writer = None
    if args.load_path:
        print(f'=> resuming from {args.load_path}')
        assert os.path.exists(args.load_path)
        checkpoint_file = os.path.join(args.load_path)
        assert os.path.exists(checkpoint_file)
        loc = 'cuda:{}'.format(None)
        checkpoint = torch.load(checkpoint_file, map_location=loc)
        start_epoch = checkpoint['epoch']
        best_fid = checkpoint['best_fid']
        
        
        dis_net.load_state_dict(checkpoint['dis_state_dict'])
        gen_optimizer.load_state_dict(checkpoint['gen_optimizer'])
        dis_optimizer.load_state_dict(checkpoint['dis_optimizer'])
        
        gen_net.load_state_dict(checkpoint['avg_gen_state_dict'])
        gen_avg_param = copy_params(gen_net, mode='gpu')
        gen_net.load_state_dict(checkpoint['gen_state_dict'])

        args.path_helper = checkpoint['path_helper']
        print(f'=> loaded checkpoint {checkpoint_file} (epoch {start_epoch})')
        writer = SummaryWriter(args.path_helper['log_path'])
        del checkpoint
    else:
    # create new log dir
        assert args.exp_name
        args.path_helper = set_log_dir('logs', args.exp_name)
        writer = SummaryWriter(args.path_helper['log_path'])
    
    writer_dict = {
        'writer': writer,
        'train_global_steps': start_epoch * len(train_loader),
        'valid_global_steps': start_epoch // args.val_freq,
    }

    # train loop
    for epoch in range(int(start_epoch), int(args.max_epoch)):
        lr_schedulers = (gen_scheduler, dis_scheduler) if args.lr_decay else None
        cur_stage = cur_stages(epoch, args)
        print("cur_stage " + str(cur_stage))
        print(f"path: {args.path_helper['prefix']}")
        
        train(args, gen_net, dis_net, gen_optimizer, dis_optimizer, gen_avg_param, train_loader, epoch, writer_dict, lr_schedulers)
        
        gen_net.eval()

    return gen_net


if __name__ == '__main__':
    main()
