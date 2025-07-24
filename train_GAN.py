from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cfg
from dataLoader import *
from GANModels import * 
from functions import train, train_d, validate, save_samples, LinearLrDecay, load_params, copy_params, cur_stages
from utils.utils import set_log_dir, save_checkpoint, create_logger

import torch
import torch.multiprocessing as mp
from torch.utils import data
import os
import numpy as np
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from copy import deepcopy
from adamw import AdamW
import random 
import matplotlib.pyplot as plt
import io
import PIL.Image
from torchvision.transforms import ToTensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    args = cfg.parse_args()
    
    if args.seed is not None:
        torch.manual_seed(args.random_seed)
        torch.cuda.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)
        np.random.seed(args.random_seed)
        random.seed(args.random_seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # weight init
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            if args.init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif args.init_type == 'orth':
                nn.init.orthogonal_(m.weight.data)
            elif args.init_type == 'xavier_uniform':
                nn.init.xavier_uniform(m.weight.data, 1.)
            else:
                raise NotImplementedError('{} unknown inital type'.format(args.init_type))
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)

    # import network
    
    gen_net = Generator()
    dis_net = Discriminator()
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

    train_set = unimib_load_dataset(incl_xyz_accel = True, incl_rms_accel = False, incl_val_group = False, is_normalize = True, one_hot_encode = False, data_mode = 'Train', single_class = True, class_name = args.class_name, augment_times=args.augment_times)
    train_loader = data.DataLoader(train_set, batch_size=args.batch_size, num_workers=args.num_workers, shuffle = True)
 
    if args.max_iter:
        args.max_epoch = np.ceil(args.max_iter / len(train_loader))

    # initial
    fixed_z = torch.tensor(np.random.normal(0, 1, (100, args.latent_dim))).to(device, dtype=torch.float32)
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
        fixed_z = checkpoint['fixed_z']

        args.path_helper = checkpoint['path_helper']
        logger = create_logger(args.path_helper['log_path'])
        print(f'=> loaded checkpoint {checkpoint_file} (epoch {start_epoch})')
        writer = SummaryWriter(args.path_helper['log_path'])
        del checkpoint
    else:
    # create new log dir
        assert args.exp_name
        args.path_helper = set_log_dir('logs', args.exp_name)
        logger = create_logger(args.path_helper['log_path'])
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
        
        train(args, gen_net, dis_net, gen_optimizer, dis_optimizer, gen_avg_param, train_loader, epoch, writer_dict,fixed_z, lr_schedulers)
        


#TO DO: Validate add synthetic data plot in tensorboard 
        gen_net.eval()
        plot_buf = gen_plot(gen_net, epoch, args.class_name)
        image = PIL.Image.open(plot_buf)
        image = ToTensor()(image).unsqueeze(0)
        writer.add_image('Image', image[0], epoch)
        
        is_best = False
        avg_gen_net = deepcopy(gen_net)
        load_params(avg_gen_net, gen_avg_param, args)
# Add module in model saving code exp'gen_net.module.state_dict()' to solve the model loading unpaired name problem
        save_checkpoint({
            'epoch': epoch + 1,
            'gen_model': args.gen_model,
            'dis_model': args.dis_model,
            'gen_state_dict': gen_net.state_dict(),
            'dis_state_dict': dis_net.state_dict(),
            'avg_gen_state_dict': avg_gen_net.state_dict(),
            'gen_optimizer': gen_optimizer.state_dict(),
            'dis_optimizer': dis_optimizer.state_dict(),
            'best_fid': best_fid,
            'path_helper': args.path_helper,
            'fixed_z': fixed_z
        }, is_best, args.path_helper['ckpt_path'], filename="checkpoint")
        del avg_gen_net
        
def gen_plot(gen_net, epoch, class_name):
    """Create a pyplot plot and save to buffer."""
    synthetic_data = [] 

    for i in range(10):
        fake_noise = torch.FloatTensor(np.random.normal(0, 1, (1, 100)))
        fake_sigs = gen_net(fake_noise).to('cpu').detach().numpy()
        synthetic_data.append(fake_sigs)

    fig, axs = plt.subplots(2, 5, figsize=(20,5))
    fig.suptitle(f'Synthetic {class_name} at epoch {epoch}', fontsize=30)
    for i in range(2):
        for j in range(5):
            axs[i, j].plot(synthetic_data[i*5+j][0][0][0][:])
            axs[i, j].plot(synthetic_data[i*5+j][0][1][0][:])
            axs[i, j].plot(synthetic_data[i*5+j][0][2][0][:])
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    return buf

if __name__ == '__main__':
    main()
