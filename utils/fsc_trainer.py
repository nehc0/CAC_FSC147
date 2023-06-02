import os
import time
import logging
from math import ceil

import torch
from torch import optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

from datasets.fsc_data import FSCData
from models.convtrans import VGG16Trans
from utils.trainer import Trainer
from utils.helper import Save_Handle, AverageMeter
from losses.losses import DownMSELoss

import numpy as np
from tqdm import tqdm
import wandb


def train_collate(batch):
    transposed_batch = list(zip(*batch))
    images = torch.stack(transposed_batch[0], 0)  # -> b c(3) w1 w1(384)
    dmaps = torch.stack(transposed_batch[1], 0)  # -> b w1 w1(384)
    shots = torch.stack(transposed_batch[2], 0)  # -> b c(3) w2 w2(64)
    scale_embeds = torch.stack(transposed_batch[3], 0)  # -> b 1 w2 w2(64)
    return images, dmaps, shots, scale_embeds


class FSCTrainer(Trainer):
    def setup(self):
        """initial the datasets, model, loss and optimizer"""
        args = self.args
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            raise Exception("gpu is not available")

        train_datasets = FSCData(args.data_dir,
                                 args.crop_size,
                                 args.downsample_ratio,
                                 method='train')
        train_dataloaders = DataLoader(train_datasets,
                                       collate_fn=train_collate,
                                       batch_size=args.batch_size,
                                       shuffle=True,
                                       num_workers=args.num_workers,
                                       pin_memory=True)
        val_datasets = FSCData(args.data_dir, method='test')
        val_dataloaders = DataLoader(val_datasets, 1, shuffle=False,
                                                      num_workers=args.num_workers, pin_memory=True)

        self.dataloaders = {'train': train_dataloaders, 'val': val_dataloaders}

        self.model = VGG16Trans()
        self.model.to(self.device)

        self.optimizer = optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        self.criterion = DownMSELoss(size=2)
        # self.criterion = torch.nn.MSELoss(reduction='sum')

        self.save_list = Save_Handle(max_num=args.max_model_num)
        self.best_mae = np.inf
        self.best_rmse = np.inf
        self.best_mae_at = 0
        self.best_count = 0

        self.start_epoch = 0
        if args.resume:
            suf = args.resume.rsplit('.', 1)[-1]
            if suf == 'tar':
                checkpoint = torch.load(args.resume, self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.start_epoch = checkpoint['epoch'] + 1
                self.best_mae = checkpoint['best_mae']
                self.best_rmse = checkpoint['best_rmse']
                self.best_mae_at = checkpoint['best_mae_at']
            elif suf == 'pth':
                self.model.load_state_dict(torch.load(args.resume, self.device))

        if args.scheduler == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=args.step, gamma=args.gamma, last_epoch=self.start_epoch-1)
        elif args.scheduler == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.t_max, eta_min=args.eta_min, last_epoch=self.start_epoch-1)


    def train(self):
        args = self.args
        self.epoch = None
        # self.val_epoch()
        for epoch in range(self.start_epoch, args.max_epoch):
            logging.info('-' * 5 + 'Epoch {}/{}'.format(epoch, args.max_epoch - 1) + '-' * 5)
            self.epoch = epoch
            self.train_epoch()
            self.scheduler.step()
            if epoch >= args.val_start and (epoch % args.val_epoch == 0 or epoch == args.max_epoch - 1):
                self.val_epoch()

    def train_epoch(self):
        epoch_loss = AverageMeter()
        epoch_mae = AverageMeter()
        epoch_rmse = AverageMeter()
        epoch_start = time.time()
        self.model.train()

        # Iterate over data.
        for inputs, targets, shots, scale_embeds in tqdm(self.dataloaders['train']):
            # b c h w
            inputs = inputs.to(self.device)
            targets = targets.to(self.device) * self.args.log_param # 100
            shots = shots.to(self.device)
            scale_embeds = scale_embeds.to(self.device)

            with torch.set_grad_enabled(True):
                N = inputs.size(0)
                et_dmaps = self.model(inputs, shots, scale_embeds)
                pre_count = torch.sum(et_dmaps.view(N, -1), dim=1).detach().cpu().numpy()
                gd_count = torch.sum(targets.view(N, -1), dim=1).detach().cpu().numpy()

                loss = self.criterion(et_dmaps, targets) / N

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                res = (pre_count - gd_count) / self.args.log_param
                epoch_loss.update(loss.item(), N)
                epoch_rmse.update(np.mean(res * res), N)
                epoch_mae.update(np.mean(abs(res)), N)

        logging.info('Epoch {} Train, Loss: {:.2f}, RMSE: {:.2f} MAE: {:.2f}, Cost {:.1f} sec'
                     .format(self.epoch, epoch_loss.get_avg(), np.sqrt(epoch_rmse.get_avg()), epoch_mae.get_avg(),
                             time.time() - epoch_start))
        wandb.log({'Train/loss': epoch_loss.get_avg(),
                   'Train/lr': self.scheduler.get_last_lr()[0],
                   'Train/rmse': np.sqrt(epoch_rmse.get_avg()),
                   'Train/mae': epoch_mae.get_avg()}, step=self.epoch)
        wandb.config.update({'sizeof_train_dataset': epoch_loss.get_count()})

        model_state_dic = self.model.state_dict()
        save_path = os.path.join(self.save_dir, '{}_ckpt.tar'.format(self.epoch))
        torch.save({
            'epoch': self.epoch,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_state_dict': model_state_dic,
            'best_mae': self.best_mae,
            'best_rmse': self.best_rmse,
            'best_mae_at': self.best_mae_at,
        }, save_path)
        self.save_list.append(save_path)  # control the number of saved models

    def val_epoch(self):
        epoch_start = time.time()
        self.model.eval()
        epoch_res = []

        for inputs, count, shot, scale_embed, name in tqdm(self.dataloaders['val']):
            shot = shot.to(self.device)
            scale_embed = scale_embed.to(self.device)
            # inputs are images with different sizes
            b, c, h, w = inputs.shape
            h, w = int(h), int(w)
            assert b == 1, 'the batch size should equal to 1 in validation mode'

            max_size = 384
            if w > max_size:
                w_stride = int(ceil(1.0 * w / max_size))
                w_step = max_size
                input_list = []
                for j in range(w_stride):
                    w_start = j * w_step
                    if j < w_stride - 1:
                        w_end = (j + 1) * w_step
                        input_list.append(inputs[:, :, :, w_start:w_end])
                    else:
                        res_img = inputs[:, :, :, w_start:w]
                        lenh = w - w_start
                        if lenh < 8:
                            continue
                        pad_len = float(max_size - lenh)
                        res_img = F.pad(res_img, [int(pad_len/2), ceil(pad_len/2)])
                        assert res_img.shape == (b, c, h, max_size)
                        input_list.append(res_img)
                with torch.set_grad_enabled(False):
                    pre_count = 0.0
                    for input_ in input_list:
                        input_ = input_.to(self.device)
                        output = self.model(input_, shot, scale_embed)
                        cur_count = torch.sum(output)
                        pre_count += cur_count
            else:
                with torch.set_grad_enabled(False):
                    inputs = inputs.to(self.device)
                    output = self.model(inputs, shot, scale_embed)
                    pre_count = torch.sum(output)

            epoch_res.append(count[0].item() - pre_count.item() / self.args.log_param)
            # epoch_res.append(count[0].item())

        epoch_res = np.array(epoch_res)
        rmse = np.sqrt(np.mean(np.square(epoch_res)))
        mae = np.mean(np.abs(epoch_res))

        logging.info('Epoch {} Val, RMSE: {:.2f} MAE: {:.2f}, Cost {:.1f} sec'
                     .format(self.epoch, rmse, mae, time.time() - epoch_start))

        model_state_dic = self.model.state_dict()
        if mae < self.best_mae:
            self.best_rmse = rmse
            self.best_mae = mae
            self.best_mae_at = self.epoch
            logging.info("SAVE best rmse {:.2f} mae {:.2f} model @epoch {}".format(self.best_rmse, self.best_mae, self.epoch))
            if self.args.save_all:
                torch.save(model_state_dic, os.path.join(self.save_dir, 'best_model_{}.pth'.format(self.best_count)))
                self.best_count += 1
            else:
                torch.save(model_state_dic, os.path.join(self.save_dir, 'best_model.pth'))

        logging.info("best mae {:.2f} rmse {:.2f} @epoch {}".format(self.best_mae, self.best_rmse, self.best_mae_at))

        if self.epoch is not None:
            wandb.log({'Val/best_mae': self.best_mae,
                       'Val/mae': mae,
                       'Val/rmse': rmse,
                       'Val/best@': self.best_mae_at,
                       }, step=self.epoch)
            wandb.config.update({'sizeof_val_dataset': len(epoch_res)})
