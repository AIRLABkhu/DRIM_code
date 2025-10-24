import os
import numpy as np
from stream_metrics import StreamSegMetrics
import tqdm
import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch import optim
import math
from focal_frequency_loss import FocalFrequencyLoss as FFL

from model import *
from dataset import *

from utils import *

import zipfile
import shutil

import gc
import warnings
warnings.filterwarnings('ignore')
import logging

logging.getLogger("transformers").setLevel(logging.ERROR)

from torch.cuda.amp import autocast, GradScaler

def unzip_if_needed(zip_path, extract_to):
    """Unzips the file if the target directory does not exist."""
    if not os.path.exists(extract_to):
        print(f"Directory '{extract_to}' does not exist. Extracting the zip file...")

        if os.path.exists(zip_path):
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall("/data2/local_datasets/")  # Extracts to /local_datasets/
            print("Extraction completed!")
        else:
            print(f"Error: Zip file '{zip_path}' does not exist.")
    else:
        print(f"Directory '{extract_to}' already exists. No need to extract.")
        
def save_experiment_code(exp_dir, scripts=["train.py", "model.py", "dataset.py"]):
    for script in scripts:
        if os.path.exists(script):
            shutil.copy(script, os.path.join(exp_dir, script))
        else:
            print(f"Warning: {script} not found.")
            
class CosineAnnealingWithDecayLR(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1, decay_factor=0.9):
        self.T_max = T_max
        self.eta_min = eta_min
        self.decay_factor = decay_factor
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch == 0:
            return [group["lr"] for group in self.optimizer.param_groups]

        current_cycle = self.last_epoch // self.T_max 
        decay = self.decay_factor ** current_cycle

        if self.last_epoch % self.T_max == 0 and self.last_epoch != 0:
            cosine_factor = 0
        else:
            cosine_factor = (1 + math.cos(math.pi * (self.last_epoch % self.T_max) / self.T_max)) / 2

        return [
            decay * (self.eta_min + (base_lr - self.eta_min) * cosine_factor)
            for base_lr in self.base_lrs
        ]

from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau


class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater thant or equal to 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
                self.last_epoch = self.after_scheduler.last_epoch + self.total_epoch + 1
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)

        
class FFTLoss(nn.Module):
    def __init__(self):
        super(FFTLoss, self).__init__()

    def forward(self, x, y):
        fft_x = torch.fft.fft2(x.to(torch.complex64))
        fft_y = torch.fft.fft2(y.to(torch.complex64))
        
        diff = fft_x - fft_y        
        loss = torch.mean(abs(diff))
        return loss

class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(), k).unsqueeze(0).unsqueeze(0)  # [1,1,5,5]
        if torch.cuda.is_available():
            self.kernel = self.kernel.cuda()
        self.loss = CharbonnierLoss()
        
    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)
        
    def laplacian_kernel(self, current):
        filtered = self.conv_gauss(current)
        down = filtered[:,:,::2,::2]
        new_filter = torch.zeros_like(filtered)
        new_filter[:,:,::2,::2] = down*4
        filtered = self.conv_gauss(new_filter)
        diff = current - filtered
        return diff
        
    def forward(self, x, y):
        loss = self.loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return loss
        
class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3, reduction='mean'):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, x, y):
        diff = x - y
        loss = torch.sqrt(diff ** 2 + self.eps ** 2)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

    
def calculate_depth_loss(outputs, gts):
    loss_char = CharbonnierLoss()(outputs, gts)
    loss_fft = FFTLoss()(outputs, gts) * 0.01
    loss_edge = EdgeLoss()(outputs, gts) * 0.05
    loss_l1 = nn.L1Loss()(outputs, gts) * 0.1

    loss = loss_char + loss_fft + loss_edge + loss_l1
    return loss

    
def compute_loss(outputs, gts, mask_pred, gt_masks):
    depth_loss = calculate_depth_loss(outputs, gts)
    mask_loss = nn.CrossEntropyLoss()(mask_pred, gt_masks) * 0.8
    
    return depth_loss, mask_loss

def get_parameter_number(net):
    total_num = sum(np.prod(p.size()) for p in net.parameters())
    trainable_num = sum(np.prod(p.size()) for p in net.parameters() if p.requires_grad)
    print('Total: ', total_num)
    print('Trainable: ', trainable_num)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Experiments Setting
    parser.add_argument('--exp', type=str, default='exp', help='experiment name')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    # Datasets
    parser.add_argument('--data_dir', type=str, default='/data2/local_datasets/DRIM_new')
    parser.add_argument('--zip_dir', type=str, default='/data/datasets/DRIM_new.zip')
    parser.add_argument('--crop', action='store_true')
    # Training Setting
    parser.add_argument('--encoder', type=str, default='swinv2')
    parser.add_argument('--decoder', type=str, default='upernet')
    parser.add_argument('--model_size', type=str, default='base')
    parser.add_argument('--lite', action='store_true')
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
    parser.add_argument('--encoder_lr', type=float, default=2e-4, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight_decay')
    parser.add_argument('--epoch', type=int, default=100, help='training epoch')
    parser.add_argument('--warmup_epoch', type=int, default=0, help='warmup epoch')
    parser.add_argument('--t_max', type=int, default=10)
    parser.add_argument('--decay_factor', type=float, default=0.8)
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--interval', type=int, default=100, help='image save interval')
    parser.add_argument('--val_frequency', type=int, default=1, help='validation frequency')
    parser.add_argument('--merge_policy', type=str, default='cat')
    # Debugging Setting
    parser.add_argument('--eval_batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--save_image', action='store_true')
    parser.add_argument('--save_code', action='store_true')
    parser.add_argument('--feasibility', action='store_true')
    
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--use_amp', action='store_true')
    
    args = parser.parse_args()

    exp = args.exp
    print(exp)
    unzip_if_needed(args.zip_dir, args.data_dir)
    
    seed_torch(seed=args.seed)
    device = torch.device("cuda")
    
    os.makedirs(exp, exist_ok=True)
    
    with open(os.path.join(exp, "args.txt"), "w") as f:
        for arg in vars(args).items():
            f.write(f"{arg}\n")
    if args.save_code:
        save_experiment_code(exp)
        
    output_dir = 'output_images'
    os.makedirs(os.path.join(exp, output_dir), exist_ok=True)
    os.makedirs(os.path.join(exp, 'checkpoints'), exist_ok=True)
    
    img_dir = os.path.join(args.data_dir, "raw_dataset/trainsets/depth") #args.image_dir
    gt_dir = os.path.join(args.data_dir, "raw_dataset/trainsets/gt") #args.gt_dir
    inter_dir = os.path.join(args.data_dir, "raw_dataset/trainsets/inter_artifact_mask") #args.inter_mask_dir       
    sensor_dir = os.path.join(args.data_dir, "raw_dataset/trainsets/sensor_artifact_mask") #args.sensor_mask_dir

    scaler = GradScaler() if args.use_amp else None
    
    MEAN = 3266.8149
    STD = 563.3607
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.float()),
        transforms.Normalize([MEAN], [STD]),
    ])
    
    train_dataset = DepthDataset_aug(img_dir+'/train', gt_dir+'/train', inter_dir+'/train', sensor_dir + '/train', mode = 'train', transform=transform, crop=args.crop)
    
    val_dataset = DepthDataset(img_dir+'/valid', gt_dir+'/valid', inter_dir+'/valid', sensor_dir+'/valid', mode = 'valid', transform=transform)
    test_dataset = DepthDataset(img_dir+'/test', gt_dir+'/test', inter_dir+'/test', sensor_dir+'/test', mode = 'test', transform=transform)
    
    print(f"Train: {len(train_dataset)}, Validation: {len(val_dataset)}, Test: {len(test_dataset)}, Train Sample Shape: {train_dataset[0][0].shape}")

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=8, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=8, pin_memory=True)

    steps_per_epoch = len(train_dataloader)

    model = DRIM(encoder_name=args.encoder, decoder_name=args.decoder, model_size=args.model_size, merge_policy=args.merge_policy, num_classes=3).to(device)
    get_parameter_number(model)

    for param in model.parameters():
        param.requires_grad = True

    encoder_params = list(model.encoder.parameters())        
    mask_decoder_params = list(model.mask_decoder.parameters())
    depth_decoder_params = list(model.depth_decoder.parameters())
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.decay_factor)

    start = time.time()
    print("Starting Warmup Phase")
    
    print("Encoder Freeze")
    encoder_params = list(model.encoder.parameters())
    for param in encoder_params:
        param.requires_grad = False

    optimizer.zero_grad()
    optimizer.step()
    for epoch in range(1, args.warmup_epoch+1):
        model.train()
        running_loss = 0.0
        running_depth_loss = 0.0
        running_mask_loss = 0.0
        epoch_start_time = time.time()
        for i, (images, gts, gt_masks) in enumerate(train_dataloader):
            images, gts, gt_masks = images.to(device), gts.to(device), gt_masks.to(device)
            
            optimizer.zero_grad()

            if args.use_amp:
                with autocast():
                    outputs, mask_pred = model(images, gt_masks)
                    depth_loss, mask_loss = compute_loss(outputs, gts, mask_pred, gt_masks)
                    loss = depth_loss + mask_loss
            else:
                outputs, mask_pred = model(images, gt_masks)
                depth_loss, mask_loss = compute_loss(outputs, gts, mask_pred, gt_masks)
                loss = depth_loss + mask_loss
                
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"[Warning] NaN detected at batch {i}, skipping...")
                
                optimizer.zero_grad(set_to_none=True)
                del images, gts, gt_masks, outputs, mask_pred, loss
                torch.cuda.empty_cache()
                gc.collect()
                #continue
                break
                                
            if args.use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward() 
                optimizer.step()

            running_loss += loss.item()
            running_depth_loss += depth_loss.item()
            running_mask_loss += mask_loss.item()
            
            if (i > 0) and (i % args.interval == 0):
                avg_loss = running_loss / (i + 1)
                avg_depth_loss = running_depth_loss / (i + 1)
                avg_mask_loss = running_mask_loss / (i + 1)
                
                decoder_lr = optimizer.param_groups[-1]['lr']
                encoder_lr = optimizer.param_groups[0]['lr']
                
                epoch_elapsed_time = time.time() - epoch_start_time
                epoch_elapsed_minutes = int(epoch_elapsed_time // 60)
                epoch_elapsed_seconds = int(epoch_elapsed_time % 60)
                
                total_time = (epoch_elapsed_time / (i + 1)) * (len(train_dataloader))
                total_minutes = int(total_time // 60)
                total_seconds = int(total_time % 60)
                print(f"Warmup Epoch [{epoch}/{args.warmup_epoch}], Iteration {i+1}/{len(train_dataloader)}: "
                  f"Total Loss: {avg_loss:.5f}, Mask Loss: {avg_mask_loss:.5f}, Depth Loss: {avg_depth_loss:.5f}, "
                  f"Decoder lr: {decoder_lr:.2e}, Time: [{epoch_elapsed_minutes:02d}:{epoch_elapsed_seconds:02d}/{total_minutes:02d}:{total_seconds:02d}]")

    # Training loop
    best_psnr = 0.0
    best_rmse = float('inf')
    best_epoch = 0
    
    best_miou = 0.0
    print("Starting Main Training Phase")
    
    for param in model.parameters():
        param.requires_grad = True
        
    for epoch in range(1, args.epoch+1):
        model.train()
        running_loss = 0.0
        running_depth_loss = 0.0
        running_mask_loss = 0.0
        epoch_start_time = time.time()
        for i, (images, gts, gt_masks) in enumerate(train_dataloader):
            images, gts, gt_masks = images.to(device), gts.to(device), gt_masks.to(device)
            
            optimizer.zero_grad()
            if args.use_amp:
                with autocast():
                    outputs, mask_pred = model(images)
                    depth_loss, mask_loss = compute_loss(outputs, gts, mask_pred, gt_masks)
                    loss = depth_loss + mask_loss
            else:
                outputs, mask_pred = model(images)
                depth_loss, mask_loss = compute_loss(outputs, gts, mask_pred, gt_masks)
                loss = depth_loss + mask_loss
                
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"[Warning] NaN detected at batch {i}, skipping... / NAN: {torch.isnan(loss)}, INF: {torch.isinf(loss)}")
                optimizer.zero_grad(set_to_none=True)
                del images, gts, gt_masks, outputs, mask_pred, loss
                torch.cuda.empty_cache()
                gc.collect()
                break
                        
            if args.use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward() 
                optimizer.step()

            if isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
                iter_fraction = i / len(train_dataloader)
                scheduler.step((epoch - 1) + iter_fraction)
            elif isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                scheduler.step()

            running_loss += loss.item()
            running_depth_loss += depth_loss.item()
            running_mask_loss += mask_loss.item()

            if (i > 0) and (i % args.interval == 0 or i == len(train_dataloader) - 1):            
                avg_loss = running_loss / (i + 1)
                avg_depth_loss = running_depth_loss / (i + 1)
                avg_mask_loss = running_mask_loss / (i + 1)
                decoder_lr = optimizer.param_groups[-1]['lr']
                encoder_lr = optimizer.param_groups[0]['lr']
                
                epoch_elapsed_time = time.time() - epoch_start_time
                epoch_elapsed_minutes = int(epoch_elapsed_time // 60)
                epoch_elapsed_seconds = int(epoch_elapsed_time % 60)
                
                total_time = (epoch_elapsed_time / (i + 1)) * (len(train_dataloader))
                total_minutes = int(total_time // 60)
                total_seconds = int(total_time % 60)
                print(f"Epoch [{epoch}/{args.epoch}], Iteration {i+1}/{len(train_dataloader)}: "
                  f"Total Loss: {avg_loss:.5f}, Mask Loss: {avg_mask_loss:.5f}, Depth Loss: {avg_depth_loss:.5f}, "
                  f"Decoder lr: {decoder_lr:.2e}, Time: [{epoch_elapsed_minutes:02d}:{epoch_elapsed_seconds:02d}/{total_minutes:02d}:{total_seconds:02d}]")
                
                if args.save_image and (i % (args.interval * 2) == 0): # (i % (args.interval * 2 if epoch > 1 else args.interval) == 0):
                    print("Image Saved!")
                    
                                
                    gt_inter_mask = torch.zeros_like(gt_masks).float()
                    gt_inter_mask[gt_masks==1] = 1
                    gt_inter_mask = gt_inter_mask.unsqueeze(1)
                    gt_sensor_mask = torch.zeros_like(gt_masks).float()
                    gt_sensor_mask[gt_masks==2] = 1
                    gt_sensor_mask = gt_sensor_mask.unsqueeze(1)
            
                    images = denormalize(images, MEAN, STD)
                    outputs = denormalize(outputs, MEAN, STD)
                    gts = denormalize(gts, MEAN, STD)
                    
                    mask = torch.argmax(mask_pred, dim=1).unsqueeze(1)
                
                    inter_mask = torch.zeros_like(mask).float()
                    inter_mask[mask==1] = 1
                    sensor_mask = torch.zeros_like(mask).float()
                    sensor_mask[mask==2] = 1
                    

                    for j in range(1):
                        #save_depth(outputs[j].detach().cpu(), os.path.join(os.path.join(exp, output_dir), f'epoch_{epoch}_output_{i*args.batch_size+j:05d}.png'))
                        combined_outputs = torch.cat([minmax(images[j].detach().cpu()), minmax(outputs[j].detach().cpu()), minmax(gts[j].detach().cpu())], dim=2)
                        combined_masks = torch.cat([inter_mask[j].detach().cpu(), sensor_mask[j].detach().cpu(), gt_inter_mask[j].detach().cpu(), gt_sensor_mask[j].detach().cpu()], dim=2)
                        save_image(combined_outputs, os.path.join(os.path.join(exp, output_dir), f'epoch_{epoch}_combined_outputs_{i*args.batch_size+j:05d}.png'), normalize=False)
                        save_image(combined_masks, os.path.join(os.path.join(exp, output_dir), f'epoch_{epoch}_combined_masks_{i*args.batch_size+j:05d}.png'))
                        
        if not isinstance(scheduler, (torch.optim.lr_scheduler.OneCycleLR, 
                                    torch.optim.lr_scheduler.CosineAnnealingWarmRestarts)):
            scheduler.step()
        torch.save(model.state_dict(), os.path.join(os.path.join(exp, 'checkpoints'), f'latest.pth'))
        
        metrics = StreamSegMetrics(3) 
        ###
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        gc.collect()
         
        if True: #epoch % args.val_frequency == 0 or epoch == args.epoch:
            #with torch.no_grad():
            with torch.no_grad(), autocast():
                val_rmse, val_psnr, val_a1, val_a2, val_a3, val_score, val_artifact, val_are, val_mae, val_artifact_mae = improved_evaluate(model, val_dataloader, device, args.batch_size, metrics, save_img=False, exp=exp, use_sensor_mask=False, sliding=False)
                
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            gc.collect()
            ###
            print()
            print(f'Val RMSE: {val_rmse:.4f} m')
            print(f'Val ARE: {val_are:.4f}')
            print(f'Val MAE: {val_mae:.4f} m')
            print(f'Val A1: {val_a1:.4f}')
            print(f'Val A2: {val_a2:.4f}')
            print(f'Val A3: {val_a3:.4f}')
            #print(f'Val PSNR: {val_psnr:.2f}')
            print(f'Val artifact: {val_artifact:.4f} m')
            print(f'Val artifact MAE: {val_artifact_mae:.4f} m')
            print(f"Val Class IoU: {{{', '.join(f'{value:.4f}' for value in val_score['Class IoU'].values())}}}")
            print(f'Val Mean IoU: {val_score["Mean IoU"]:.4f}')
            #print(metrics.to_str(val_score))
            if val_rmse < best_rmse:
                best_epoch = epoch
                best_rmse = val_rmse
                best_are = val_are
                best_mae = val_mae
                best_a1 = val_a1
                best_a2 = val_a2
                best_a3 = val_a3
                best_artifact = val_artifact
                best_artifact_mae = val_artifact_mae
                
                best_miou = val_score["Mean IoU"]
                best_class_iou = val_score["Class IoU"]
                
                torch.save(model.state_dict(), os.path.join(os.path.join(exp, 'checkpoints'), f'best.pth'))
                save_results(exp, epoch, val_rmse, val_a1, val_a2, val_a3, val_artifact, val_score, val_are, val_mae, val_artifact_mae)
                
                #if epoch > 10:
                if True: #val_rmse < 0.0210 and args.seed == 42:
                    print()
                    with torch.no_grad():
                        test_rmse, test_psnr, test_a1, test_a2, test_a3, test_score, test_artifact, test_are, test_mae, test_artifact_mae = improved_evaluate(model, test_dataloader, device, args.batch_size, metrics, save_img=False, exp=exp, use_sensor_mask=False, sliding=False)
                    print()
                    print(f'Test RMSE: {test_rmse:.4f} m')
                    print(f'Test ARE: {test_are:.4f}')
                    print(f'Test MAE: {test_mae:.4f} m')
                    print(f'Test A1: {test_a1:.4f}')
                    print(f'Test A2: {test_a2:.4f}')
                    print(f'Test A3: {test_a3:.4f}')
                    print(f'Test artifact: {test_artifact:.4f} m')
                    print(f'Test artifact MAE: {test_artifact_mae:.4f} m')
                    print(f"Test Class IoU: {{{', '.join(f'{value:.4f}' for value in test_score['Class IoU'].values())}}}")
                    print(f'Test Mean IoU: {test_score["Mean IoU"]:.4f}')
                    save_results(exp, "Test", test_rmse, test_a1, test_a2, test_a3, test_artifact, test_score, test_are, test_mae, test_artifact_mae, csv_path="test_results.csv")
            print()
            print(f'Best Epoch: {best_epoch}')
            print(f'Best RMSE: {best_rmse:.4f} m')
            print(f'Best ARE: {best_are:.4f}')
            print(f'Best MAE: {best_mae:.4f} m')
            print(f'Best A1: {best_a1:.4f}')
            print(f'Best A2: {best_a2:.4f}')
            print(f'Best A3: {best_a3:.4f}')
            print(f'Best artifact: {best_artifact:.4f} m')
            print(f'Best artifact MAE: {best_artifact_mae:.4f} m')
            print(f"Best Class IoU: {{{', '.join(f'{value:.4f}' for value in best_class_iou.values())}}}")
            print(f'Best Mean IoU: {best_miou:.4f}')
            print()
    end = time.time()
    
    print('training time:', end-start, 's')
    print()
    model.load_state_dict(torch.load(os.path.join(exp, 'checkpoints', f'best.pth')))
    
    ########################################################
    print("Start measure time")
    time_test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
    run_time, fps = measure_inference_time(model, time_test_dataloader, device)
    print()
    print(f"run_time: {run_time:.4f}")
    print(f"fps: {fps:.4f}")
    print()
    ########################################################
    metrics = StreamSegMetrics(3)

    test_rmse, test_psnr, test_a1, test_a2, test_a3, test_score, test_artifact, test_are, test_mae, test_artifact_mae = improved_evaluate(model, test_dataloader, device, args.batch_size, metrics, save_img=False, exp=exp, use_sensor_mask=False, sliding=False)
    
    print(f'Test RMSE: {test_rmse:.4f} m')
    print(f'Test ARE: {test_are:.4f}')
    print(f'Test MAE: {test_mae:.4f} m')
    print(f'Test A1: {test_a1:.4f}')
    print(f'Test A2: {test_a2:.4f}')
    print(f'Test A3: {test_a3:.4f}')
    #print(f'Test PSNR: {test_psnr:.2f}')
    print(f'Test artifact: {test_artifact:.4f} m')
    print(f'Test artifact MAE: {test_artifact_mae:.4f} m')
    print(f"Test Class IoU: {{{', '.join(f'{value:.4f}' for value in test_score['Class IoU'].values())}}}")
    print(f'Test Mean IoU: {test_score["Mean IoU"]:.4f}')
    save_results(exp, "Test", test_rmse, test_a1, test_a2, test_a3, test_artifact, test_score, test_are, test_mae, test_artifact_mae, run_time, fps, csv_path="test_results.csv")

