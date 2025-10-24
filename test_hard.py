import os
import numpy as np
from stream_metrics import StreamSegMetrics
import tqdm
import argparse

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image

from model import *
from dataset import *
from utils import *

import warnings
warnings.filterwarnings('ignore')
import logging

logging.getLogger("transformers").setLevel(logging.ERROR)


import importlib.util
import sys
import os

import zipfile
import shutil

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

def load_model_from_exp(exp_path, device, encoder, decoder, size):
    model_path = os.path.join(exp_path, 'model.py')
    module_name = f"model_{os.path.basename(exp_path)}"

    spec = importlib.util.spec_from_file_location(module_name, model_path)
    model_module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = model_module
    spec.loader.exec_module(model_module)

    model = model_module.DRIM(encoder_name=encoder, decoder_name=decoder, model_size=size, merge_policy="cat").to(device)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Experiments Setting
    parser.add_argument('--exp', type=str, default='exp', help='experiment name')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    # Datasets
    parser.add_argument('--data_dir', type=str, default='/data2/local_datasets/DRIM_new_hard') #hard2
    parser.add_argument('--zip_dir', type=str, default='/data/datasets/DRIM_new_hard.zip') #hard2
    # Training Setting
    parser.add_argument('--encoder', type=str, default='swinv2')
    parser.add_argument('--decoder', type=str, default='upernet')
    parser.add_argument('--model_size', type=str, default='base')

    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--epoch', type=int, default=100, help='training epoch')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--interval', type=int, default=100, help='image save interval')
    # Debugging Setting
    parser.add_argument('--save_image', action='store_true')
    parser.add_argument('--save_code', action='store_true')
    parser.add_argument('--feasibility', action='store_true')
    
    parser.add_argument('--checkpoint', type=str, default='best')
    parser.add_argument('--lite', action='store_true')
    
    
    args = parser.parse_args()

    exp = args.exp
    print(exp)
    unzip_if_needed(args.zip_dir, args.data_dir)
    os.makedirs(exp, exist_ok=True)
    
    seed_torch(seed=args.seed)
    device = torch.device("cuda")
    
    
    img_dir = os.path.join(args.data_dir, "depth") #args.image_dir
    gt_dir = os.path.join(args.data_dir, "gt") #args.gt_dir
    inter_dir = os.path.join(args.data_dir, "inter_artifact_mask") #args.inter_mask_dir       
    sensor_dir = os.path.join(args.data_dir, "sensor_artifact_mask") #args.sensor_mask_dir

    MEAN = 3266.8149
    STD = 563.3607
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.float()),
        transforms.Normalize([MEAN], [STD]),
    ])
    
    test_dataset = DepthDataset(img_dir+'/test', gt_dir+'/test', inter_dir+'/test', sensor_dir+'/test', mode = 'test', transform=transform)
    print(len(test_dataset))
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)
    model = load_model_from_exp(args.exp, device, args.encoder, args.decoder, args.model_size)
    model.load_state_dict(torch.load(os.path.join(exp, 'checkpoints', f'{args.checkpoint}.pth')))
    #print("DRIM : ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    
    print()
    print(f'Save Image: {args.save_image}')

    metrics = StreamSegMetrics(3)
        
    test_rmse, test_psnr, test_a1, test_a2, test_a3, test_score, test_artifact, test_are, test_mae, test_artifact_mae = improved_evaluate(model, test_dataloader, device, args.batch_size, metrics, save_img=args.save_image, exp=exp, use_sensor_mask=False)
    print("-" * 40)
    print(f'Experiment: {exp.replace("result/", "")}')
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
    save_results(exp, "Test", test_rmse, test_a1, test_a2, test_a3, test_artifact, test_score, test_are, test_mae, test_artifact_mae, csv_path="test_results.csv")
    print("-" * 40)
