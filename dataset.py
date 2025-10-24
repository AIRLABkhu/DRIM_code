import os
from PIL import Image, ImageDraw
import numpy as np
import cv2
import random

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


class DepthDataset(Dataset):
    def __init__(self, image_dir, gt_dir, inter_dir, sensor_dir, mode, transform=None):
        self.image_dir = image_dir
        self.gt_dir = gt_dir
        self.inter_dir = inter_dir
        self.sensor_dir = sensor_dir
        self.mode = mode

        self.transform = transform

        self.image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
        self.gt_files = [f for f in os.listdir(gt_dir) if os.path.isfile(os.path.join(gt_dir, f))]
        self.inter_files = [f for f in os.listdir(inter_dir) if os.path.isfile(os.path.join(inter_dir, f))]
        self.sensor_files = [f for f in os.listdir(sensor_dir) if os.path.isfile(os.path.join(sensor_dir, f))]

        self.image_files.sort()
        self.gt_files.sort()
        self.inter_files.sort()
        self.sensor_files.sort()

        assert len(self.image_files) == len(self.gt_files) == len(self.inter_files) == len(self.sensor_files), "The number of images and gts should be the same"

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        gt_path = os.path.join(self.gt_dir, self.gt_files[idx])
        inter_path = os.path.join(self.inter_dir, self.inter_files[idx])
        sensor_path = os.path.join(self.sensor_dir, self.sensor_files[idx])

        image = Image.open(img_path)
        gt = Image.open(gt_path)
        inter = Image.open(inter_path)
        sensor = Image.open(sensor_path)

        if self.transform:
            image = self.transform(image)
            gt = self.transform(gt)
            
            image_size = image.shape[1:]
            
            inter = transforms.ToTensor()(inter)
            sensor = transforms.ToTensor()(sensor)
            
            inter = transforms.Resize(image_size)(inter)
            sensor = transforms.Resize(image_size)(sensor)
            

        mask = torch.zeros_like(inter).long()
        mask[inter == 0] = 1
        mask[sensor == 0] = 2
        
        return image, gt, mask.squeeze() #inter, sensor


class RandomErase(object):
    def __init__(self, probability=1.0, scale=(0.001, 0.05), ratio=(0.3, 3.3)):
        self.probability = probability
        self.scale = scale
        self.ratio = ratio
    
    def __call__(self, img, gt, inter, sensor):
        return self._random_erase(img, gt, inter, sensor)

    def _random_erase(self, img, gt, inter, sensor):
        img_array = np.array(img)
        gt_array = np.array(gt)
        inter_array = np.array(inter)
        sensor_array = np.array(sensor)
        height, width = img_array.shape[:2]

        # Calculate area and aspect ratio
        area = width * height
        target_area = np.random.uniform(self.scale[0], self.scale[1]) * area
        aspect_ratio = np.random.uniform(self.ratio[0], self.ratio[1])

        h = int(np.round(np.sqrt(target_area * aspect_ratio)))
        w = int(np.round(np.sqrt(target_area / aspect_ratio)))

        # Ensure that the width and height are within bounds
        if w <= width and h <= height:
            top = np.random.randint(0, height - h + 1)
            left = np.random.randint(0, width - w + 1)

            # Apply random erase
            img_array[top:top+h, left:left+w] = 0
            gt_array[top:top+h, left:left+w] = 0
            inter_array[top:top+h, left:left+w] = 255
            sensor_array[top:top+h, left:left+w] = 0

        image = Image.fromarray(img_array.astype(np.uint16))
        gt = Image.fromarray(gt_array.astype(np.uint16))
        inter = Image.fromarray(inter_array.astype(np.uint8))
        sensor = Image.fromarray(sensor_array.astype(np.uint8))
        return image, gt, inter, sensor



class RandomEraseEdges(object):
    def __init__(self, erase_ratio=0.5, patch_size=5):
        self.erase_ratio = erase_ratio
        self.patch_size = patch_size

    def __call__(self, img, gt, inter, sensor):
        img_array = np.array(img)
        gt_array = np.array(gt)

        gt_array_org = np.array(gt)
        gt_array_org = (((gt_array_org-gt_array_org.min()))/(gt_array_org.max() - gt_array_org.min())*255).astype(np.uint8)
        
        inter_array = np.array(inter)
        sensor_array = np.array(sensor)
        height, width = img_array.shape[:2]

        # Step 1: Edge detection using Canny
        edges = cv2.Canny(gt_array_org, threshold1=50, threshold2=150)

        # Step 2: Find edge pixels
        edge_pixels = np.argwhere(edges > 0)

        num_pixels_to_erase = int(len(edge_pixels) * self.erase_ratio)
        if num_pixels_to_erase > 0:
            selected_pixels = edge_pixels[np.random.choice(len(edge_pixels), num_pixels_to_erase, replace=False)]

            # Step 4: Erase the selected edge pixels with patches
            for (y, x) in selected_pixels:
                top_left_y = max(0, y - self.patch_size // 2)
                top_left_x = max(0, x - self.patch_size // 2)
                bottom_right_y = min(height, y + self.patch_size // 2 + 1)
                bottom_right_x = min(width, x + self.patch_size // 2 + 1)
                
                # Apply the patch erase
                img_array[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = 0
                gt_array[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = 0
                inter_array[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = 255
                sensor_array[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = 0
        
        image = Image.fromarray(img_array.astype(np.uint16))
        gt = Image.fromarray(gt_array.astype(np.uint16))
        inter = Image.fromarray(inter_array.astype(np.uint8))
        sensor = Image.fromarray(sensor_array.astype(np.uint8))

        return image, gt, inter, sensor


class RandomEraseFromEdge(object):
    def __init__(self, scale=(0.001, 0.05), ratio=(0.3, 3.3)):
        self.scale = scale
        self.ratio = ratio

    def __call__(self, img, gt, inter, sensor):
        img_array = np.array(img)
        gt_array = np.array(gt)

        gt_array_org = np.array(gt)
        gt_array_org = (((gt_array_org-gt_array_org.min()))/(gt_array_org.max() - gt_array_org.min())*255).astype(np.uint8)
        
        inter_array = np.array(inter)
        sensor_array = np.array(sensor)
        height, width = img_array.shape[:2]
        
        # Step 1: Edge detection using Canny
        edges = cv2.Canny(gt_array_org, threshold1=50, threshold2=150)

        # Step 2: Find edge pixels
        edge_pixels = np.argwhere(edges > 0)
        
        # Step 3: Randomly select one edge pixel
        if len(edge_pixels) > 0:
            y, x = edge_pixels[random.randint(0, len(edge_pixels) - 1)]

            # Step 4: Calculate random patch size
            area = width * height
            target_area = np.random.uniform(self.scale[0], self.scale[1]) * area
            aspect_ratio = np.random.uniform(self.ratio[0], self.ratio[1])

            h = int(np.round(np.sqrt(target_area * aspect_ratio)))
            w = int(np.round(np.sqrt(target_area / aspect_ratio)))

            # Ensure the patch size fits within the image boundaries
            top_left_y = max(0, y - h // 2)
            top_left_x = max(0, x - w // 2)
            bottom_right_y = min(height, y + h // 2 + 1)
            bottom_right_x = min(width, x + w // 2 + 1)

            # Step 5: Apply random erase
            img_array[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = 0
            gt_array[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = 0
            inter_array[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = 255
            sensor_array[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = 0
        
        image = Image.fromarray(img_array.astype(np.uint16))
        gt = Image.fromarray(gt_array.astype(np.uint16))
        inter = Image.fromarray(inter_array.astype(np.uint8))
        sensor = Image.fromarray(sensor_array.astype(np.uint8))

        return image, gt, inter, sensor


class Patching(object):
    def __init__(self, image_size=(480, 640)):
        self.image_size = image_size
        self.height, self.width = image_size
        self.ps = 256

    def __call__(self, img, gt, inter, sensor):
        img_array = np.array(img)
        gt_array = np.array(gt)
        inter_array = np.array(inter)
        sensor_array = np.array(sensor)

        rr     = random.randint(0, self.height-self.ps)
        cc     = random.randint(0, self.width-self.ps)

        # Crop patch
        img_array = img_array[rr:rr+self.ps, cc:cc+self.ps]
        gt_array = gt_array[rr:rr+self.ps, cc:cc+self.ps]
        inter_array = inter_array[rr:rr+self.ps, cc:cc+self.ps]
        sensor_array = sensor_array[rr:rr+self.ps, cc:cc+self.ps]

        image = Image.fromarray(img_array.astype(np.uint16))
        gt = Image.fromarray(gt_array.astype(np.uint16))
        inter = Image.fromarray(inter_array.astype(np.uint8))
        sensor = Image.fromarray(sensor_array.astype(np.uint8))

        return image, gt, inter, sensor


class DepthDataset_aug(Dataset):
    def __init__(self, image_dir, gt_dir, inter_dir, sensor_dir, mode, transform=None, crop=True):
        self.image_dir = image_dir
        self.gt_dir = gt_dir
        self.inter_dir = inter_dir
        self.sensor_dir = sensor_dir
        self.mode = mode

        self.transform = transform
        self.crop = crop

        self.image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
        self.gt_files = [f for f in os.listdir(gt_dir) if os.path.isfile(os.path.join(gt_dir, f))]
        self.inter_files = [f for f in os.listdir(inter_dir) if os.path.isfile(os.path.join(inter_dir, f))]
        self.sensor_files = [f for f in os.listdir(sensor_dir) if os.path.isfile(os.path.join(sensor_dir, f))]

        self.image_files.sort()
        self.gt_files.sort()
        self.inter_files.sort()
        self.sensor_files.sort()

        assert len(self.image_files) == len(self.gt_files) == len(self.inter_files) == len(self.sensor_files), "The number of images and gts should be the same"

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        gt_path = os.path.join(self.gt_dir, self.gt_files[idx])
        inter_path = os.path.join(self.inter_dir, self.inter_files[idx])
        sensor_path = os.path.join(self.sensor_dir, self.sensor_files[idx])

        image = Image.open(img_path)
        gt = Image.open(gt_path)
        inter = Image.open(inter_path)
        sensor = Image.open(sensor_path)

        if self.crop:
            image, gt, inter, sensor = Patching()(image, gt, inter, sensor)
            
        # aug erase
        random_int = random.randint(0,4)
        if random_int == 1:
            image, gt, inter, sensor = RandomEraseFromEdge()(image, gt, inter, sensor)
        elif random_int == 2:
            image, gt, inter, sensor = RandomErase()(image, gt, inter, sensor)
        elif random_int == 3:
            image, gt, inter, sensor = RandomEraseEdges()(image, gt, inter, sensor)
        

        if self.transform:
            image = self.transform(image)
            gt = self.transform(gt)
            
            image_size = image.shape[1:]
            
            inter = transforms.ToTensor()(inter)
            sensor = transforms.ToTensor()(sensor)
            
            inter = transforms.Resize(image_size)(inter)
            sensor = transforms.Resize(image_size)(sensor)

        aug = random.randint(0, 8)
        # Data Augmentations
        if aug==1:
            image = image.flip(1)
            gt = gt.flip(1)
            inter = inter.flip(1)
            sensor = sensor.flip(1)
        elif aug==2:
            image = image.flip(2)
            gt = gt.flip(2)
            inter = inter.flip(2)
            sensor = sensor.flip(2)
        elif aug==3:
            image = torch.rot90(image,dims=(1,2))
            gt = torch.rot90(gt,dims=(1,2))
            inter = torch.rot90(inter,dims=(1,2))
            sensor = torch.rot90(sensor,dims=(1,2))
        elif aug==4:
            image = torch.rot90(image,dims=(1,2), k=2)
            gt = torch.rot90(gt,dims=(1,2), k=2)
            inter = torch.rot90(inter,dims=(1,2), k=2)
            sensor = torch.rot90(sensor,dims=(1,2), k=2)
        elif aug==5:
            image = torch.rot90(image,dims=(1,2), k=3)
            gt = torch.rot90(gt,dims=(1,2), k=3)
            inter = torch.rot90(inter,dims=(1,2), k=3)
            sensor = torch.rot90(sensor,dims=(1,2), k=3)
        elif aug==6:
            image = torch.rot90(image.flip(1),dims=(1,2))
            gt = torch.rot90(gt.flip(1),dims=(1,2))
            inter = torch.rot90(inter.flip(1),dims=(1,2))
            sensor = torch.rot90(sensor.flip(1),dims=(1,2))
        elif aug==7:
            image = torch.rot90(image.flip(2),dims=(1,2))
            gt = torch.rot90(gt.flip(2),dims=(1,2))
            inter = torch.rot90(inter.flip(2),dims=(1,2))
            sensor = torch.rot90(sensor.flip(2),dims=(1,2))

        mask = torch.zeros_like(inter).long()
        mask[inter == 0] = 1
        mask[sensor == 0] = 2

        return image, gt, mask.squeeze() #inter, sensor
