import os
import random
import numpy as np
import cv2

import torch
import torch.nn.functional as F
from torchvision.utils import save_image

import tqdm
import time

MEAN = 3266.8149
STD = 563.3607

import pandas as pd

def save_results(exp, epoch, val_rmse, val_a1, val_a2, val_a3, val_artifact, val_score, val_are=None, val_mae=None, val_artifact_mae=None, run_time=None, fps=None, csv_path="results.csv"):
    """
    Save evaluation results to a CSV file using pandas DataFrame.
    
    Args:
        exp (str): Experiment directory path
        epoch (int or str): Current epoch or identifier (e.g., "Test")
        val_rmse (float): Root Mean Squared Error
        val_a1 (float): Threshold accuracy for 1.25
        val_a2 (float): Threshold accuracy for 1.25^2
        val_a3 (float): Threshold accuracy for 1.25^3
        val_artifact (float): Artifact error
        val_score (dict): Segmentation scores including IoU
        val_are (float, optional): Absolute Relative Error
        val_mae (float, optional): Mean Absolute Error
        csv_path (str, optional): CSV file name for results
    """
    import pandas as pd

    # Process experiment name if it contains "result/"
    exp = exp.replace("result/", "")

    # Create a new row dictionary with all metrics
    new_row = {
        "exp": exp,
        "epoch": epoch,
        "rmse": val_rmse,
        "a1": val_a1,
        "a2": val_a2,
        "a3": val_a3,
        "artifact": val_artifact,
        "mean_iou": val_score["Mean IoU"],
    }

    # Add class IoU values
    for i, iou in enumerate(val_score["Class IoU"].values(), 1):
        new_row[f"class_iou_{i}"] = iou

    # Add new metrics if provided
    if val_are is not None:
        new_row["are"] = val_are

    if val_mae is not None:
        new_row["mae"] = val_mae

    if val_artifact_mae is not None:
        new_row["artifact_mae"] = val_artifact_mae

    if run_time is not None:
        new_row["run_time"] = run_time

    if fps is not None:
        new_row["fps"] = fps

    # Read existing CSV or create new DataFrame
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        # Remove existing entry for this experiment
        df = df[df["exp"] != exp]
        # Append new row
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        df = pd.DataFrame([new_row])

    # Save to CSV
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
    
def seed_torch(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def focal_loss(pred, target, gamma=2., alpha=0.25):
    # Calculate CrossEntropy loss with reduction set to 'none' to apply focal loss to each sample
    bce_loss = F.cross_entropy(pred, target, reduction='none')
    # p_t is the probability of the correct class (for focal loss adjustment)
    p_t = torch.exp(-bce_loss)
    # Apply the focal loss formula
    loss = alpha * (1 - p_t) ** gamma * bce_loss
    return loss.mean() # Return the mean loss for the batch
    
def dice_loss(pred, target, smooth=1.0, num_classes=3):
    """
    Compute Dice Loss for multi-class segmentation.
    
    Args:
        pred (Tensor): Model output (logits or probabilities) with shape (B, C, H, W).
        target (Tensor): Ground truth with shape (B, H, W) or one-hot encoded (B, C, H, W).
        smooth (float): Smoothing factor to prevent division by zero.
        num_classes (int): Number of classes.
    
    Returns:
        Tensor: Dice loss.
    """

    # Ensure target is one-hot encoded if necessary
    if target.dim() == 3 and target.dtype == torch.long:
        target = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()

    pred = F.softmax(pred, dim=1)
    total_loss = 0.0

    for c in range(num_classes):
        pred_c = pred[:, c, :, :]
        target_c = target[:, c, :, :]

        intersection = torch.sum(pred_c * target_c, dim=[1, 2])  # (B,)
        union = torch.sum(pred_c, dim=[1, 2]) + torch.sum(target_c, dim=[1, 2])  # (B,)

        dice = (2. * intersection + smooth) / (union + smooth)
        total_loss += (1 - dice).mean()
    return total_loss / num_classes
    
def rmse(predictions, targets):
    return torch.sqrt(torch.mean((predictions - targets) ** 2, dim=(1, 2, 3)))

def psnr(predictions, targets, max_pixel_value=1.0):
    mse = torch.mean((predictions - targets) ** 2, dim=(1, 2, 3))
    psnr_value = 20 * torch.log10(max_pixel_value / torch.sqrt(mse))
    return psnr_value

def denormalize(image, mean=0.5, std=0.5):
    return image * std + mean

def ssim(x, y, C1=0.01**2, C2=0.03**2):
    """Compute the SSIM between two images."""
    mu_x = F.avg_pool2d(x, 3, 1, 0)
    mu_y = F.avg_pool2d(y, 3, 1, 0)

    sigma_x = F.avg_pool2d(x ** 2, 3, 1, 0) - mu_x ** 2
    sigma_y = F.avg_pool2d(y ** 2, 3, 1, 0) - mu_y ** 2
    sigma_xy = F.avg_pool2d(x * y, 3, 1, 0) - mu_x * mu_y

    ssim_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    ssim_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

    ssim_map = ssim_n / ssim_d
    return torch.clamp((1 - ssim_map) / 2, 0, 1)

def ssim_loss(x, y):
    """SSIM loss."""
    return ssim(x, y).mean()
    
def gradient_x(img):
    """Compute gradient along x-axis."""
    gx = img[:, :, :, :-1] - img[:, :, :, 1:]
    return gx

def gradient_y(img):
    """Compute gradient along y-axis."""
    gy = img[:, :, :-1, :] - img[:, :, 1:, :]
    return gy

def depth_smoothness_loss(pred, gt):
    """Compute the smoothness loss for the predicted depth map."""
    pred_dx = gradient_x(pred)
    pred_dy = gradient_y(pred)

    gt_dx = gradient_x(gt)
    gt_dy = gradient_y(gt)

    weights_x = torch.exp(-torch.mean(torch.abs(gt_dx), 1, keepdim=True))
    weights_y = torch.exp(-torch.mean(torch.abs(gt_dy), 1, keepdim=True))

    smoothness_x = pred_dx * weights_x
    smoothness_y = pred_dy * weights_y

    return (smoothness_x.abs().mean() + smoothness_y.abs().mean())

def gradient_loss(pred, gt):
    grad_pred_x = torch.abs(gradient_x(pred))
    grad_pred_y = torch.abs(gradient_y(pred))
    
    grad_gt_x = torch.abs(gradient_x(gt))
    grad_gt_y = torch.abs(gradient_y(gt))
    
    loss_x = nn.L1Loss()(grad_pred_x, grad_gt_x)
    loss_y = nn.L1Loss()(grad_pred_y, grad_gt_y)
    
    return (loss_x + loss_y) / 2


def minmax(image):
    return (image - image.min()) / (image.max() - image.min())
    

def save_depth(depth, path):
    depth = depth# * 65535
        
    np_image = depth.squeeze().numpy().astype(np.uint16)
    cv2.imwrite(path, np_image)
    
    return

def threshold_accuracy(output, target, threshold, eps=1e-8):
    delta = torch.max((output + eps) / (target + eps), (target + eps) / (output + eps))
    return (delta < threshold).float().mean(dim=(1, 2, 3))#.item()


def absolute_relative_error(pred, target):
    """
    Calculate the Absolute Relative Error (ARE) between predicted and target depth maps.
    
    Args:
        pred (torch.Tensor): Predicted depth map
        target (torch.Tensor): Ground truth depth map
        
    Returns:
        torch.Tensor: Mean absolute relative error for each sample in the batch
    """
    # Avoid division by zero by adding a small epsilon
    epsilon = 1e-8
    # Calculate absolute relative error
    abs_rel_err = torch.abs(pred - target) / (target + epsilon)
    # Return mean for each sample
    return torch.mean(abs_rel_err, dim=(1, 2, 3))

def mean_absolute_error(pred, target):
    """
    Calculate the Mean Absolute Error (MAE) between predicted and target depth maps.
    
    Args:
        pred (torch.Tensor): Predicted depth map
        target (torch.Tensor): Ground truth depth map
        
    Returns:
        torch.Tensor: Mean absolute error for each sample in the batch
    """
    # Calculate absolute error
    abs_err = torch.abs(pred - target)
    # Return mean for each sample
    return torch.mean(abs_err, dim=(1, 2, 3))

def improved_evaluate(model, dataloader, device, batch_size, metrics, exp=None, save_img=False, use_sensor_mask=False, disable_tqdm=True, sliding=False):
    metrics.reset()
    model.eval()
    test_output_dir = 'test_output' #test_output_hard / hard2
    os.makedirs(os.path.join(exp, test_output_dir), exist_ok=True)

    total_samples = len(dataloader.dataset)
    
    #test_times = []
    # Initialize metrics arrays
    all_rmse = np.zeros(total_samples)
    all_psnr = np.zeros(total_samples)
    all_a1 = np.zeros(total_samples)
    all_a2 = np.zeros(total_samples)
    all_a3 = np.zeros(total_samples)
    all_artifact = np.zeros(total_samples) #np.zeros(len(dataloader))
    all_artifact_mae = np.zeros(total_samples) #np.zeros(len(dataloader))
    # New metrics arrays
    all_are = np.zeros(total_samples)  # Absolute Relative Error
    all_mae = np.zeros(total_samples)  # Mean Absolute Error
     
    total_depth = 0
    total_samples = 0
    sample_count = 0
    
    print("Start evaluate")
    interval = len(dataloader) // 5 #5
    eval_start_time = time.time()
    with torch.no_grad():
        for i, (images, gts, masks) in enumerate(tqdm.tqdm(dataloader, disable=disable_tqdm)):
            current_batch_size = images.size(0)
            
            images, gts, masks = images.to(device), gts.to(device), masks.to(device)
            if sliding:
                outputs, mask_pred = sliding_window_inference(model, images)
            else:
                #tic = time.time()
                outputs, mask_pred = model(images)
                #toc = time.time()
                #test_times.append(toc-tic)

            mask = torch.argmax(mask_pred, dim=1)
            inter_mask = (mask == 1).float().unsqueeze(1)
            sensor_mask = (mask == 2).float().unsqueeze(1)
    
            # Create ground truth masks
            gt_inter_mask = (masks == 1).float().unsqueeze(1)
            gt_sensor_mask = (masks == 2).float().unsqueeze(1)
        
            metrics.update(masks.cpu().numpy(), mask_pred.detach().cpu().max(dim=1)[1].numpy())

            images = denormalize(images, MEAN, STD) # 0~1
            outputs = denormalize(outputs, MEAN, STD)
            gts = denormalize(gts, MEAN, STD)
            
            outputs = torch.clamp(outputs, min=0, max=65535)
            
            if use_sensor_mask:
                outputs = outputs * (1-sensor_mask)
                
            batch_psnr = psnr(outputs, gts).detach().cpu().numpy()
            
            outputs_m = outputs*0.00025
            gts_m =  gts*0.00025
            
            batch_rmse = rmse(outputs_m, gts_m).detach().cpu().numpy()
            
            batch_a1 = threshold_accuracy(outputs_m, gts_m, 1.25).detach().cpu().numpy()
            batch_a2 = threshold_accuracy(outputs_m, gts_m, 1.25 ** 2).detach().cpu().numpy()
            batch_a3 = threshold_accuracy(outputs_m, gts_m, 1.25 ** 3).detach().cpu().numpy()

            # Calculate new metrics
            batch_are = absolute_relative_error(outputs_m, gts_m).detach().cpu().numpy()  # ARE
            batch_mae = mean_absolute_error(outputs_m, gts_m).detach().cpu().numpy()  # MAE

            mask_union = torch.logical_or(gt_inter_mask, gt_sensor_mask)
            
            # Calculate artifact metrics per sample instead of per batch
            batch_artifact = []
            batch_artifact_mae = []

            for b in range(current_batch_size):
                sample_mask_union = mask_union[b]
                if sample_mask_union.sum() > 0:
                    # RMSE (artifact)
                    sample_artifact = torch.sqrt(((outputs_m[b][sample_mask_union] - gts_m[b][sample_mask_union]) ** 2).mean()).item()
                    # MAE (artifact_mae)
                    sample_artifact_mae = (outputs_m[b][sample_mask_union] - gts_m[b][sample_mask_union]).abs().mean().item()
                else:
                    sample_artifact = 0
                    sample_artifact_mae = 0

                batch_artifact.append(sample_artifact)
                batch_artifact_mae.append(sample_artifact_mae)

            batch_artifact = np.array(batch_artifact)
            batch_artifact_mae = np.array(batch_artifact_mae)

            total_depth += torch.sum(gts).cpu().numpy()
            total_samples += gts.numel()
            
            idx_start = sample_count
            idx_end = sample_count + current_batch_size
            
            # Store batch results
            all_rmse[idx_start:idx_end] = batch_rmse
            all_psnr[idx_start:idx_end] = batch_psnr
            all_a1[idx_start:idx_end] = batch_a1
            all_a2[idx_start:idx_end] = batch_a2
            all_a3[idx_start:idx_end] = batch_a3
            all_artifact[idx_start:idx_end] = batch_artifact
            all_artifact_mae[idx_start:idx_end] = batch_artifact_mae
            # Store new metrics results
            all_are[idx_start:idx_end] = batch_are
            all_mae[idx_start:idx_end] = batch_mae
                        
            sample_count += current_batch_size
            
            if (i >= 0) and (i % interval == 0):
                elapsed_time = time.time() - eval_start_time
                elapsed_minutes = int(elapsed_time // 60)
                elapsed_seconds = int(elapsed_time % 60)
                
                total_time = (elapsed_time / (i + 1)) * (len(dataloader))
                total_minutes = int(total_time // 60)
                total_seconds = int(total_time % 60)
                print(f"Iteration {i}/{len(dataloader)}: Time: [{elapsed_minutes:02d}:{elapsed_seconds:02d}/{total_minutes:02d}:{total_seconds:02d}]")
                    
                if save_img:
                    for j in range(1): #outputs.size(0)):
                        save_depth(outputs[j].detach().cpu(), os.path.join(os.path.join(exp, test_output_dir), f'output_{i*batch_size+j:05d}.png'))
                        combined_outputs = torch.cat([minmax(images[j].detach().cpu()), minmax(outputs[j].detach().cpu()), minmax(gts[j].detach().cpu())], dim=2)
                        save_image(combined_outputs, os.path.join(os.path.join(exp, test_output_dir), f'combined_outputs_{i*batch_size+j:05d}.png'), normalize=False)
    
                        gt_inter_mask = torch.zeros_like(masks).float()
                        gt_inter_mask[masks==1] = 1
                        gt_inter_mask = gt_inter_mask.unsqueeze(1)
                        gt_sensor_mask = torch.zeros_like(masks).float()
                        gt_sensor_mask[masks==2] = 1
                        gt_sensor_mask = gt_sensor_mask.unsqueeze(1)
                  
                        combined_masks = torch.cat([inter_mask[j].detach().cpu(), sensor_mask[j].detach().cpu(), gt_inter_mask[j].detach().cpu(), gt_sensor_mask[j].detach().cpu()], dim=2)
                        save_image(combined_masks, os.path.join(os.path.join(exp, test_output_dir), f'combined_masks_{i*batch_size+j:05d}.png'))
                    
    # Calculate average metrics            
    mean_depth = total_depth / total_samples
    avg_rmse = np.mean(all_rmse)
    avg_rrmse = (avg_rmse / mean_depth) * 100
    
    avg_psnr = np.mean(all_psnr)
    avg_a1 = np.mean(all_a1)
    avg_a2 = np.mean(all_a2)
    avg_a3 = np.mean(all_a3)
    
    avg_artifact = np.mean(all_artifact)
    avg_artifact_mae = np.mean(all_artifact_mae)

    # Calculate averages for new metrics
    avg_are = np.mean(all_are)
    avg_mae = np.mean(all_mae)
    
    score = metrics.get_results()
    
    # Return all metrics including the new ones
    #print('test_times: ', np.mean(test_times))
    return avg_rmse, avg_psnr, avg_a1, avg_a2, avg_a3, score, avg_artifact, avg_are, avg_mae, avg_artifact_mae
    
    
def sliding_window_inference(model, image, window_size=(256, 256), overlap=0.5):
    """
    Perform sliding window inference for DRIM model with Swin Transformer backbone.
    
    Args:
        model: The DRIM model
        image: Input depth image tensor [B, C, H, W]
        window_size: Size of sliding window
        overlap: Overlap ratio between windows
    
    Returns:
        Restored depth image and segmentation mask
    """
    device = image.device
    B, C, H, W = image.shape
    
    # Calculate stride based on overlap
    stride_h = int(window_size[0] * (1 - overlap))
    stride_w = int(window_size[1] * (1 - overlap))
    
    # Calculate number of windows
    n_h = max(1, (H - window_size[0]) // stride_h + 1 + int((H - window_size[0]) % stride_h > 0))
    n_w = max(1, (W - window_size[1]) // stride_w + 1 + int((W - window_size[1]) % stride_w > 0))
    
    # Initialize output tensors
    output_depth = torch.zeros((B, 1, H, W), device=device)
    output_mask = torch.zeros((B, 3, H, W), device=device)  # 3 classes: normal, interference, sensor
    count = torch.zeros((B, 1, H, W), device=device)
    
    # Process each window
    model.eval()
    with torch.no_grad():
        for i in range(n_h):
            for j in range(n_w):
                # Calculate window coordinates
                h_start = min(i * stride_h, H - window_size[0])
                w_start = min(j * stride_w, W - window_size[1])
                h_end = h_start + window_size[0]
                w_end = w_start + window_size[1]
                
                # Extract window
                window = image[:, :, h_start:h_end, w_start:w_end]
                
                # Run inference
                depth_output, mask_pred = model(window)
                
                # Add results to output tensors with overlap handling
                output_depth[:, :, h_start:h_end, w_start:w_end] += depth_output
                output_mask[:, :, h_start:h_end, w_start:w_end] += mask_pred
                count[:, :, h_start:h_end, w_start:w_end] += 1
    
    # Average overlapping regions
    output_depth = output_depth / count
    output_mask = output_mask / count.repeat(1, 3, 1, 1)
    
    return output_depth, output_mask

def measure_inference_time(model, dataloader, device, num_prints=100):
    model.eval()
    model = model.to(device)

    total_time = 0
    total_images = 0

    total_batches = len(dataloader)
    interval = max(1, total_batches // num_prints)

    test_times = []
    with torch.no_grad():
        for i, (images, gts, gt_masks) in enumerate(dataloader):
            images, gts, gt_masks = images.to(device), gts.to(device), gt_masks.to(device)

            tic = time.time()
            outputs, mask_pred = model(images)
            toc = time.time()
            time_taken = toc-tic
            test_times.append(time_taken)

            if i % interval == 0:
                avg_time_so_far = np.mean(test_times)
                fps_so_far = 1 / avg_time_so_far
                print(f"[Batch {i} / {total_batches}] Inference Time per Image: {avg_time_so_far:.4f} s, FPS: {fps_so_far:.2f}")

    avg_time_per_image_sec = np.mean(test_times) #avg_time_per_image * 1e-3
    fps = 1 / avg_time_per_image_sec

    return avg_time_per_image_sec, fps
