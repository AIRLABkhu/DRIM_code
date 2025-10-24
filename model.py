import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from transformers import SegformerForSemanticSegmentation
from transformers import (
    ConvNextV2Model, ConvNextV2Config, 
    Swinv2Model, Swinv2Config,
    SegformerModel, SegformerConfig
)
import numpy as np
import math

ACTIVATION = nn.SiLU(inplace=True)
MEAN = 3266.8149
STD = 563.3607

class SAM(nn.Module):
    def __init__(self, bias=False):
        super(SAM, self).__init__()
        self.bias = bias
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3, dilation=1, bias=self.bias)

    def forward(self, x):
        max = torch.max(x,1)[0].unsqueeze(1)
        avg = torch.mean(x,1).unsqueeze(1)
        concat = torch.cat((max,avg), dim=1)
        output = self.conv(concat)
        output = F.sigmoid(output) * x
        return output

class CAM(nn.Module):
    def __init__(self, channels, r):
        super(CAM, self).__init__()
        self.channels = channels
        self.r = r
        self.linear = nn.Sequential(
            nn.Linear(in_features=self.channels, out_features=self.channels//self.r, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=self.channels//self.r, out_features=self.channels, bias=True))

    def forward(self, x):
        max = F.adaptive_max_pool2d(x, output_size=1)
        avg = F.adaptive_avg_pool2d(x, output_size=1)
        b, c, _, _ = x.size()
        linear_max = self.linear(max.view(b,c)).view(b, c, 1, 1)
        linear_avg = self.linear(avg.view(b,c)).view(b, c, 1, 1)
        output = linear_max + linear_avg
        output = F.sigmoid(output) * x
        return output

class CBAM(nn.Module):
    def __init__(self, channels, r=2):
        super(CBAM, self).__init__()
        self.channels = channels
        self.r = r
        self.sam = SAM(bias=False)
        self.cam = CAM(channels=self.channels, r=self.r)

    def forward(self, x):
        output = self.cam(x)
        output = self.sam(output)
        return output + x

class DRIM(nn.Module):
    def __init__(self, in_channels=1, num_classes=3, encoder_name="convnextv2", decoder_name="fpn", model_size="tiny", merge_policy="cat"):
        super().__init__()
        self.num_classes = num_classes
        self.encoder_name = encoder_name
        self.decoder_name = decoder_name

        if encoder_name == "convnextv2":
            self._init_convnextv2_encoder(model_size)
        elif encoder_name == "swinv2":
            self._init_swinv2_encoder(model_size)
        elif encoder_name == "segformer":
            self._init_segformer_encoder(model_size)
        elif encoder_name == "resnet":
            self._init_resnet_encoder(model_size)
        else:
            raise ValueError(f"Unsupported encoder type: {encoder_name}")

        if decoder_name == "fpn":
            self.mask_decoder = ImprovedSegFPNHead(dims=self.dims, decoder_dim=self.decoder_dim, 
                                                   num_classes=num_classes, merge_policy=merge_policy)
            self.depth_decoder = ImprovedDepthFPNHead(dims=self.dims, decoder_dim=self.decoder_dim, 
                                                     merge_policy=merge_policy)
        elif decoder_name == "dpt":
            self.mask_decoder = SegDPTDecoder(dims=self.dims, decoder_dim=self.decoder_dim, 
                                              num_classes=num_classes)
            self.depth_decoder = DPTDecoder(dims=self.dims, decoder_dim=self.decoder_dim, 
                                           output_channels=1)
        elif decoder_name == "unet":
            self.mask_decoder = SegUnet(dims=self.dims, decoder_dim=self.decoder_dim, 
                                        num_classes=num_classes)
            self.depth_decoder = DepthUnet(dims=self.dims, decoder_dim=self.decoder_dim)
        elif decoder_name == "bifpn":
            self.mask_decoder = BiFPNSegHead(dims=self.dims, decoder_dim=self.decoder_dim, 
                                                   num_classes=num_classes, merge_policy=merge_policy)
            self.depth_decoder = BiFPNDepthHead(dims=self.dims, decoder_dim=self.decoder_dim, 
                                                     merge_policy=merge_policy)
        elif decoder_name == "upernet":
            self.mask_decoder = UperNetSegHead(dims=self.dims, decoder_dim=self.decoder_dim, 
                                                   num_classes=num_classes, merge_policy=merge_policy)
            self.depth_decoder = UperNetDepthHead(dims=self.dims, decoder_dim=self.decoder_dim, 
                                                     num_classes=num_classes, merge_policy=merge_policy)
        elif decoder_name == "unetPP":
            self.mask_decoder = SegUNetPlusPlus(dims=self.dims, num_classes=num_classes)
            self.depth_decoder = DepthUNetPlusPlus(dims=self.dims)
        elif decoder_name == "fapn":
            self.mask_decoder = SegFaPNHead(dims=self.dims, decoder_dim=self.decoder_dim, 
                                                   num_classes=num_classes, merge_policy=merge_policy)
            self.depth_decoder = DepthFaPNHead(dims=self.dims, decoder_dim=self.decoder_dim, 
                                                     merge_policy=merge_policy)
        else:
            raise ValueError(f"Unsupported decoder type: {decoder_name}")
    

        self.c0_conv = nn.Sequential(
            ResBlock(3, self.decoder_dim),
            CBAM(self.decoder_dim),
            ResBlock(self.decoder_dim, self.decoder_dim),
            CBAM(self.decoder_dim),
        )

    def _init_convnextv2_encoder(self, model_size):
        """Initialize ConvNeXtV2 encoder"""
        size_mapping = {
            "atto": "facebook/convnextv2-atto-1k-224",
            "nano": "facebook/convnextv2-nano-1k-224",
            "tiny": "facebook/convnextv2-tiny-22k-224",
            "base": "facebook/convnextv2-base-22k-224",
            "large": "facebook/convnextv2-large-22k-224"
        }
        
        # Load ConvNeXtV2 backbone
        self.encoder = ConvNextV2Model.from_pretrained(size_mapping[model_size])
        
        # Feature dimensions based on model size
        if model_size == "atto":
            self.dims = [40, 80, 160, 320]
            self.decoder_dim = 128
        elif model_size == "nano":
            self.dims = [80, 160, 320, 640]
            self.decoder_dim = 256
        elif model_size == "tiny":
            self.dims = [96, 192, 384, 768]
            self.decoder_dim = 256
        elif model_size == "base":
            self.dims = [128, 256, 512, 1024] 
            self.decoder_dim = 512
        elif model_size == "large":
            self.dims = [192, 384, 768, 1536]
            self.decoder_dim = 768
    
    def _init_swinv2_encoder(self, model_size):
        """Initialize Swin Transformer V2 encoder"""
        size_mapping = {
            "tiny": "microsoft/swinv2-tiny-patch4-window8-256",
            #"tiny": "microsoft/swinv2-tiny-patch4-window16-256",
            "small": "microsoft/swinv2-small-patch4-window8-256",
            "base": "microsoft/swinv2-base-patch4-window8-256",
            #"large": "microsoft/swinv2-large-patch4-window12-192-22k",
            "large": "microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft",
        }
        
        # Load Swin Transformer V2 backbone
        self.encoder = Swinv2Model.from_pretrained(size_mapping[model_size])
        
        # Feature dimensions based on model size
        if model_size == "tiny":
            self.dims = [96, 192, 384, 768]
            self.decoder_dim = 128
        elif model_size == "small":
            self.dims = [96, 192, 384, 768]
            self.decoder_dim = 256
        elif model_size == "base":
            self.dims = [128, 256, 512, 1024]
            self.decoder_dim = 512
        elif model_size == "large":
            self.dims = [192, 384, 768, 1536]
            self.decoder_dim = 768

    def _init_segformer_encoder(self, model_size):
        """Initialize Segformer encoder"""
        # Segformer size mapping
        if model_size == "b0":
            model_name = "nvidia/mit-b0"
            self.dims = [32, 64, 160, 256]
            self.decoder_dim = 256
        elif model_size == "b1":
            model_name = "nvidia/mit-b1"
            self.dims = [64, 128, 320, 512]
            self.decoder_dim = 256
        elif model_size == "b2":
            model_name = "nvidia/mit-b2"
            self.dims = [64, 128, 320, 512]
            self.decoder_dim = 768
        elif model_size == "b3":
            model_name = "nvidia/mit-b3"
            self.dims = [64, 128, 320, 512]
            self.decoder_dim = 768
        elif model_size == "b4":
            model_name = "nvidia/mit-b4"
            self.dims = [64, 128, 320, 512]
            self.decoder_dim = 768
        elif model_size == "b5":
            model_name = "nvidia/mit-b5"
            self.dims = [64, 128, 320, 512]
            self.decoder_dim = 768
        else:
            raise ValueError(f"Unsupported SegFormer size: {model_size}")
            
        # Load SegFormer backbone
        model = SegformerModel.from_pretrained(model_name)
        self.encoder = model.encoder

    def _init_resnet_encoder(self, model_size):
        """Initialize ResNet encoder"""
        import torchvision.models as models
        
        size_mapping = {
            "18": models.resnet18,
            "34": models.resnet34,
            "50": models.resnet50,
            "101": models.resnet101,
            "152": models.resnet152,
        }
        
        # Load ResNet backbone with pretrained weights
        resnet = size_mapping[model_size](pretrained=True)
        
        self.encoder = nn.ModuleList([
            nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu),  # layer0
            nn.Sequential(resnet.maxpool, resnet.layer1),          # layer1
            resnet.layer2,                                         # layer2
            resnet.layer3,                                         # layer3
            resnet.layer4,                                         # layer4
        ])
        
        # Feature dimensions based on model size
        if model_size in ["18", "34"]:
            self.dims = [64, 64, 128, 256, 512]
            self.decoder_dim = 256
        elif model_size in ["50", "101", "152"]:
            self.dims = [64, 256, 512, 1024, 2048]
            self.decoder_dim = 512
        
        # Remove first dimension (conv1) if needed to match other encoders
        self.dims = self.dims[1:]
    
    def _get_features(self, x):
        """Extract features from encoder based on type"""
        if self.encoder_name == "convnextv2":
            outputs = self.encoder(x, output_hidden_states=True).hidden_states[-4:]
            return outputs
            
        elif self.encoder_name == "swinv2":
            outputs = self.encoder(x, output_hidden_states=True).reshaped_hidden_states[:-1]
            return outputs
            
        elif self.encoder_name == "segformer":
            outputs = self.encoder(x, output_hidden_states=True).hidden_states
            return outputs
        
        elif self.encoder_name == "resnet":
            features = []
            x = self.encoder[0](x)  # conv1, bn1, relu
            for i in range(1, len(self.encoder)):
                x = self.encoder[i](x)
                features.append(x)
            return features


    def forward(self, x, gt_masks=None):
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
            #x = depth_to_normal(x)
            
        encoder_outputs = self._get_features(x)
        c0 = self.c0_conv(x)
        
        mask_pred = self.mask_decoder(encoder_outputs, c0)
        
        probs = torch.softmax(mask_pred, dim=1)
        if gt_masks is None:
            x = self.depth_decoder(encoder_outputs, probs, c0) #.detach())
        else:
            gt_masks = F.one_hot(gt_masks, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
            x = self.depth_decoder(encoder_outputs, gt_masks, c0)
            
        mask_pred =  F.interpolate(mask_pred, size=x.size()[2:], mode='bilinear', align_corners=True)
        return x, mask_pred
        
        
class SobelEdge(nn.Module):
    def __init__(self):
        super(SobelEdge, self).__init__()
        
        self.sobel_x = torch.nn.Parameter(
            torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).reshape(1, 1, 3, 3),
            requires_grad=False
        )
        self.sobel_y = torch.nn.Parameter(
            torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).reshape(1, 1, 3, 3),
            requires_grad=False
        )
        
    def forward(self, x):
        edge_x = F.conv2d(x, self.sobel_x, padding=1)
        edge_y = F.conv2d(x, self.sobel_y, padding=1)
        
        edge_magnitude = torch.sqrt(edge_x**2 + edge_y**2)
        
        edge_norm = edge_magnitude / (torch.max(edge_magnitude) + 1e-8)
        edge_attention = torch.sigmoid(edge_norm * 5.0)
        
        return edge_attention

class MLP(nn.Module):
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
        )
    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x
    
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, use_bn=True):
        super().__init__()
        out_channels = out_channels if out_channels else in_channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.activation = ACTIVATION
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)
        return x

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, use_bn=True):
        super().__init__()
        out_channels = out_channels if out_channels else in_channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.activation = ACTIVATION
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()

        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x):
        identity = self.skip(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x += identity
        x = self.activation(x)
        return x

######################################################################
class ConfidenceAwareMaskRefinement(nn.Module):
    def __init__(self, in_channels, mask_channels=3):
        super(ConfidenceAwareMaskRefinement, self).__init__()
        
        # Process input features
        self.feature_process = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            ACTIVATION
        )
        
        # Learn confidence weights for the mask
        self.confidence_estimator = nn.Sequential(
            nn.Conv2d(in_channels + mask_channels, mask_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mask_channels),
            ACTIVATION,
            nn.Conv2d(mask_channels, mask_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Soft refinement module
        self.soft_refinement = nn.Sequential(
            nn.Conv2d(in_channels + mask_channels * 2, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            ACTIVATION,
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(in_channels),
            ACTIVATION
        )
        
        # General refinement (not relying on mask)
        self.general_refinement = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(in_channels),
            ACTIVATION,
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            ACTIVATION
        )
        
        # Fusion module
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            ACTIVATION
        )
    
    def forward(self, x, mask_probs):
        # Process input features
        processed_features = self.feature_process(x)
        
        # Estimate confidence for each mask class
        # Higher confidence where the mask is accurate
        combined = torch.cat([processed_features, mask_probs], dim=1)
        confidence_weights = self.confidence_estimator(combined)
        
        # Apply confidence weights to mask
        weighted_mask = mask_probs * confidence_weights
        
        # Soft refinement using weighted mask
        soft_input = torch.cat([processed_features, mask_probs, weighted_mask], dim=1)
        soft_refined = self.soft_refinement(soft_input)
        
        # General refinement (mask-independent)
        general_refined = self.general_refinement(processed_features)
        
        # Fuse both refinements
        # The network learns to balance between mask-guided and general refinement
        fused = torch.cat([soft_refined, general_refined], dim=1)
        output = self.fusion(fused)
        
        return output

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2, align_corners=True, use_bn=False):
        super().__init__()
        self.block = nn.Sequential(
            #nn.ConvTranspose2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1),
            #nn.BatchNorm2d(in_channels),
            #ACTIVATION,
            nn.Upsample(scale_factor=scale_factor, mode="bilinear", align_corners=align_corners), #False),
            
            ConvBlock(in_channels, out_channels, use_bn=use_bn)
        )
        
    def forward(self, x):
        return self.block(x)

class ConvUp(nn.Sequential):
    def __init__(self, in_channels, out_channels, upsample=False):
        layers = [
            #nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            #nn.GroupNorm(32, out_channels),
            #nn.BatchNorm2d(out_channels),
            #ACTIVATION,
            ConvBlock(in_channels, out_channels)
        ]
        if upsample:
            layers.insert(0, nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False))
        super().__init__(*layers)


class SegmentationBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_upsamples=0):
        super().__init__()
        layers = [ConvUp(in_channels, out_channels, upsample=bool(n_upsamples))]
        for _ in range(1, n_upsamples):
            layers.append(ConvUp(out_channels, out_channels, upsample=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class MergeBlock(nn.Module):
    def __init__(self, policy="add"):
        super().__init__()
        assert policy in ["add", "cat"]
        self.policy = policy

    def forward(self, features):
        if self.policy == "add":
            return torch.stack(features).sum(dim=0)
        elif self.policy == "cat":
            return torch.cat(features, dim=1)

class ZeroBelowThreshold(torch.nn.Module):
    def __init__(self, threshold=0.25):
        super().__init__()
        self.threshold = ((threshold / 0.00025)-MEAN)/STD
        
    def forward(self, x):
        return torch.where(x <= self.threshold, torch.tensor((0.0-MEAN)/STD, device=x.device), x)


def depth_to_normal(depth):
    """
    depth: (B, 1, H, W) tensor (normalized to [0, 1])
    returns: (B, 3, H, W) tensor ? normal map
    """
    B, _, H, W = depth.shape

    # Sobel kernels (for gradient in x and y)
    sobel_x = torch.tensor([[[-1, 0, 1],
                             [-2, 0, 2],
                             [-1, 0, 1]]], dtype=torch.float32).expand(1, 1, 3, 3)
    sobel_y = torch.tensor([[[-1, -2, -1],
                             [ 0,  0,  0],
                             [ 1,  2,  1]]], dtype=torch.float32).expand(1, 1, 3, 3)

    sobel_x = sobel_x.to(depth.device)
    sobel_y = sobel_y.to(depth.device)

    # Compute gradients
    dx = F.conv2d(depth, sobel_x, padding=1)
    dy = F.conv2d(depth, sobel_y, padding=1)

    # Construct normal map (x, y, z)
    #normal = torch.cat([-dx, -dy, torch.ones_like(depth)], dim=1)  # (B, 3, H, W)

    # Normalize
    #norm = torch.norm(normal, dim=1, keepdim=True) + 1e-8
    #normal = normal / norm

    # Optional: scale from [-1, 1] �� [0, 1]
    #normal = (normal + 1.0) / 2.0
    
    norm = torch.sqrt(dx**2 + dy**2 + 1e-8)
    nx = -dx / norm
    ny = -dy / norm
    out = torch.cat([nx, ny, depth], dim=1)
    return out

# 1. Channel and Spatial Attention Mechanisms
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        reduced_channels = max(in_channels // reduction_ratio, 8)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, 1, bias=False),
            ACTIVATION,
            nn.Conv2d(reduced_channels, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Generate channel attention using both average and max pooling
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Generate spatial attention using both average and max pooling
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out)

class Squeeze_Excitation(nn.Module):
    def __init__(self, channel, r=8):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.net = nn.Sequential(
            nn.Linear(channel, channel // r, bias=False),
            ACTIVATION,
            nn.Linear(channel // r, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, inputs):
        b, c, _, _ = inputs.shape
        x = self.pool(inputs).view(b, c)
        x = self.net(x).view(b, c, 1, 1)
        x = inputs * x
        return x
    
# 2. Edge-Aware Smoothing Module
class EdgeAwareSmoothingModule(nn.Module):
    def __init__(self, channels):
        super(EdgeAwareSmoothingModule, self).__init__()
        
        # Edge detection branch
        self.edge_detect = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            ACTIVATION,
            nn.Conv2d(channels, 1, kernel_size=1)
        )
        
        # Multi-scale smoothing branches with different dilation rates
        self.smooth_branch = nn.ModuleList([
            nn.Conv2d(channels, channels // 4, kernel_size=3, padding=d, dilation=d)
            for d in [1, 2, 4, 8]
        ])
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Conv2d(channels + 1, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            ACTIVATION
        )
        
    def forward(self, x):
        # Detect edges
        edge_map = self.edge_detect(x)
        edge_attention = torch.sigmoid(edge_map)
        
        # Apply multi-scale smoothing
        smooth_feats = []
        for conv in self.smooth_branch:
            smooth_feats.append(conv(x))
        
        # Concatenate smoothing results
        smooth_out = torch.cat(smooth_feats, dim=1)
        
        # Fuse edge information with smoothed features
        combined = torch.cat([smooth_out, edge_attention], dim=1)
        out = self.fusion(combined)
        
        # Adaptive blending based on edge information
        # Preserve edges while smoothing flat regions
        final_out = x * edge_attention + out * (1 - edge_attention)
        
        return final_out
        
class SobelEdgeAwareSmoothingModule(nn.Module):
    def __init__(self, channels):
        super(SobelEdgeAwareSmoothingModule, self).__init__()
        
        self.smooth = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(channels),
            ACTIVATION
        )
        
    def forward(self, x, edge):
        if edge.shape[2:] != x.shape[2:]:
            edge = F.interpolate(
                edge, 
                size=x.shape[2:], 
                mode='bilinear', 
                align_corners=False
            )
        
        smooth_out = self.smooth(x)
        
        final_out = x * edge + smooth_out * (1 - edge)
        return final_out

# 3. Residual Upsampling Block
class ResidualUpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(ResidualUpsampleBlock, self).__init__()
        
        #self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True) #
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act = ACTIVATION
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection with 1x1 conv for channel adjustment if needed
        self.skip = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
        ) if in_channels != out_channels else nn.Identity()
        
        self.scale_factor = scale_factor
        
    def forward(self, x):
        # Main branch
        identity = x
        
        # Upsample then process
        x = self.up(x) #F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Upsample skip connection
        identity = F.interpolate(identity, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        identity = self.skip(identity)
        
        # Residual connection
        out += identity
        out = self.act(out)
        
        return out

class LightResidualUpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(LightResidualUpsampleBlock, self).__init__()
        
        #self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True) #
        # Single convolution after upsampling
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = ACTIVATION
        
        # Lightweight skip connection
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        
        self.scale_factor = scale_factor
        
    def forward(self, x):
        # Upsample the input
        x_up = self.up(x) #F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        
        # Main branch: just one conv
        main = self.conv(x_up)
        main = self.bn(main)
        
        # Skip connection
        skip = self.skip(x_up)
        
        # Combine with skip connection
        out = main + skip
        out = self.act(out)
        
        return out

# Even lighter version with pixel shuffle
class PixelShuffleUpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(PixelShuffleUpsampleBlock, self).__init__()
        
        # Pixel shuffle for efficient upsampling
        self.conv = nn.Conv2d(
            in_channels, 
            out_channels * (scale_factor ** 2), 
            kernel_size=3, 
            padding=1
        )
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = ACTIVATION
        
    def forward(self, x):
        out = self.conv(x)
        out = self.pixel_shuffle(out)
        out = self.bn(out)
        out = self.act(out)
        return out
        
class MaskAttentionModule(nn.Module):
    def __init__(self, feature_channels, mask_channels):
        super(MaskAttentionModule, self).__init__()
        
        self.conv_reduce = nn.Conv2d(mask_channels, mask_channels, kernel_size=1)
        self.conv_attention = nn.Conv2d(mask_channels, feature_channels, kernel_size=1)
        
        self.conv_fuse = nn.Conv2d(feature_channels + mask_channels, feature_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(feature_channels)
        self.activation = ACTIVATION
        
    def forward(self, features, mask):
        mask = self.conv_reduce(mask)
        attention = self.conv_attention(mask)
        
        # Apply attention mechanism
        enhanced_features = features * (1 + attention)
        
        # Concatenate and fuse
        fused = torch.cat([enhanced_features, mask], dim=1)
        out = self.conv_fuse(fused)
        out = self.bn(out)
        out = self.activation(out)
        
        return out
        


class ImprovedSegFPNHead(nn.Module):
    def __init__(self, dims, decoder_dim, num_classes, merge_policy="cat"):
        super(ImprovedSegFPNHead, self).__init__()
        
        c1, c2, c3, c4 = dims
        self.num_classes = num_classes
        self.merge_policy = merge_policy
        
        self.dropout = nn.Dropout2d(0.2)
        
        self.fpn_in = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, decoder_dim, kernel_size=1),
                nn.BatchNorm2d(decoder_dim),
                ACTIVATION,
            )
            for c in [c1, c2, c3, c4]
        ])
        
        self.fpn_out = nn.ModuleList([
            ConvBlock(decoder_dim, decoder_dim)
            for _ in range(4)
        ])
        
        self.merge = MergeBlock(merge_policy)
        
        if merge_policy == "cat":
            final_in_channels = decoder_dim * 4
        else:
            final_in_channels = decoder_dim
            
        self.merge_conv = nn.Sequential(
            ConvBlock(final_in_channels, decoder_dim)
        )
        
        self.seg_head = nn.Sequential(
            nn.Conv2d(decoder_dim, decoder_dim//4, kernel_size=3, padding=1),
            nn.BatchNorm2d(decoder_dim//4),
            ACTIVATION,

            self.dropout,
            nn.Conv2d(decoder_dim//4, num_classes, kernel_size=1)
        )
        
    def forward(self, features):
        c1, c2, c3, c4 = features
        
        f4 = self.fpn_in[3](c4)
        f3 = self.fpn_in[2](c3)
        f2 = self.fpn_in[1](c2)
        f1 = self.fpn_in[0](c1)
        
        f3 = f3 + F.interpolate(f4, size=f3.shape[2:], mode='bilinear', align_corners=False)
        f2 = f2 + F.interpolate(f3, size=f2.shape[2:], mode='bilinear', align_corners=False)
        f1 = f1 + F.interpolate(f2, size=f1.shape[2:], mode='bilinear', align_corners=False)
        
        p4 = self.fpn_out[3](f4)
        p3 = self.fpn_out[2](f3)
        p2 = self.fpn_out[1](f2)
        p1 = self.fpn_out[0](f1)
        
        p4 = self.dropout(p4)
        p3 = self.dropout(p3)
        p2 = self.dropout(p2)
        p1 = self.dropout(p1)
        
        p2 = F.interpolate(p2, size=p1.shape[2:], mode='bilinear', align_corners=False)
        p3 = F.interpolate(p3, size=p1.shape[2:], mode='bilinear', align_corners=False)
        p4 = F.interpolate(p4, size=p1.shape[2:], mode='bilinear', align_corners=False)
        
        out = self.merge([p1, p2, p3, p4])
        
        out = self.merge_conv(out)
        
        out = F.interpolate(out, scale_factor=4, mode='bilinear', align_corners=True)
        out = self.seg_head(out)
        
        return out
    
class ImprovedDepthFPNHead(nn.Module):
    def __init__(self, dims, decoder_dim, merge_policy="cat", channel_reduction=2):
        super(ImprovedDepthFPNHead, self).__init__()
        c1, c2, c3, c4 = dims

        self.merge_policy = merge_policy
        self.dropout = nn.Dropout2d(0.2)

        self.fpn_in = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, decoder_dim, kernel_size=1),
                nn.BatchNorm2d(decoder_dim),
                ACTIVATION,
            )
            for c in [c1, c2, c3, c4]
        ])
        
        self.fpn_out = nn.ModuleList([
            ConvBlock(decoder_dim, decoder_dim)
            for _ in range(4)
        ])

        self.merge = MergeBlock(merge_policy)
        
        if merge_policy == "cat":
            final_in_channels = decoder_dim * 4
        else:
            final_in_channels = decoder_dim
            
        self.merge_conv = nn.Sequential(
            ConvBlock(final_in_channels, decoder_dim)
            # ConvLite
        )
        
        self.upsample1 = UpsampleBlock(decoder_dim, decoder_dim // channel_reduction, align_corners=True, use_bn=False)
        self.upsample2 = UpsampleBlock(decoder_dim // channel_reduction, decoder_dim // (channel_reduction ** 2), align_corners=True, use_bn=False)
        
        self.depth_head = nn.Sequential(
            nn.Conv2d(decoder_dim // (channel_reduction ** 2), decoder_dim // (channel_reduction ** 2), kernel_size=3, padding=1),
            ACTIVATION,
            
            self.dropout,
            nn.Conv2d(decoder_dim // (channel_reduction ** 2), 1, kernel_size=1),
        )
        
        self.mask_proj = nn.Sequential(
            nn.Conv2d(decoder_dim // (channel_reduction ** 2) + 3, decoder_dim // (channel_reduction ** 2), kernel_size=1),
            nn.BatchNorm2d(decoder_dim // (channel_reduction ** 2)),
            ACTIVATION
        )
        
        self.mask_conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c+3, c, kernel_size=1),
                nn.BatchNorm2d(c),
                ACTIVATION
            ) for c in dims
        ])
        
    def forward(self, features, mask=None):
        if mask is not None:
            processed_features = []
            for idx, feature in enumerate(features):
                #feature = self.adapt_block[idx](feature)
                
                resized_mask = F.interpolate(mask, size=feature.shape[2:], mode='bilinear', align_corners=False)
                feature = torch.cat([feature, resized_mask], dim=1)
                feature = self.mask_conv_layers[idx](feature)
                processed_features.append(feature)
            
            c1, c2, c3, c4 = processed_features
        else:
            c1, c2, c3, c4 = features
        
        f4 = self.fpn_in[3](c4)
        f3 = self.fpn_in[2](c3)
        f2 = self.fpn_in[1](c2)
        f1 = self.fpn_in[0](c1)

        f3 = f3 + F.interpolate(f4, size=f3.shape[2:], mode='bilinear', align_corners=False)
        f2 = f2 + F.interpolate(f3, size=f2.shape[2:], mode='bilinear', align_corners=False)
        f1 = f1 + F.interpolate(f2, size=f1.shape[2:], mode='bilinear', align_corners=False)
        
        p4 = self.fpn_out[3](f4)
        p3 = self.fpn_out[2](f3)
        p2 = self.fpn_out[1](f2)
        p1 = self.fpn_out[0](f1)
        
        p4 = self.dropout(p4)
        p3 = self.dropout(p3)
        p2 = self.dropout(p2)
        p1 = self.dropout(p1)
        
        
        p2 = F.interpolate(p2, size=p1.shape[2:], mode='bilinear', align_corners=False)
        p3 = F.interpolate(p3, size=p1.shape[2:], mode='bilinear', align_corners=False)
        p4 = F.interpolate(p4, size=p1.shape[2:], mode='bilinear', align_corners=False)
        
        out = self.merge([p1, p2, p3, p4])
        
        out = self.merge_conv(out)
                        
        out = self.upsample1(out)
        out = self.upsample2(out)
        
        if mask is not None:
            resized_mask = F.interpolate(mask, size=out.shape[2:], mode='bilinear', align_corners=True)
            out = torch.cat([out, resized_mask], dim=1)
            out = self.mask_proj(out)
        
        out = self.depth_head(out)
        
        return out

from torchvision.ops import DeformConv2d


class FeatureSelectionModule(nn.Module):
    """Feature selection module with channel attention"""
    def __init__(self, in_channels, out_channels, use_bn=True):
        super().__init__()
        self.conv_atten = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.bn_atten = nn.BatchNorm2d(in_channels) if use_bn else nn.Identity()
        self.sigmoid = nn.Sigmoid()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        
    def forward(self, x):
        # Channel attention
        atten = F.avg_pool2d(x, x.size()[2:])
        atten = self.conv_atten(atten)
        atten = self.bn_atten(atten)
        atten = self.sigmoid(atten)
        
        # Feature modulation
        feat = x * atten
        x = x + feat
        
        # Output projection
        feat = self.conv(x)
        feat = self.bn(feat)
        return feat


class FeatureAlignmentModule(nn.Module):
    """Feature alignment module using torchvision's deformable convolution"""
    def __init__(self, in_channels=128, out_channels=128, use_bn=True):
        super().__init__()
        self.lateral_conv = FeatureSelectionModule(in_channels, out_channels, use_bn=use_bn)
        
        # Offset prediction for deformable convolution
        self.offset_conv = nn.Conv2d(
            out_channels * 2, 
            2 * 3 * 3,  # 2 * kernel_size * kernel_size for x,y offsets
            kernel_size=1, 
            stride=1, 
            padding=0, 
            bias=True
        )
        
        # Deformable convolution from torchvision
        self.dcn = DeformConv2d(
            out_channels, 
            out_channels, 
            kernel_size=3, 
            stride=1, 
            padding=1,
            groups=1,
            bias=True
        )
        
        self.activation = ACTIVATION
        
        # Initialize offset to zero
        nn.init.zeros_(self.offset_conv.weight)
        nn.init.zeros_(self.offset_conv.bias)
        
    def forward(self, feat_low_level, feat_high_level):
        """
        Args:
            feat_low_level: Low-level features (higher resolution)
            feat_high_level: High-level features (lower resolution)
        """
        # Upsample high-level features if needed
        if feat_low_level.size()[2:] != feat_high_level.size()[2:]:
            feat_up = F.interpolate(
                feat_high_level, 
                size=feat_low_level.size()[2:], 
                mode='bilinear', 
                align_corners=False
            )
        else:
            feat_up = feat_high_level
            
        # Lateral connection
        feat_arm = self.lateral_conv(feat_low_level)
        
        # Compute offset for deformable convolution
        offset_input = torch.cat([feat_arm, feat_up * 2], dim=1)
        offset = self.offset_conv(offset_input)
        
        # Apply deformable convolution
        feat_align = self.dcn(feat_up, offset)
        feat_align = self.activation(feat_align)
        
        # Residual connection
        return feat_align + feat_arm
    
class SegFaPNHead(nn.Module):
    def __init__(self, dims, decoder_dim, num_classes, merge_policy="cat"):
        super(SegFaPNHead, self).__init__()     
        c1, c2, c3, c4 = dims
        c0 = c1

        self.num_classes = num_classes
        self.merge_policy = merge_policy
        
        self.dropout = nn.Dropout2d(0.2)

        self.context_block = ASPP(c4, decoder_dim)
        
        # FPN input projections
        self.fpn_in = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, decoder_dim, kernel_size=1),
                nn.BatchNorm2d(decoder_dim),
                ACTIVATION,
            )
            for c in [c1, c2, c3]
        ])
        
        # Feature alignment modules (instead of simple addition)
        self.align_modules = nn.ModuleList([
            FeatureAlignmentModule(decoder_dim, decoder_dim)
            for _ in range(3)  # for f3, f2, f1
        ])
        
        # FPN output convolutions
        self.fpn_out = nn.ModuleList([
            ConvBlock(decoder_dim, decoder_dim)
            for _ in range(4)
        ])
        
        self.merge = MergeBlock(merge_policy)
        
        if merge_policy == "cat":
            final_in_channels = decoder_dim * 4
        else:
            final_in_channels = decoder_dim
            
        self.merge_conv = nn.Sequential(
            ConvBlock(final_in_channels, decoder_dim)
            # ConvLite
        )

        channel_reduction = 2

        self.upsample1 = UpsampleBlock(decoder_dim, decoder_dim // channel_reduction, align_corners=False, use_bn=True) # nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False) # 
        self.upsample2 = UpsampleBlock(decoder_dim // channel_reduction, decoder_dim // (channel_reduction ** 2), align_corners=False, use_bn=True) # nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False) # 

        self.c0_merge = ConvBlock(decoder_dim // (channel_reduction ** 2)+c0, decoder_dim // (channel_reduction ** 2))

        self.seg_head = nn.Sequential(
            nn.Conv2d(decoder_dim // (channel_reduction ** 2), decoder_dim // (channel_reduction ** 2), kernel_size=3, padding=1),
            nn.BatchNorm2d(decoder_dim//4),
            ACTIVATION,

            self.dropout,
            nn.Conv2d(decoder_dim // (channel_reduction ** 2), num_classes, kernel_size=1),
        )

    def forward(self, features, c0):
        c1, c2, c3, c4 = features
        
        # Lateral connections
        f4 = self.context_block(c4) # self.fpn_in[3](c4)
        f3 = self.fpn_in[2](c3)
        f2 = self.fpn_in[1](c2)
        f1 = self.fpn_in[0](c1)
        
        # Top-down pathway with feature alignment (FAN)
        f3 = self.align_modules[0](f3, f4)  # Instead of f3 + upsample(f4)
        f2 = self.align_modules[1](f2, f3)  # Instead of f2 + upsample(f3)
        f1 = self.align_modules[2](f1, f2)  # Instead of f1 + upsample(f2)
        
        # Apply output convolutions
        p4 = self.fpn_out[3](f4)
        p3 = self.fpn_out[2](f3)
        p2 = self.fpn_out[1](f2)
        p1 = self.fpn_out[0](f1)
        
        p2 = F.interpolate(p2, size=p1.shape[2:], mode='bilinear', align_corners=False)
        p3 = F.interpolate(p3, size=p1.shape[2:], mode='bilinear', align_corners=False)
        p4 = F.interpolate(p4, size=p1.shape[2:], mode='bilinear', align_corners=False)
        
        p4 = self.dropout(p4)
        p3 = self.dropout(p3)
        p2 = self.dropout(p2)
        p1 = self.dropout(p1)

        # Merge features
        out = self.merge([p1, p2, p3, p4])
        out = self.merge_conv(out)
        
        out = self.upsample1(out)
        out = self.upsample2(out)

        # Final upsampling and prediction
        #out = F.interpolate(out, scale_factor=4, mode='bilinear', align_corners=True)
        
        out = self.c0_merge(torch.cat([out, c0], dim=1))

        out = self.seg_head(out)
        return out
    
class DepthFaPNHead(nn.Module):
    def __init__(self, dims, decoder_dim, merge_policy="cat", channel_reduction=2):
        super(DepthFaPNHead, self).__init__()
        c1, c2, c3, c4 = dims
        c0 = c1

        self.merge_policy = merge_policy
        self.dropout = nn.Dropout2d(0.2)

        self.context_block = ASPP(c4, decoder_dim)
        # FPN input projections
        self.fpn_in = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, decoder_dim, kernel_size=1),
                nn.BatchNorm2d(decoder_dim),
                ACTIVATION,
            )
            for c in [c1, c2, c3]
        ])
        
        # Feature alignment modules (FAN key component)
        self.align_modules = nn.ModuleList([
            FeatureAlignmentModule(decoder_dim, decoder_dim)
            for _ in range(3)  # for f3, f2, f1
        ])
        
        # FPN output convolutions
        self.fpn_out = nn.ModuleList([
            ConvBlock(decoder_dim, decoder_dim)
            for _ in range(4)
        ])

        self.merge = MergeBlock(merge_policy)
        
        if merge_policy == "cat":
            final_in_channels = decoder_dim * 4
        else:
            final_in_channels = decoder_dim
            
        self.merge_conv = nn.Sequential(
            ConvBlock(final_in_channels, decoder_dim)
        )
        
        self.upsample1 = UpsampleBlock(decoder_dim, decoder_dim // channel_reduction, align_corners=False, use_bn=True) # nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False) # 
        self.upsample2 = UpsampleBlock(decoder_dim // channel_reduction, decoder_dim // (channel_reduction ** 2), align_corners=False, use_bn=True) # nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False) # 

        self.c0_merge = ConvBlock(decoder_dim // (channel_reduction ** 2)+c0, decoder_dim // (channel_reduction ** 2))

        self.depth_head = nn.Sequential(
            nn.Conv2d(decoder_dim // (channel_reduction ** 2), decoder_dim // (channel_reduction ** 2), kernel_size=3, padding=1),
            ACTIVATION,

            self.dropout,
            nn.Conv2d(decoder_dim // (channel_reduction ** 2), 1, kernel_size=1),
        )

        self.mask_proj = nn.Sequential(
            nn.Conv2d(decoder_dim // (channel_reduction ** 2) + 3, decoder_dim // (channel_reduction ** 2), kernel_size=1),
            nn.BatchNorm2d(decoder_dim // (channel_reduction ** 2)),
            ACTIVATION
        )

        self.mask_conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c+3, c, kernel_size=1),
                nn.BatchNorm2d(c),
                ACTIVATION
            ) for c in dims
        ])

    def forward(self, features, mask=None, c0=None):
        # Process features with mask if available
        if mask is not None:
            processed_features = []
            for idx, feature in enumerate(features):
                resized_mask = F.interpolate(mask, size=feature.shape[2:], mode='bilinear', align_corners=False)
                feature = torch.cat([feature, resized_mask], dim=1)
                feature = self.mask_conv_layers[idx](feature)
                processed_features.append(feature)
            
            c1, c2, c3, c4 = processed_features
        else:
            c1, c2, c3, c4 = features
        
        # Lateral connections
        f4 = self.context_block(c4) # self.fpn_in[3](c4)
        f3 = self.fpn_in[2](c3)
        f2 = self.fpn_in[1](c2)
        f1 = self.fpn_in[0](c1)

        # Top-down pathway with feature alignment (FAN)
        f3 = self.align_modules[0](f3, f4)  # Deformable alignment instead of addition
        f2 = self.align_modules[1](f2, f3)
        f1 = self.align_modules[2](f1, f2)
        
        # Apply output convolutions
        p4 = self.fpn_out[3](f4)
        p3 = self.fpn_out[2](f3)
        p2 = self.fpn_out[1](f2)
        p1 = self.fpn_out[0](f1)
        
        # Resize all to p1 size
        p2 = F.interpolate(p2, size=p1.shape[2:], mode='bilinear', align_corners=False)
        p3 = F.interpolate(p3, size=p1.shape[2:], mode='bilinear', align_corners=False)
        p4 = F.interpolate(p4, size=p1.shape[2:], mode='bilinear', align_corners=False)
        
        p4 = self.dropout(p4)
        p3 = self.dropout(p3)
        p2 = self.dropout(p2)
        p1 = self.dropout(p1)

        # Merge features
        out = self.merge([p1, p2, p3, p4])
        out = self.merge_conv(out)
        
        # Upsampling
        out = self.upsample1(out)
        out = self.upsample2(out)
        
        out = self.c0_merge(torch.cat([out, c0], dim=1))

        # Add mask information if available
        if mask is not None:
            resized_mask = F.interpolate(mask, size=out.shape[2:], mode='bilinear', align_corners=False)
            out = torch.cat([out, resized_mask], dim=1)
            out = self.mask_proj(out)
        
        # Final depth prediction
        out = self.depth_head(out)
        
        return out



class BiFPNDepthHead(nn.Module):
    def __init__(self, dims, decoder_dim, merge_policy="add", fpn_num_layers=2):
        super(BiFPNDepthHead, self).__init__()
        c1, c2, c3, c4 = dims
        segmentation_channels = decoder_dim // 2
        self.fpn_num_layers = fpn_num_layers
        
        # Output channels after merging
        final_in_channels = segmentation_channels * 4 if merge_policy == "cat" else segmentation_channels
                
        self.merge_policy = merge_policy
        # Lateral connections (input projections)
        self.lat1 = nn.Conv2d(c1, decoder_dim, kernel_size=1)
        self.lat2 = nn.Conv2d(c2, decoder_dim, kernel_size=1)
        self.lat3 = nn.Conv2d(c3, decoder_dim, kernel_size=1)
        self.lat4 = nn.Conv2d(c4, decoder_dim, kernel_size=1)
        
        # Channel and spatial attention for each feature level
        self.channel_attention = nn.ModuleList([
            ChannelAttention(decoder_dim) for _ in range(4)
        ])
        
        self.spatial_attention = nn.ModuleList([
            SpatialAttention() for _ in range(4)
        ])
        
        # BiFPN layers
        self.bifpn_layers = nn.ModuleList([
            BiFPNLayer(decoder_dim) for _ in range(fpn_num_layers)
        ])
        
        # Segmentation blocks for each level
        self.seg_blocks = nn.ModuleList([
            SegmentationBlock(decoder_dim, segmentation_channels, n)
            for n in [3, 2, 1, 0]
        ])
        
        # Feature merging
        self.merge = MergeBlock(merge_policy)
        
        # For cat merge policy, add a projection layer
        if merge_policy == "cat":
            self.merge_conv = nn.Sequential(
                nn.Conv2d(final_in_channels, segmentation_channels, kernel_size=1),
                nn.BatchNorm2d(segmentation_channels),
                ACTIVATION
            )
        
        # Upsampling blocks
        self.upsample1 = UpsampleBlock(segmentation_channels, segmentation_channels)
        self.upsample2 = UpsampleBlock(segmentation_channels, segmentation_channels)
        
        # Mask fusion for final output
        self.mask_proj = nn.Sequential(
            nn.Conv2d(segmentation_channels+3, segmentation_channels, kernel_size=1),
            nn.BatchNorm2d(segmentation_channels),
            ACTIVATION
        )
        
        # Final convolution with dilated convolutions for multi-scale context
        self.final_conv = nn.Sequential(
            nn.Conv2d(segmentation_channels, segmentation_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(segmentation_channels),
            ACTIVATION,
            
            # Multi-scale context with dilated convolutions
            nn.Conv2d(segmentation_channels, segmentation_channels, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(segmentation_channels),
            ACTIVATION,
            
            nn.Dropout2d(0.2),
            nn.Conv2d(segmentation_channels, 1, kernel_size=1)
        )
        
        # Mask processing layers
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim+3, dim, kernel_size=1),
                nn.BatchNorm2d(dim),
                ACTIVATION
            ) for dim in dims
        ])
    
    def forward(self, inputs, mask=None):
        # Process input features with mask if available
        if mask is not None:
            processed_layers = []
            for layer, conv in zip(inputs, self.conv_layers):
                resized_mask = F.interpolate(mask, size=layer.shape[2:], mode='bilinear', align_corners=False) 
                layer = torch.cat([layer, resized_mask], dim=1)
                layer = conv(layer)
                processed_layers.append(layer)
        else:
            processed_layers = inputs
            
        c1, c2, c3, c4 = processed_layers
        
        # Initial feature projection
        p1 = self.lat1(c1)  # Highest resolution
        p2 = self.lat2(c2)
        p3 = self.lat3(c3)
        p4 = self.lat4(c4)  # Lowest resolution
        
        # Apply attention mechanisms
        p1 = p1 * self.channel_attention[0](p1)
        p1 = p1 * self.spatial_attention[0](p1)
        
        p2 = p2 * self.channel_attention[1](p2)
        p2 = p2 * self.spatial_attention[1](p2)
        
        p3 = p3 * self.channel_attention[2](p3)
        p3 = p3 * self.spatial_attention[2](p3)
        
        p4 = p4 * self.channel_attention[3](p4)
        p4 = p4 * self.spatial_attention[3](p4)
        
        # BiFPN processing
        features = [p1, p2, p3, p4]
        for bifpn in self.bifpn_layers:
            features = bifpn(features)
            
        p1, p2, p3, p4 = features
        
        # Process with segmentation blocks
        s4 = self.seg_blocks[0](p4)
        s3 = self.seg_blocks[1](p3)
        s2 = self.seg_blocks[2](p2)
        s1 = self.seg_blocks[3](p1)
        
        # Merge features
        out = self.merge([s4, s3, s2, s1])
        
        # Apply projection for cat merge policy
        if self.merge_policy == "cat":
            out = self.merge_conv(out)
        
        # Upsampling
        out = self.upsample1(out)
        out = self.upsample2(out)
        
        # Add mask information to the output features
        if mask is not None:
            resized_mask = F.interpolate(mask, size=out.shape[2:], mode='bilinear', align_corners=False)
            out = torch.cat([out, resized_mask], dim=1)
            out = self.mask_proj(out)
        
        # Final convolution
        out = self.final_conv(out)
        
        return out

class BiFPNSegHead(nn.Module):
    def __init__(self, dims, decoder_dim, num_classes=3, merge_policy="add", fpn_num_layers=2):
        super(BiFPNSegHead, self).__init__()
        c1, c2, c3, c4 = dims
        segmentation_channels = decoder_dim // 2
        self.fpn_num_layers = fpn_num_layers
        self.num_classes = num_classes
        
        # Output channels after merging
        final_in_channels = segmentation_channels * 4 if merge_policy == "cat" else segmentation_channels
                
        self.merge_policy = merge_policy
        # Lateral connections (input projections)
        self.lat1 = nn.Conv2d(c1, decoder_dim, kernel_size=1)
        self.lat2 = nn.Conv2d(c2, decoder_dim, kernel_size=1)
        self.lat3 = nn.Conv2d(c3, decoder_dim, kernel_size=1)
        self.lat4 = nn.Conv2d(c4, decoder_dim, kernel_size=1)
        
        # Channel and spatial attention for each feature level
        self.channel_attention = nn.ModuleList([
            ChannelAttention(decoder_dim) for _ in range(4)
        ])
        
        self.spatial_attention = nn.ModuleList([
            SpatialAttention() for _ in range(4)
        ])
        
        # BiFPN layers
        self.bifpn_layers = nn.ModuleList([
            BiFPNLayer(decoder_dim) for _ in range(fpn_num_layers)
        ])
        
        # Segmentation blocks for each level
        self.seg_blocks = nn.ModuleList([
            SegmentationBlock(decoder_dim, segmentation_channels, n)
            for n in [3, 2, 1, 0]
        ])
        
        # Feature merging
        self.merge = MergeBlock(merge_policy)
        
        # For cat merge policy, add a projection layer
        if merge_policy == "cat":
            self.merge_conv = nn.Sequential(
                nn.Conv2d(final_in_channels, segmentation_channels, kernel_size=1),
                nn.BatchNorm2d(segmentation_channels),
                ACTIVATION
            )
        
        # Final classification head with class-aware features
        self.final_conv = nn.Sequential(
            nn.Conv2d(segmentation_channels, 
                     segmentation_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(segmentation_channels),
            ACTIVATION,
            
            # Multi-scale context with dilated convolutions
            nn.Conv2d(segmentation_channels, segmentation_channels, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(segmentation_channels),
            ACTIVATION,
            
            nn.Dropout2d(0.2),
            nn.Conv2d(segmentation_channels, num_classes, kernel_size=1)
        )
        
        # Class-aware attention module
        self.class_attention = nn.Sequential(
            nn.Conv2d(segmentation_channels, num_classes, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, inputs):
        c1, c2, c3, c4 = inputs
        
        # Initial feature projection
        p1 = self.lat1(c1)  # Highest resolution
        p2 = self.lat2(c2)
        p3 = self.lat3(c3)
        p4 = self.lat4(c4)  # Lowest resolution
        
        # Apply attention mechanisms
        p1 = p1 * self.channel_attention[0](p1)
        p1 = p1 * self.spatial_attention[0](p1)
        
        p2 = p2 * self.channel_attention[1](p2)
        p2 = p2 * self.spatial_attention[1](p2)
        
        p3 = p3 * self.channel_attention[2](p3)
        p3 = p3 * self.spatial_attention[2](p3)
        
        p4 = p4 * self.channel_attention[3](p4)
        p4 = p4 * self.spatial_attention[3](p4)
        
        # BiFPN processing
        features = [p1, p2, p3, p4]
        for bifpn in self.bifpn_layers:
            features = bifpn(features)
            
        p1, p2, p3, p4 = features
        
        # Process with segmentation blocks
        s4 = self.seg_blocks[0](p4)
        s3 = self.seg_blocks[1](p3)
        s2 = self.seg_blocks[2](p2)
        s1 = self.seg_blocks[3](p1)
        
        # Merge features
        out = self.merge([s4, s3, s2, s1])
        
        # Apply projection for cat merge policy
        if self.merge_policy == "cat":
            out = self.merge_conv(out)
        
        # Apply class-aware attention (optional enhancement)
        # class_weights = self.class_attention(out)
        # out = out * class_weights  # Apply attention weights
        
        # Final convolution
        out = self.final_conv(out)
        
        # Upsample to original size
        out = F.interpolate(out, scale_factor=4, mode='bilinear', align_corners=False)
        
        return out
        
        
class BiFPNLayer(nn.Module):
    """
    Bidirectional Feature Pyramid Network layer
    """
    def __init__(self, decoder_dim):
        super(BiFPNLayer, self).__init__()
        
        # Weights for feature fusion (learned)
        self.w1 = nn.Parameter(torch.ones(2, dtype=torch.float32))
        self.w2 = nn.Parameter(torch.ones(3, dtype=torch.float32))
        self.w3 = nn.Parameter(torch.ones(3, dtype=torch.float32))
        self.w4 = nn.Parameter(torch.ones(2, dtype=torch.float32))
        self.epsilon = 1e-4
        
        # Top-down pathway (p4 -> p3 -> p2 -> p1)
        self.td_conv3 = SepConvBN(decoder_dim, decoder_dim)
        self.td_conv2 = SepConvBN(decoder_dim, decoder_dim)
        self.td_conv1 = SepConvBN(decoder_dim, decoder_dim)
        
        # Bottom-up pathway (p1 -> p2 -> p3 -> p4)
        self.bu_conv2 = SepConvBN(decoder_dim, decoder_dim)
        self.bu_conv3 = SepConvBN(decoder_dim, decoder_dim)
        self.bu_conv4 = SepConvBN(decoder_dim, decoder_dim)
    
    def _weight_fusion(self, weights, inputs):
        """Apply weighted fusion with normalized weights"""
        w = F.softmax(weights, dim=0)
        return sum(w[i] * inputs[i] for i in range(len(inputs)))
    
    def forward(self, features):
        p1, p2, p3, p4 = features
        
        # Top-down pathway
        # p4_td is just p4
        p4_td = p4
        
        # p3_td = Conv(w1[0]*p3 + w1[1]*Upsample(p4_td))
        p3_in = [p3, F.interpolate(p4_td, size=p3.shape[2:], mode='bilinear', align_corners=False)]
        p3_td = self.td_conv3(self._weight_fusion(self.w1, p3_in))
        
        # p2_td = Conv(w2[0]*p2 + w2[1]*Upsample(p3_td))
        p2_in = [p2, F.interpolate(p3_td, size=p2.shape[2:], mode='bilinear', align_corners=False)]
        p2_td = self.td_conv2(self._weight_fusion(self.w1, p2_in))
        
        # p1_td = Conv(w3[0]*p1 + w3[1]*Upsample(p2_td))
        p1_in = [p1, F.interpolate(p2_td, size=p1.shape[2:], mode='bilinear', align_corners=False)]
        p1_out = self.td_conv1(self._weight_fusion(self.w1, p1_in))
        
        # Bottom-up pathway
        # p1_out is already computed
        
        # p2_out = Conv(w4[0]*p2 + w4[1]*p2_td + w4[2]*Downsample(p1_out))
        p2_in = [p2, p2_td, F.interpolate(p1_out, size=p2.shape[2:], mode='bilinear', align_corners=False)]
        p2_out = self.bu_conv2(self._weight_fusion(self.w2, p2_in))
        
        # p3_out = Conv(w5[0]*p3 + w5[1]*p3_td + w5[2]*Downsample(p2_out))
        p3_in = [p3, p3_td, F.interpolate(p2_out, size=p3.shape[2:], mode='bilinear', align_corners=False)]
        p3_out = self.bu_conv3(self._weight_fusion(self.w2, p3_in))
        
        # p4_out = Conv(w6[0]*p4 + w6[1]*Downsample(p3_out))
        p4_in = [p4, F.interpolate(p3_out, size=p4.shape[2:], mode='bilinear', align_corners=False)]
        p4_out = self.bu_conv4(self._weight_fusion(self.w1, p4_in))
        
        return [p1_out, p2_out, p3_out, p4_out]


class SepConvBN(nn.Module):
    """Depthwise separable convolution with BatchNorm and Activation"""
    def __init__(self, in_channels, out_channels):
        super(SepConvBN, self).__init__()
        self.conv = nn.Sequential(
            # Depthwise convolution
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False),
            # Pointwise convolution
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            ACTIVATION
        )
    
    def forward(self, x):
        return self.conv(x)
        

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, rates=[6, 12, 18]):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                ACTIVATION
            )
        ] + [
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(out_channels),
                ACTIVATION
            ) for rate in rates
        ])
        
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            ACTIVATION
        )

        self.project = nn.Sequential(
            nn.Conv2d(out_channels * (len(rates) + 2), out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            ACTIVATION
        )

    def forward(self, x):
        size = x.shape[2:]
        feats = [branch(x) for branch in self.branches]
        
        gp = self.global_pool(x)
        gp = F.interpolate(gp, size=size, mode='bilinear', align_corners=False)
        
        feats.append(gp)
        return self.project(torch.cat(feats, dim=1))


    
class PPM(nn.Module):
    """
    Pyramid Pooling Module (PPM)
    """
    def __init__(self, in_channels, out_channels, pool_sizes=[1, 2, 3, 6]):
        super(PPM, self).__init__()
        self.pool_layers = nn.ModuleList()
        
        for pool_size in pool_sizes:
            self.pool_layers.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(pool_size),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                ACTIVATION
            ))

        total_channels = in_channels + len(pool_sizes) * out_channels
        self.bottleneck = nn.Sequential(
            nn.Conv2d(total_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            ACTIVATION
            # ConvBlock(total_channels, out_channels)
        )
    def forward(self, x):
        h, w = x.size(2), x.size(3)
        features = [x]
        
        for pool_layer in self.pool_layers:
            feat = pool_layer(x)
            features.append(F.interpolate(feat, size=(h, w), mode='bilinear', align_corners=False))
        out = torch.cat(features, dim=1)
        out = self.bottleneck(out)
        return out

class ConvLite(nn.Module):
    def __init__(self, in_channels, out_channels=None, use_bn=True):
        super().__init__()
        out_channels = out_channels if out_channels else in_channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.activation = ACTIVATION

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        return x

class UpsampleLite(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2, align_corners=True, use_bn=False):
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode="bilinear", align_corners=align_corners),
            
            ConvLite(in_channels, out_channels, use_bn=use_bn)
        )
        
    def forward(self, x):
        return self.block(x)


class UperNetSegHead(nn.Module):
    def __init__(self, dims, decoder_dim, num_classes, merge_policy="cat", channel_reduction=1):
        super(UperNetSegHead, self).__init__()
        c1, c2, c3, c4 = dims
        c0 = c1 #// 2
        
        self.num_classes = num_classes
        self.merge_policy = merge_policy
        
        self.dropout = nn.Dropout2d(0.2)

        self.context_block = PPM(c4, decoder_dim)

        self.fpn_in = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, decoder_dim, kernel_size=1),
                nn.BatchNorm2d(decoder_dim),
                ACTIVATION,
            )
            for c in [c1, c2, c3]
        ])
        
        self.fpn_out = nn.ModuleList([
            ConvBlock(decoder_dim, decoder_dim)
            for _ in range(4)
        ])
        
        self.merge = MergeBlock(merge_policy)
        
        if merge_policy == "cat":
            final_in_channels = decoder_dim * 4
        else:
            final_in_channels = decoder_dim
            
        self.merge_conv = nn.Sequential(
            ConvBlock(final_in_channels, decoder_dim)
        )

        self.upsample1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.upsample2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        if merge_policy == "cat":
            c0_in_channels = decoder_dim * 2
        else:
            c0_in_channels = decoder_dim

        self.c0_merge = ConvBlock(c0_in_channels, decoder_dim // (channel_reduction ** 2))

        self.seg_head = nn.Sequential(
            nn.Conv2d(decoder_dim // (channel_reduction ** 2), decoder_dim // (channel_reduction ** 2), kernel_size=3, padding=1),
            nn.BatchNorm2d(decoder_dim//(channel_reduction ** 2)),
            ACTIVATION,
            
            self.dropout,
            nn.Conv2d(decoder_dim // (channel_reduction ** 2), self.num_classes, kernel_size=1),
        )
        

    def forward(self, features, c0):
        c1, c2, c3, c4 = features

        f4 = self.context_block(c4)
        f3 = self.fpn_in[2](c3)
        f2 = self.fpn_in[1](c2)
        f1 = self.fpn_in[0](c1)
        
        f3 = f3 + F.interpolate(f4, size=f3.shape[2:], mode='bilinear', align_corners=False)
        f2 = f2 + F.interpolate(f3, size=f2.shape[2:], mode='bilinear', align_corners=False)
        f1 = f1 + F.interpolate(f2, size=f1.shape[2:], mode='bilinear', align_corners=False)
        
        p4 = self.fpn_out[3](f4)
        p3 = self.fpn_out[2](f3)
        p2 = self.fpn_out[1](f2)
        p1 = self.fpn_out[0](f1)
        
        p2 = F.interpolate(p2, size=p1.shape[2:], mode='bilinear', align_corners=False)
        p3 = F.interpolate(p3, size=p1.shape[2:], mode='bilinear', align_corners=False)
        p4 = F.interpolate(p4, size=p1.shape[2:], mode='bilinear', align_corners=False)
        
        p4 = self.dropout(p4)
        p3 = self.dropout(p3)
        p2 = self.dropout(p2)
        p1 = self.dropout(p1)

        out = self.merge([p1, p2, p3, p4])

        out = self.merge_conv(out)

        out = self.upsample1(out)
        out = self.upsample2(out)
        
        out = self.c0_merge(torch.cat([out, c0], dim=1))

        out = self.seg_head(out)
        return out



class UperNetDepthHead(nn.Module):
    def __init__(self, dims, decoder_dim, num_classes, merge_policy="cat", channel_reduction=1):
        super(UperNetDepthHead, self).__init__()
        c1, c2, c3, c4 = dims
        c0 = c1 #// 2
        
        self.num_classes = num_classes
        self.merge_policy = merge_policy
        self.dropout = nn.Dropout2d(0.2)

        self.context_block = PPM(c4, decoder_dim)

        self.fpn_in = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, decoder_dim, kernel_size=1),
                nn.BatchNorm2d(decoder_dim),
                ACTIVATION,
            )
            for c in [c1, c2, c3]
        ])
        
        self.fpn_out = nn.ModuleList([
            ConvBlock(decoder_dim, decoder_dim)
            for _ in range(4)
        ])
        # ConvLite
        self.merge = MergeBlock(merge_policy)
        
        if merge_policy == "cat":
            final_in_channels = decoder_dim * 4
        else:
            final_in_channels = decoder_dim
            
        self.merge_conv = nn.Sequential(
            ConvBlock(final_in_channels, decoder_dim)
        )

        self.upsample1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False) 
        self.upsample2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        if merge_policy == "cat":
            c0_in_channels = decoder_dim * 2
        else:
            c0_in_channels = decoder_dim

        self.c0_merge = ConvBlock(c0_in_channels, decoder_dim // (channel_reduction ** 2))

        self.depth_head = nn.Sequential(
            nn.Conv2d(decoder_dim // (channel_reduction ** 2), decoder_dim // (channel_reduction ** 2), kernel_size=3, padding=1),
            ACTIVATION,
            
            self.dropout,
            nn.Conv2d(decoder_dim // (channel_reduction ** 2), 1, kernel_size=1),
        )
        
        self.mask_proj = nn.Sequential(
            nn.Conv2d(decoder_dim // (channel_reduction ** 2) + 3, decoder_dim // (channel_reduction ** 2), kernel_size=1),
            nn.BatchNorm2d(decoder_dim // (channel_reduction ** 2)),
            ACTIVATION
        )
        
        self.mask_conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c+self.num_classes, c, kernel_size=1),
                nn.BatchNorm2d(c),
                ACTIVATION
            ) for c in dims
        ])


    def forward(self, features, mask=None, c0=None):
        c1, c2, c3, c4 = features
    
        if mask is not None:
            processed_features = []
            for idx, feature in enumerate(features):
                
                resized_mask = F.interpolate(mask, size=feature.shape[2:], mode='bilinear', align_corners=False)
                feature = torch.cat([feature, resized_mask], dim=1)
                feature = self.mask_conv_layers[idx](feature)
                processed_features.append(feature)
            

            c1, c2, c3, c4 = processed_features
        else:

            c1, c2, c3, c4 = features

        f4 = self.context_block(c4)
        f3 = self.fpn_in[2](c3)
        f2 = self.fpn_in[1](c2)
        f1 = self.fpn_in[0](c1)
        
        f3 = f3 + F.interpolate(f4, size=f3.shape[2:], mode='bilinear', align_corners=False)
        f2 = f2 + F.interpolate(f3, size=f2.shape[2:], mode='bilinear', align_corners=False)
        f1 = f1 + F.interpolate(f2, size=f1.shape[2:], mode='bilinear', align_corners=False)
        
        p4 = self.fpn_out[3](f4)
        p3 = self.fpn_out[2](f3)
        p2 = self.fpn_out[1](f2)
        p1 = self.fpn_out[0](f1)
        
        
        p2 = F.interpolate(p2, size=p1.shape[2:], mode='bilinear', align_corners=False)
        p3 = F.interpolate(p3, size=p1.shape[2:], mode='bilinear', align_corners=False)
        p4 = F.interpolate(p4, size=p1.shape[2:], mode='bilinear', align_corners=False)
        
        p4 = self.dropout(p4)
        p3 = self.dropout(p3)
        p2 = self.dropout(p2)
        p1 = self.dropout(p1)
        
        out = self.merge([p1, p2, p3, p4])
        
        out = self.merge_conv(out)
                        
        out = self.upsample1(out)
        out = self.upsample2(out)

        out = self.c0_merge(torch.cat([out, c0], dim=1))

        if mask is not None:
            resized_mask = F.interpolate(mask, size=out.shape[2:], mode='bilinear', align_corners=False)
            out = torch.cat([out, resized_mask], dim=1)
            out = self.mask_proj(out)
        
        out = self.depth_head(out)
        
        return out


class ResidualConvUnit_custom(nn.Module):
    """Residual convolution module."""
    def __init__(self, features, activation, bn=True):
        super().__init__()
        self.bn = bn
        self.groups = 1

        self.conv1 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, 
            bias=not self.bn, groups=self.groups
        )
        self.conv2 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, 
            bias=not self.bn, groups=self.groups
        )

        if self.bn:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)

        self.activation = activation
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        out = self.activation(x)
        out = self.conv1(out)
        if self.bn:
            out = self.bn1(out)

        out = self.activation(out)
        out = self.conv2(out)
        if self.bn:
            out = self.bn2(out)

        return self.skip_add.add(out, x)


class FeatureFusionBlock_custom(nn.Module):
    """Feature fusion block."""
    def __init__(
        self, features, activation, deconv=False, bn=False, 
        expand=False, align_corners=True,
    ):
        super().__init__()
        self.deconv = deconv
        self.align_corners = align_corners
        self.groups = 1
        self.expand = expand
        
        out_features = features
        if self.expand:
            out_features = features // 2

        self.out_conv = nn.Conv2d(
            features, out_features, kernel_size=1, stride=1, 
            padding=0, bias=True, groups=1,
        )

        self.resConfUnit1 = ResidualConvUnit_custom(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit_custom(features, activation, bn)
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, *xs):
        output = xs[0]

        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)

        output = self.resConfUnit2(output)
        output = F.interpolate(
            output, scale_factor=2, mode="bilinear", 
            align_corners=self.align_corners
        )
        output = self.out_conv(output)
        return output


class SegDPTDecoder(nn.Module):
    def __init__(self, dims, decoder_dim=256, num_classes=3):
        super().__init__()
        c1, c2, c3, c4 = dims
        self.decoder_dim = decoder_dim
        
        self.layer1_rn = nn.Conv2d(c1, decoder_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer2_rn = nn.Conv2d(c2, decoder_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer3_rn = nn.Conv2d(c3, decoder_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer4_rn = nn.Conv2d(c4, decoder_dim, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.activation = ACTIVATION
        
        self.fusion_block4 = FeatureFusionBlock_custom(decoder_dim, self.activation)
        self.fusion_block3 = FeatureFusionBlock_custom(decoder_dim, self.activation)
        self.fusion_block2 = FeatureFusionBlock_custom(decoder_dim, self.activation)
        self.fusion_block1 = FeatureFusionBlock_custom(decoder_dim, self.activation)

        self.head = nn.Sequential(
            nn.Conv2d(decoder_dim, decoder_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU(True),
            nn.Dropout(0.2, False),
            nn.Conv2d(decoder_dim, num_classes, kernel_size=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
        )
        
    def forward(self, features):
        c1, c2, c3, c4 = features
        
        layer_1 = self.layer1_rn(c1)
        layer_2 = self.layer2_rn(c2)
        layer_3 = self.layer3_rn(c3)
        layer_4 = self.layer4_rn(c4)
        
        x = self.fusion_block4(layer_4)
        x = self.fusion_block3(x, layer_3)
        x = self.fusion_block2(x, layer_2)
        x = self.fusion_block1(x, layer_1)

        x = self.head(x)
        
        return x
        
class DPTDecoder(nn.Module):
    def __init__(self, dims, decoder_dim=256, output_channels=1):
        super().__init__()
        c1, c2, c3, c4 = dims
        self.decoder_dim = decoder_dim
        
        self.layer1_rn = nn.Conv2d(c1, decoder_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer2_rn = nn.Conv2d(c2, decoder_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer3_rn = nn.Conv2d(c3, decoder_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer4_rn = nn.Conv2d(c4, decoder_dim, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.activation = ACTIVATION
        
        self.fusion_block4 = FeatureFusionBlock_custom(decoder_dim, self.activation)
        self.fusion_block3 = FeatureFusionBlock_custom(decoder_dim, self.activation)
        self.fusion_block2 = FeatureFusionBlock_custom(decoder_dim, self.activation)
        self.fusion_block1 = FeatureFusionBlock_custom(decoder_dim, self.activation)
        
        self.head = nn.Sequential(
            nn.Conv2d(decoder_dim, decoder_dim // 2, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(decoder_dim // 2, 32, kernel_size=3, stride=1, padding=1),
            ACTIVATION,
            nn.Conv2d(32, output_channels, kernel_size=1, stride=1, padding=0)
        )
        
        self.aux_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(decoder_dim, decoder_dim // 2, kernel_size=3, padding=1),
                #nn.BatchNorm2d(decoder_dim // 2),
                nn.Conv2d(decoder_dim // 2, 32, kernel_size=3, stride=1, padding=1),
                ACTIVATION,
                nn.Conv2d(32, output_channels, kernel_size=1)
            ) for _ in range(4)
        ])

        self.mask_proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim+3, dim, kernel_size=1),
                nn.BatchNorm2d(dim),
                ACTIVATION
            ) for dim in dims
        ])
        
    def forward(self, features, mask=None):
        
        if mask is not None:
            processed_features = []
          
            for feature, m_p in zip(features, self.mask_proj):

                resized_mask = F.interpolate(mask, size=feature.shape[2:], mode='bilinear', align_corners=False) 
                
                feature = torch.cat([feature, resized_mask], dim=1)
                feature = m_p(feature)
                processed_features.append(feature)
        else:
            processed_features = features
            
        c1, c2, c3, c4 = processed_features
        
        layer_1 = self.layer1_rn(c1)
        layer_2 = self.layer2_rn(c2)
        layer_3 = self.layer3_rn(c3)
        layer_4 = self.layer4_rn(c4)
        
        x = self.fusion_block4(layer_4)
        x = self.fusion_block3(x, layer_3)
        x = self.fusion_block2(x, layer_2)
        x = self.fusion_block1(x, layer_1)
        
        aux_outputs = [
            self.aux_heads[0](layer_1),
            self.aux_heads[1](layer_2),
            self.aux_heads[2](layer_3),
            self.aux_heads[3](layer_4)
        ]
        x = self.head(x)
        
        return x #, aux_outputs


def double_conv(in_channels, out_channels):
    """Double convolution block with batch normalization and activation"""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        ACTIVATION,
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        ACTIVATION
    )
    
class SegUnet(nn.Module):
    def __init__(self, dims, decoder_dim, num_classes):
        super().__init__()
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = dims
        c0_in_channels = c1_in_channels // 2

        self.num_classes = num_classes
        
        self.dropout = nn.Dropout2d(0.2)

        self.up_conv4 = UpsampleLite(c4_in_channels, c4_in_channels)
        self.up_conv3 = UpsampleLite(c3_in_channels, c3_in_channels)
        self.up_conv2 = UpsampleLite(c2_in_channels, c2_in_channels)
        self.up_conv1 = UpsampleLite(c1_in_channels, c1_in_channels)

        self.dconv4 = ConvBlock(c4_in_channels + c3_in_channels, c3_in_channels)
        self.dconv3 = ConvBlock(c3_in_channels + c2_in_channels, c2_in_channels)
        self.dconv2 = ConvBlock(c2_in_channels + c1_in_channels, c1_in_channels)
        self.dconv1 = ConvBlock(c1_in_channels + c0_in_channels, c0_in_channels)
        
        self.final_conv = nn.Sequential(
            nn.Conv2d(c0_in_channels, c0_in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(c0_in_channels),
            ACTIVATION,
            self.dropout,
            nn.Conv2d(c0_in_channels, num_classes, kernel_size=1)
        )

    def forward(self, features, c0):
        c1, c2, c3, c4 = features

        x = self.up_conv4(c4)
        if x.shape[2:] != c3.shape[2:]:
            x = F.interpolate(x, size=c3.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, c3], dim=1)
        x = self.dconv4(x)
        
        x = self.up_conv3(x)
        if x.shape[2:] != c2.shape[2:]:
            x = F.interpolate(x, size=c2.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, c2], dim=1)
        x = self.dconv3(x)
        
        x = self.up_conv2(x)
        if x.shape[2:] != c1.shape[2:]:
            x = F.interpolate(x, size=c1.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, c1], dim=1)
        x = self.dconv2(x)

        x = self.up_conv1(x)
        if x.shape[2:] != c0.shape[2:]:
            x = F.interpolate(x, size=c0.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, c0], dim=1)
        x = self.dconv1(x)

        mask_pred = self.final_conv(x)
        
        return mask_pred

class DepthUnet(nn.Module):
    def __init__(self, dims, decoder_dim):
        super().__init__()
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = dims
        c0_in_channels = c1_in_channels // 2

        self.c0_in_channels = c0_in_channels
        self.num_classes = 1
        
        self.up_conv4 = UpsampleLite(c4_in_channels, c4_in_channels)
        self.up_conv3 = UpsampleLite(c3_in_channels, c3_in_channels)
        self.up_conv2 = UpsampleLite(c2_in_channels, c2_in_channels)
        self.up_conv1 = UpsampleLite(c1_in_channels, c1_in_channels)
        self.up_conv0 = UpsampleLite(c0_in_channels, c0_in_channels)

        self.dconv4 = ConvBlock(c4_in_channels + c3_in_channels, c3_in_channels)
        self.dconv3 = ConvBlock(c3_in_channels + c2_in_channels, c2_in_channels)
        self.dconv2 = ConvBlock(c2_in_channels + c1_in_channels, c1_in_channels)
        self.dconv1 = ConvBlock(c1_in_channels + c0_in_channels, c0_in_channels)
        self.dconv0 = ConvBlock(c0_in_channels, c0_in_channels)

        self.dropout = nn.Dropout2d(0.2)

        self.final_conv = nn.Sequential(
            nn.Conv2d(c0_in_channels, c0_in_channels, kernel_size=3, padding=1),
            ACTIVATION,
            self.dropout,
            nn.Conv2d(c0_in_channels, 1, kernel_size=1)
        )

        self.mask_contexts = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch + 3, in_ch, kernel_size=1),
                nn.BatchNorm2d(in_ch),
                ACTIVATION
            )
            for in_ch in [c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels]
        ])

        self.mask_proj = nn.Sequential(
            nn.Conv2d(c0_in_channels + 3, c0_in_channels, kernel_size=1),
            nn.BatchNorm2d(c0_in_channels),
            ACTIVATION
        )

    def forward(self, features, sensor_inter_mask, c0):
        c1, c2, c3, c4 = features

        if sensor_inter_mask is not None:
            updated_features = []
            for feature, mask_context in zip(features, self.mask_contexts):
                resized_mask = F.interpolate(sensor_inter_mask, size=feature.shape[2:], mode='bilinear', align_corners=False)
                combined = torch.cat([feature, resized_mask], dim=1)
                updated_features.append(mask_context(combined))
            c1, c2, c3, c4 = updated_features

            resized_mask = F.interpolate(sensor_inter_mask, size=c0.shape[2:], mode='bilinear', align_corners=False)
            c0 = torch.cat([c0, resized_mask], dim=1)
            c0 = self.mask_proj(c0)

        x = self.up_conv4(c4)
        if x.shape[2:] != c3.shape[2:]:
            x = F.interpolate(x, size=c3.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, c3], dim=1)
        x = self.dconv4(x)

        x = self.up_conv3(x)
        if x.shape[2:] != c2.shape[2:]:
            x = F.interpolate(x, size=c2.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, c2], dim=1)
        x = self.dconv3(x)

        x = self.up_conv2(x)
        if x.shape[2:] != c1.shape[2:]:
            x = F.interpolate(x, size=c1.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, c1], dim=1)
        x = self.dconv2(x)

        x = self.up_conv1(x)
        if x.shape[2:] != c0.shape[2:]:
            x = F.interpolate(x, size=c0.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, c0], dim=1)
        x = self.dconv1(x)

        depth_pred = self.final_conv(x)

        return depth_pred

    
class SegUNetPlusPlus(nn.Module):
    def __init__(self, dims, num_classes):
        super().__init__()

        c1, c2, c3, c4 = dims

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_1 = ConvBlock(c1 + c2, c1)
        self.conv0_2 = ConvBlock(c1*2 + c2, c1)
        self.conv0_3 = ConvBlock(c1*3 + c2, c1)

        # Level 1 blocks
        self.conv1_1 = ConvBlock(c2 + c3, c2)
        self.conv1_2 = ConvBlock(c2*2 + c3, c2)

        # Level 2 blocks
        self.conv2_1 = ConvBlock(c3 + c4, c3)


        self.refine = nn.Sequential(
            ResBlock(c1, c1, use_bn=False),
            ResBlock(c1, c1, use_bn=False),
        )
        
        self.dropout = nn.Dropout2d(0.2)
        self.final = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, padding=1),
            nn.BatchNorm2d(c1),
            ACTIVATION,

            self.dropout,
            nn.Conv2d(c1, num_classes, kernel_size=1)
        )

    def forward(self, inputs):
        x0_0, x1_0, x2_0, x3_0 = inputs

        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        out = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        out = self.refine(out)

        out = F.interpolate(out, scale_factor=4, mode='bilinear', align_corners=True)
        out = self.final(out)
        return out
    
class DepthUNetPlusPlus(nn.Module):
    def __init__(self, dims):
    #def __init__(self, num_classes, input_channels=3, deep_supervision=False, **kwargs):
        super().__init__()

        c1, c2, c3, c4 = dims

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upconv = nn.Sequential(
            UpsampleBlock(c1, c1),
            UpsampleBlock(c1, c1)
        )

        self.conv0_1 = ConvBlock(c1 + c2, c1)
        self.conv0_2 = ConvBlock(c1*2 + c2, c1)
        self.conv0_3 = ConvBlock(c1*3 + c2, c1)

        # Level 1 blocks
        self.conv1_1 = ConvBlock(c2 + c3, c2)
        self.conv1_2 = ConvBlock(c2*2 + c3, c2)

        # Level 2 blocks
        self.conv2_1 = ConvBlock(c3 + c4, c3)

        self.dropout = nn.Dropout2d(0.2)
        self.final = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, padding=1),
            ACTIVATION,

            self.dropout,
            nn.Conv2d(c1, 1, kernel_size=1)
        )

        self.mask_contexts = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch + 3, in_ch, kernel_size=1),
                nn.BatchNorm2d(in_ch),
                ACTIVATION
            )
            for in_ch in [c1, c2, c3, c4]
        ])
        self.mask_proj = nn.Sequential(
                nn.Conv2d(c1 + 3, c1, kernel_size=1),
                nn.BatchNorm2d(c1),
                ACTIVATION
            )
        self.refine = nn.Sequential(
            ResBlock(c1, c1, use_bn=False),
            ResBlock(c1, c1, use_bn=False),
        )
    def forward(self, inputs, sensor_inter_mask=None):
        if sensor_inter_mask is not None:
            updated_features = []
            for feature, mask_context in zip(inputs, self.mask_contexts):
                resized_mask = F.interpolate(sensor_inter_mask, size=feature.shape[2:], mode='bilinear', align_corners=False)
                combined = torch.cat([feature, resized_mask], dim=1)
                updated_features.append(mask_context(combined))
            x0_0, x1_0, x2_0, x3_0 = updated_features
        else:
            x0_0, x1_0, x2_0, x3_0 = inputs

        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))

        out = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        out = self.refine(out)

        out = self.upconv(out)
        
        if sensor_inter_mask is not None:
            resized_mask = F.interpolate(sensor_inter_mask, size=out.shape[2:], mode='bilinear', align_corners=True)
            out = torch.cat([out, resized_mask], dim=1)
            out = self.mask_proj(out)

        out = self.final(out)
        return out
