import torch
import torch.nn as nn
import torch.nn.functional as F
import settings
from torchvision.ops import DeformConv2d
from unet_transformer import SwinTransformerBlock
from temporal_attention import TemporalAttention

logger = settings.logger

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()

        self.extractor = nn.Sequential(
            nn.Conv2d(14, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        '''
        input:
        x: (b, 9, w, h)
        return:
        (b, 9 + 23, w, h)
        '''
        shortcut = x
        x = self.extractor(x)
        return torch.cat((shortcut, x), dim=1)

class FeatureReweighting(nn.Module):
    def __init__(self):
        super(FeatureReweighting, self).__init__()

        self.reweighter = nn.Sequential(
            nn.Conv2d(18, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, cur_raw_feats, prev_raw_feats, previous_features):
        '''
        input:
        current_features:  (b, 32, c, w, h)
        previous_features: (b, 32, c, w, h)
        '''
        input = torch.cat((cur_raw_feats, prev_raw_feats), dim = 1)
        scores = self.reweighter(input)
        # scale to 0-10
        scores = (scores + 1) * 5. #b, 4, w, h
        # scores = scores.unsqueeze(2)
        return scores * previous_features

class TemporalStabilization(nn.Module):
    """Temporal Spatial Attention (TSA) fusion module.

    Temporal: Calculate the correlation between center frame and
        neighboring frames;

    Args:
        num_feat (int): Channel number of middle features. Default: 64.
        num_frame (int): Number of frames. Default: 5.
        center_frame_idx (int): The index of center frame. Default: 2.
    """

    def __init__(self, input_feat=9, num_feat=32):
        super(TemporalStabilization, self).__init__()
        # temporal stabilization
        self.temporal_attn1 = nn.Conv2d(input_feat, num_feat, 3, 1, 1)
        self.temporal_attn2 = nn.Conv2d(input_feat, num_feat, 3, 1, 1)

    def forward(self, cur_raw_feats, prev_raw_feats, x):
        """
        Args:
            cur_raw_feats (Tensor): current raw features with shape (b, 9, h, w).
            prev_raw_feats (Tensor): previous raw features with shape (b, 9, h, w).
            previous_features (Tensor): Aligned features with shape (b, c, h, w).

        Returns:
            Tensor: Features after TSA with the shape (b, c, h, w).
        """
        b, c, h, w = x.size()
        cur_embedding = self.temporal_attn1(cur_raw_feats)
        prev_embedding = self.temporal_attn2(prev_raw_feats)
        corr = torch.sum(prev_embedding * cur_embedding, 1) # (b, h, w)
        corr = corr.unsqueeze(1)  # (b, 1, h, w)

        corr_prob = torch.sigmoid(corr)  # (b, 1, h, w)
        corr_prob = corr_prob.expand(b, c, h, w)
        aligned_x = x * corr_prob
        
        return aligned_x

class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ConvBlock, self).__init__()
        self.in_channel=in_channel
        self.out_channel=out_channel
        self.block = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        out = self.block(x)
        return out

    def flops(self, H, W): 
        flops = H*W*self.in_channel*self.out_channel*(3*3+1)+H*W*self.out_channel*self.out_channel*3*3
        return flops

class DilationConv(nn.Module):
    def __init__(self, inchannel):
        super(DilationConv, self).__init__()
        self.dilation_1 = nn.Sequential(
            nn.Conv2d(inchannel, 64, 3, 1, 1, 1),
            nn.ReLU(inplace=True)
        )
        self.dilation_2 = nn.Sequential(
            nn.Conv2d(inchannel, 64, 3, 1, 2, 2),
            nn.ReLU(inplace=True)
        )
        self.dilation_5 = nn.Sequential(
            nn.Conv2d(inchannel, 64, 3, 1, 3, 3),
            nn.ReLU()
        )
        self.cat = nn.Sequential(
            nn.Conv2d(64*3, 64, 1, 1),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):

        x_1 = self.dilation_1(x)
        x_2 = self.dilation_2(x)
        x_5 = self.dilation_5(x)
        x = self.cat(torch.cat((x_1, x_2, x_5), dim=1))

        return x

class DeformConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DeformConvBlock, self).__init__()
        self.in_channel=in_channel
        self.out_channel=out_channel
        self.deform = DeformConv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, offsets):
        out = self.relu(self.deform(x, offsets))
        return out

# Upsample Block
class UpsamplePS(nn.Module):
    def __init__(self, in_channel, up_scale):
        super(UpsamplePS, self).__init__()
        self.in_channel = in_channel
        self.out_channel = in_channel * (2 ** up_scale)
        self.conv = nn.Conv2d(self.in_channel, self.out_channel, kernel_size=3, stride=1, padding=1)
        self.ps = nn.PixelShuffle(upscale_factor=up_scale)
                
    def forward(self, x):
        out = self.ps(self.conv(x))
        return out

    def flops(self, H, W):
        flops = 0
        # conv
        flops += H*2*W*2*self.in_channel*self.out_channel*2*2 
        print("Upsample:{%.2f}"%(flops/1e9))
        return flops

class ReconstructionNetwork(nn.Module):
    def __init__(self, in_channel, dilation, deform):
        super(ReconstructionNetwork, self).__init__()
        self.deform = deform
        self.encoder0 = SwinTransformerBlock(in_channel, input_resolution=(settings.train_width, settings.train_width), num_heads=7)
        self.encoder1 = ConvBlock(in_channel, 64)
        self.encoder2 = ConvBlock(64, 32)

        self.encoder3 = ConvBlock(32, 64)
        self.encoder4 = ConvBlock(64, 64)

        self.mid1 = ConvBlock(64, 128)
        self.mid2 = ConvBlock(128, 128)

        self.decoder1 = ConvBlock(128 + 64, 64)
        self.decoder3 = ConvBlock(64 + 32, 64)

        self.ps1 = nn.PixelShuffle(2)
        self.ps2 = nn.PixelShuffle(2)
        self.conv_bf_ps1 = nn.Conv2d(128, 128 * 4, kernel_size=3, stride=1, padding=1)
        self.conv_bf_ps2 = nn.Conv2d(64, 64 * 4, kernel_size=3, stride=1, padding=1)

        if self.deform:
            self.offsets2 = nn.Conv2d(64, 2 * 3 ** 2, kernel_size=3, stride=1, padding=1)
            # self.mask2 = nn.Conv2d(64, 3 ** 2, 3, 1, 1)
            self.decoder2 = DeformConvBlock(64, 64)

            self.offsets4 = nn.Conv2d(64, 2 * 3 ** 2, kernel_size=3, stride=1, padding=1)
            # self.mask4 = nn.Conv2d(64, 3 ** 2, 3, 1, 1)
            self.decoder4 = DeformConvBlock(64, 4)
        else:
            self.decoder2 = ConvBlock(64, 64)
            self.decoder4 = ConvBlock(64, 4)

        self.sigmoid = nn.Sigmoid()

    def forward(self, current_features, acc_x):
        '''
        current_features: (b, c, w, h)
        previous_features: (b, c2, w, h)
        '''

        x = torch.cat((current_features, acc_x), dim=1)

        # encoder
        conv1 = self.encoder0(x)
        conv1 = self.encoder1(conv1)
        conv2 = self.encoder2(conv1)
        pool2 = F.max_pool2d(conv2, 2)

        conv3 = self.encoder3(pool2)
        conv4 = self.encoder4(conv3)
        pool4 = F.max_pool2d(conv4, 2)

        # middle stage
        mid1 = self.mid1(pool4)
        mid2 = self.mid2(mid1)

        # decoder
        up5 = self.ps1(self.conv_bf_ps1(mid2))
        up5 = torch.cat((up5, conv4), dim=1)
        conv5 = self.decoder1(up5)

        if self.deform:
            offsets2 = self.offsets2(conv5)
            # mask2 = F.sigmoid(self.mask2(x))
            # x = F.relu(self.decoder2(x, offsets2, mask=mask2), inplace=True)
            conv6 = self.decoder2(conv5, offsets2)
            up6 = self.ps2(self.conv_bf_ps2(conv6))
            up6 = torch.cat([up6, conv2], dim=1)

            # x = F.interpolate(x, size=skips[-2].size()[2:], mode='bilinear', align_corners=False)
            # x = torch.cat((x, skips[-2]), dim=1)
            conv7 = self.decoder3(up6)

            offsets4 = self.offsets4(conv7)
            # mask4 = F.sigmoid(self.mask4(x))
            # x = F.relu(self.decoder4(x, offsets4, mask=mask4), inplace=True)
            out = self.decoder4(conv7, offsets4)

        else:
            conv6 = self.decoder2(conv5)
            up6 = self.ps2(self.conv_bf_ps2(conv6))
            up6 = torch.cat([up6, conv2], dim=1)

            conv7 = self.decoder3(up6)
            out = self.decoder4(conv7)

        x_t = out[:, :3, ...]
        alpha = self.sigmoid(out[:, 3:, ...])

        ret = alpha * x_t + (1. - alpha) * acc_x
        return ret

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.previous_est = None
        self.mode = 'train'
        self.recursion_step = settings.recursion_step
        self.feature_extractor = FeatureExtractor()
        self.reconstruct = ReconstructionNetwork(49, dilation=settings.dilation, deform=settings.deform)
        # self.temporal_attention = TemporalStabilization()
        self.temporal_attention = TemporalAttention(ch=32, checkpoint=False)

    def warper2d(self, history, flow):
        h, w = history.size()[-2:]
        x_grid = torch.arange(0., w).to(history.device).float() + 0.5
        y_grid = torch.arange(0., h).to(history.device).float() + 0.5
        x_grid = (x_grid / w).view(1, 1, -1, 1).expand(1, h, -1, 1)  # 1, h, w, 1
        y_grid = (y_grid / h).view(1, -1, 1, 1).expand(1, -1, w, 1)  # 1, h, w, 1
        x_grid = x_grid * 2 - 1.
        y_grid = y_grid * 2 - 1.

        grid = torch.cat((x_grid, y_grid), dim=-1)  # b, h, w, 2
        flow = flow.permute(0, 2, 3, 1)  # b, h, w, 2

        grid = grid - flow*2

        warped = F.grid_sample(history, grid, align_corners=True)
        return warped

    def initial_est(self):
        self.previous_est = None

    def change_mode(self, mode):
        self.mode = mode
        if self.mode == 'test':
            self.recursion_step = 1
        else:
            self.recursion_step = settings.recursion_step

    def forward(self, inputs):
        frames = inputs['image']
        mvs = inputs['mv']
        depths = inputs['depth']
        normals = inputs['normal']
        # pbrs = inputs['pbr']
        albedos = inputs['albedo']
        trans = inputs['tran']
        outputs = []

        for step in range(self.recursion_step):
            frame = frames[..., step]
            depth = depths[..., step]
            normal = normals[..., step]
            # pbr = pbrs[..., step]
            albedo = albedos[..., step]
            tran = trans[..., step]
            mv = mvs[..., step]
            x = torch.cat((frame, normal, depth, albedo, tran), dim=1)
            if self.mode == 'train' and step == 0 or self.previous_est == None:
                self.previous_est = x       

            prev_raw_feats = self.warper2d(self.previous_est, mv)
            aligned_img = self.temporal_attention([prev_raw_feats[:, 0:3, ...], x, prev_raw_feats])

            cur_features = self.feature_extractor(x)
            denoised_img = self.reconstruct(cur_features, aligned_img)

            self.previous_est = torch.cat([denoised_img, normal, depth, albedo, tran], dim=1)

            outputs.append(denoised_img)
        outputs = torch.stack(outputs, dim=-1)
        return outputs

