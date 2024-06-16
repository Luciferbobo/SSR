import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
import os


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()

        self.extractor = nn.Sequential(
            nn.Conv2d(4, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 8, 3, 1, 1),
            nn.ReLU(),
        )

    def forward(self, x):
        '''
        input:
        x: (b, 4, w, h), RGB + depth
        return:
        (b, 8 + 4, w, h)
        '''
        shortcut = x
        x = self.extractor(x)
        return torch.cat((shortcut, x), dim=1)


class FeatureReweighting(nn.Module):
    def __init__(self):
        super(FeatureReweighting, self).__init__()

        self.reweighter0 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 12, 3, 1, 1),
            nn.Tanh())
        self.reweighter1 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 12, 3, 1, 1),
            nn.Tanh())
        self.reweighter2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 12, 3, 1, 1),
            nn.Tanh())
        self.reweighter3 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 12, 3, 1, 1),
            nn.Tanh())

    def forward(self, current_features, previous_features):
        '''
        input:
        current_features: (b, 4, w, h)
        previous_features: (b, 4, c, w, h)
        '''
        scores = [0 for i in range(4)]
        reweighter_list = [self.reweighter0, self.reweighter1, self.reweighter2, self.reweighter3]
        for i in range(4):
            scores[i] = reweighter_list[i](torch.cat((previous_features[:, i, ...], current_features), dim=1))
            scores[i] = (scores[i] + 1) * 5.
            previous_features[:, i, ...] *= scores[i]

        bs, num, c, w, h = previous_features.size()
        previous_features = previous_features.view(bs, -1, w, h)

        return previous_features


class ReconstructionNetwork(nn.Module):
    def __init__(self, in_channel):
        super(ReconstructionNetwork, self).__init__()

        self.encoder1 = nn.Conv2d(in_channel, 64, 3, 1, 1)
        self.encoder2 = nn.Conv2d(64, 32, 3, 1, 1)

        self.encoder3 = nn.Conv2d(32, 64, 3, 1, 1)
        self.encoder4 = nn.Conv2d(64, 64, 3, 1, 1)

        self.mid1 = nn.Conv2d(64, 128, 3, 1, 1)
        self.mid2 = nn.Conv2d(128, 128, 3, 1, 1)

        self.decoder1 = nn.Conv2d(128 + 64, 64, 3, 1, 1)
        self.decoder2 = nn.Conv2d(64, 64, 3, 1, 1)

        self.decoder3 = nn.Conv2d(64 + 32, 32, 3, 1, 1)
        self.decoder4 = nn.Conv2d(32, 3, 3, 1, 1)

    def forward(self, current_features, previous_features):
        '''
        current_features: (b, c, w, h)
        previous_features: (b, c2, w, h)
        '''
        x = torch.cat((current_features, previous_features), dim=1)

        skips = []
        # encoder
        x = F.relu(self.encoder1(x), inplace=True)
        x = F.relu(self.encoder2(x), inplace=True)
        skips.append(x)
        x = F.max_pool2d(x, 2)

        x = F.relu(self.encoder3(x), inplace=True)
        x = F.relu(self.encoder4(x), inplace=True)
        skips.append(x)
        x = F.max_pool2d(x, 2)

        # middle stage
        x = F.relu(self.mid1(x), inplace=True)
        x = F.relu(self.mid2(x), inplace=True)

        # decoder
        x = F.interpolate(x, size=skips[-1].size()[2:], mode='bilinear')
        x = torch.cat((x, skips[-1]), dim=1)
        x = F.relu(self.decoder1(x), inplace=True)
        x = F.relu(self.decoder2(x), inplace=True)

        x = F.interpolate(x, size=skips[-2].size()[2:], mode='bilinear')
        x = torch.cat((x, skips[-2]), dim=1)
        x = F.relu(self.decoder3(x), inplace=True)
        x = F.relu(self.decoder4(x), inplace=True)

        return x


class Network(nn.Module):
    def __init__(self, scale=1):
        super(Network, self).__init__()
        self.scale = scale
        self.affine1 = torch.Tensor([[.299, .587, .114], [-.1687, -.3313, .5], [.5, -.4187, -.0813]]).float().cuda()
        self.affine2 = torch.Tensor([[1, 0, 1.402], [1, -0.34414, -.71414], [1, 1.772, 0]]).float().cuda()
        self.current_feature_extractor = FeatureExtractor()
        self.previous_feature_extractor = FeatureExtractor()

        self.feature_reweighter = FeatureReweighting()
        self.reconstruction_network = ReconstructionNetwork(66)

        self.zero_upsampler = None

    def rgb2ycbcr(self, x):
        ycbcr = torch.matmul(x.permute(0, 2, 3, 1).contiguous(), self.affine1.T)  # b w h 3
        return ycbcr.permute(0, 3, 1, 2).contiguous()

    def ycbcr2rgb(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()
        x = torch.matmul(x, self.affine2.T)
        x = torch.clamp(x, 0., 1.).permute(0, 3, 1, 2).contiguous()
        return x

    def tmp_reprojection(self, mv, history, current_frame=None):
        '''
        wrap history to current frame
        mv: bn, 2, w, h
        history: bn, 3, w, h
        return: bn, 3, w, h
        '''
        h, w = history.size()[-2:]
        x_grid = torch.arange(0., w).to(history.device).float() + 0.5
        y_grid = torch.arange(0., h).to(history.device).float() + 0.5
        x_grid = (x_grid / w).view(1, 1, -1, 1).expand(1, h, -1, 1)  # 1, h, w, 1
        y_grid = (y_grid / h).view(1, -1, 1, 1).expand(1, -1, w, 1)  # 1, h, w, 1
        x_grid = x_grid * 2 - 1.
        y_grid = y_grid * 2 - 1.
        grid = torch.cat((x_grid, y_grid), dim=-1)  # bn, w, h, 2
        mv = mv.permute(0, 2, 3, 1)  # bn, w, h, 2

        grid = grid - mv * 2

        if current_frame is None:
            warped = F.grid_sample(history, grid, align_corners=True)
        else:
            mask = F.grid_sample(history, grid, align_corners=True)
            warped = F.grid_sample(history, grid, align_corners=True, padding_mode='border')
            mask = torch.sum(mask, dim=1, keepdim=True)
            warped = torch.where(mask > 1e-2, warped, current_frame)
        return warped

    def forward(self, data):
        depth = data['depth']  # b 1 w h n
        mv = data['mv']  # b 2 w h n
        frames = data['image']  # b 3 w h n
        normals = data['normal']
        albedos = data['albedo']
        bs, _, w, h, n = frames.size()

        x = torch.cat((frames, depth), dim=1)  # b 4 w h n

        current_frame = x[:, :, :, :, -1]
        raw_current_frame = current_frame

        previous_frame = x[:, :, :, :, :-1]

        # process current frame
        current_features = current_frame
        current_features = self.current_feature_extractor(current_features)

        # print(current_features.shape)
        # print(normals[...,-1].shape)
        # print(albedos[...,-1].shape)

        current_features = torch.cat((current_features, normals[..., -1], albedos[..., -1]), dim=1)

        # process previous frame
        current_reweight = raw_current_frame
        current_frame_mv = torch.flip(torch.cumsum(torch.flip(mv, dims=[-1]), dim=-1), dims=[-1])  # b 2 w h n
        previous_features = []
        for i in range(previous_frame.size(-1)):
            x = self.previous_feature_extractor(previous_frame[:, :, :, :, i])
            cmv = F.interpolate(current_frame_mv[:, :, :, :, i + 1], size=x.size()[-2:], mode='bilinear',
                                align_corners=True)
            x = self.tmp_reprojection(cmv, x)

            previous_features.append(x.unsqueeze(1))

        previous_features = torch.cat(previous_features, dim=1)

        # feature reweighting
        reweighted_features = self.feature_reweighter(current_reweight, previous_features)
        x = self.reconstruction_network(current_features, reweighted_features)
        # return self.ycbcr2rgb(x)
        return x


if __name__ == '__main__':
    data = {}
    data['depth'] = torch.rand(2, 1, 100, 100, 5)
    data['mv'] = torch.rand(2, 2, 100, 100, 5)
    data['image'] = torch.rand(2, 3, 100, 100, 5)
    data['normal'] = torch.rand(2, 3, 100, 100, 5)
    data['albedo'] = torch.rand(2, 3, 100, 100, 5)
    network = Network()
    output = network(data)

















