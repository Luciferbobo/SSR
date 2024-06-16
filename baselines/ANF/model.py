import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import settings


class PixelEmbeddings(nn.Module):
    def __init__(self):
        super(PixelEmbeddings, self).__init__()

        self.FC = nn.Sequential(
            nn.Conv2d(12, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
        )

    def forward(self, x):
        '''
        input: image(3), albedo(3), normal(3), pbr(2), depth(1)
        x: (b, 12, w, h), RGB
        return:
        (b, 32, w, h)
        '''
        return self.FC(x)


class UnetParameters(nn.Module):
    def __init__(self, in_channel):
        super(UnetParameters, self).__init__()

        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channel, 64, 3, padding=1), nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(64, 64, 3, padding=1), nn.LeakyReLU(negative_slope=0.1)
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1), nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(64, 64, 3, padding=1), nn.LeakyReLU(negative_slope=0.1)
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1), nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(64, 64, 3, padding=1), nn.LeakyReLU(negative_slope=0.1)
        )
        self.encoder4 = nn.Sequential(
            nn.Conv2d(64, 80, 3, padding=1), nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(80, 80, 3, padding=1), nn.LeakyReLU(negative_slope=0.1)
        )

        self.encoder4 = nn.Sequential(
            nn.Conv2d(64, 80, 3, padding=1), nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(80, 80, 3, padding=1), nn.LeakyReLU(negative_slope=0.1)
        )

        self.decoder4 = nn.Sequential(
            nn.Conv2d(80, 96, 3, padding=1), nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(96, 96, 3, padding=1), nn.LeakyReLU(negative_slope=0.1),
            nn.Upsample(scale_factor=2, mode='nearest'),
        )
        self.decoder3 = nn.Sequential(
            nn.Conv2d(160, 80, 3, padding=1), nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(80, 80, 3, padding=1), nn.LeakyReLU(negative_slope=0.1),
            nn.Upsample(scale_factor=2, mode='nearest'),
        )
        self.decoder2 = nn.Sequential(
            nn.Conv2d(144, 64, 3, padding=1), nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(64, 64, 3, padding=1), nn.LeakyReLU(negative_slope=0.1),
            nn.Upsample(scale_factor=2, mode='nearest'),
        )
        self.decoder1 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1), nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(64, 64, 3, padding=1), nn.LeakyReLU(negative_slope=0.1),
            nn.Upsample(scale_factor=2, mode='nearest'),
        )

        self.output = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1), nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(32, 32, 3, padding=1), nn.LeakyReLU(negative_slope=0.1)
        )

    def forward(self, Exyt, WExyt):
        '''
        current_features: (b, c, w, h)
        previous_features: (b, c2, w, h)
        '''
        x = torch.cat((Exyt, WExyt), dim=1)

        d1 = F.max_pool2d(input=self.encoder1(x), kernel_size=2)
        d2 = F.max_pool2d(input=self.encoder2(d1), kernel_size=2)
        d3 = F.max_pool2d(input=self.encoder3(d2), kernel_size=2)
        d4 = F.max_pool2d(input=self.encoder4(d3), kernel_size=2)

        u4 = self.decoder4(d4)
        u3 = self.decoder3(torch.cat((d3, u4), dim=1))
        u2 = self.decoder2(torch.cat((d2, u3), dim=1))
        u1 = self.decoder1(torch.cat((d1, u2), dim=1))
        o = self.output(u1)

        return o


def acc_nb(rx, ry, px, py, h, w, stride, lxyt, wxyuvt, f, i, a, c):
    for l1 in range(rx, h - rx):
        for l2 in range(ry, w - ry):
            for k1 in range(l1 - rx, l1 + rx + 1, stride):
                for k2 in range(l2 - ry, l2 + ry + 1, stride):
                    # print(l1,l2,k1,k2)
                    if l1 == k1 and l2 == k2:
                        wxyuvt[i][..., l1, l2, k1 - (l1 - rx), k2 - (l2 - ry)] = c[..., l1 - rx, l2 - ry].unsqueeze(1)
                    else:
                        wxyuvt[i][..., l1 - rx, l2 - ry, k1 - (l1 - rx), k2 - (l2 - ry)] = (
                                -a[..., l1 - rx, l2 - ry] * torch.exp(
                            torch.sum((f[..., l1, l2] - f[..., k1, k2]) ** 2, dim=1))).unsqueeze(1)
            if i != 0:
                lxyt[i][..., l1 - px, l2 - py] = torch.sum(
                    lxyt[i - 1][..., l1 - px:l1 + py + 1, l2 - py:l2 + py + 1] * wxyuvt[i][:, :, l1 - px, l2 - py, ...],
                    dim=(2, 3)) / (1e-10 + torch.sum(wxyuvt[i][:, :, l1 - rx, l2 - ry, ...], dim=(2, 3)))


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.scale = settings.scale
        self.acc_exy = None
        self.Lambda = None
        self.acc_L = None
        self.previous_fk = None
        self.previous_o = None
        self.o_xyt = None
        self.mode = 'train'
        self.recursion_step = settings.recursion_step
        self.PE = PixelEmbeddings()
        self.Unet = UnetParameters(64)
        self.sigmoid = nn.Sigmoid()
        self.out_a = None
        self.out_b = None
        self.t0 = None
        self.t1 = None
        self.t2 = None
        self.t3 = None
        self.t4 = None

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

        grid = grid - flow * 2

        warped = F.grid_sample(history, grid, align_corners=True)
        return warped

    def pixel_shuffle_inv(self, tensor, scale_factor):
        """
        Implementation of inverted pixel shuffle using numpy

        Parameters:
        -----------
        tensor: input tensor, shape is [N, C, H, W]
        scale_factor: scale factor to down-sample tensor

        Returns:
        --------
        tensor: tensor after pixel shuffle, shape is [N, (s*s)*C, H/s, W/s],
            where s refers to scale factor
        """
        num, ch, height, width = tensor.shape
        if height % scale_factor != 0 or width % scale_factor != 0:
            raise ValueError('height and width of tensor must be divisible by '
                             'scale_factor.')

        new_ch = ch * (scale_factor * scale_factor)
        new_height = height // scale_factor
        new_width = width // scale_factor

        tensor = tensor.reshape(
            [num, ch, new_height, scale_factor, new_width, scale_factor])
        # new axis: [num, ch, scale_factor, scale_factor, new_height, new_width]
        tensor = tensor.transpose([0, 1, 3, 5, 2, 4])
        tensor = tensor.reshape([num, new_ch, new_height, new_width])
        return tensor

    def initial_est(self):
        self.acc_exy = None
        self.Lambda = None
        self.acc_L = None
        self.previous_fk = None
        self.o_xyt = None

    def change_mode(self, mode):
        self.mode = mode
        if self.mode == 'test':
            self.recursion_step = 1
        else:
            self.recursion_step = settings.recursion_step

    def forward(self, inputs):
        images = inputs['image']
        albedos = inputs['albedo']
        normals = inputs['normal']
        pbrs = inputs['pbr']
        depths = inputs['depth']
        mvs = inputs['mv']

        outputs = []

        for step in range(self.recursion_step):
            image = images[:, :, :, :, step]
            albedo = albedos[:, :, :, :, step]
            normal = normals[:, :, :, :, step]
            pbr = pbrs[:, :, :, :, step]
            depth = depths[:, :, :, :, step]
            mv = mvs[:, :, :, :, step]

            # features, 12 channels
            rxyt = torch.cat((image, albedo, normal, pbr, depth), dim=1)

            # pixel embeddings, eq. 1
            exyt = self.PE(rxyt)

            first_frame = self.mode == 'train' and step == 0 or self.acc_exy == None
            if first_frame:
                # eq. 3
                self.acc_exy = exyt
                # eq. 5
                self.acc_L = image

                warped_exyt = exyt
            else:
                # warp
                warped_exyt = self.warper2d(self.acc_exy, mv)
                warped_L = self.warper2d(self.acc_L, mv)

            # eq. 2
            params = self.Unet(exyt, warped_exyt)

            # save previous features for temporal filtering
            # self.previous_fk = params[:, 20:28, :, :]

            if not first_frame:
                self.Lambda = self.sigmoid(params[:, 31:, :, :])

                # eq. 3
                self.acc_exy = (1.0 - self.Lambda) * exyt + self.Lambda * warped_exyt
                # eq. 5
                self.acc_L = (1.0 - self.Lambda) * image + self.Lambda * warped_L

            # Temporally-stable kernel-based denoising
            rgb = self.filtering(params, self.acc_L, first_frame, mv)
            outputs.append(rgb)

        outputs = torch.stack(outputs, dim=-1)

        return outputs, self.out_a, self.out_b

    def filtering(self, kernels_param, accL, first_frame, mv):
        kernel_height = 7
        kernel_width = 7

        kernel_l = kernel_height * kernel_width

        b = torch.square(kernels_param[:, 30, :, :])
        b = b.unsqueeze(-1).permute(0, 3, 1, 2)
        #print('b', b.shape)
        ta = 0
        self.out_b = b

        K = 3
        (bs, _, h, w) = kernels_param.shape
        C = 3
        wxyuvt = [torch.zeros([bs, 1, h, w, kernel_height, kernel_width]).cuda() for i in range(K)]
        w_xyuvt = torch.zeros([bs, 1, h, w, kernel_height, kernel_width]).cuda()
        warp_fuvt = 0

        lxyt = [torch.zeros([bs, 3, h, w]).cuda() for i in range(K)]
        lxyt[0] = accL
        fxyt = [0 for i in range(K)]
        fuvt = [1 for i in range(K)]
        # print('wxyuvt',wxyuvt[0].shape)
        # print('lxyt',lxyt[0].shape)

        for i in range(K):
            # print(i)
            # print('K=',i)
            stride = (2 ** i)

            px = kernel_height // 2
            py = kernel_width // 2

            rx = ((kernel_height - 1) * stride + 1) // 2
            ry = ((kernel_width - 1) * stride + 1) // 2

            f_ori = kernels_param[:, (i * 10):(i * 10 + 8), :, :]
            # f = torch.nn.functional.pad(f_ori, pad=[rx, rx, ry, ry], mode='reflect')
            # print('f_ori',f_ori.shape)
            # print('f',f.shape)

            # lxyt[i] = torch.nn.functional.pad(lxyt[i], pad=[rx, rx, ry, ry], mode='reflect')

            a = torch.square(kernels_param[:, (i * 10 + 8), :, :])
            a = a.unsqueeze(-1).permute(0, 3, 1, 2)

            ta += a
            c = self.sigmoid(kernels_param[:, (i * 10 + 9), :, :])
            # print('a',a.shape)
            # print('c',c.shape)

            # fxyt expand
            fxyt[i] = (f_ori.unsqueeze(dim=4)).repeat(1, 1, 1, 1, kernel_l)
            # fxyt[i] = fxyt[i].reshape([bs, -1, kernel_l, h, w])
            fxyt[i] = fxyt[i].permute(0, 1, 4, 2, 3)
            # print('fxyt',fxyt[i].shape)

            # fuvt expand
            unfold = torch.nn.Unfold([kernel_height, kernel_width], stride=1, dilation=stride, padding=rx)
            fuvt[i] = unfold(f_ori)
            # fuvt[i] = fuvt[i].view(bs, -1, kernel_height, kernel_width, h, w)
            fuvt[i] = fuvt[i].view(bs, 8, kernel_l, h, w)
            # print('fuvt',fuvt[i].shape)
            # print('fuvt',fuvt[i].shape)

            # eq(4)
            # print(fxyt)
            wxyuvt[i] = torch.exp(-a * torch.sum((fxyt[i] - fuvt[i]) ** 2, dim=1))

            wxyuvt[i] = wxyuvt[i].unsqueeze(-1).permute(0, 4, 2, 3, 1)
            # print('wxyuvt',wxyuvt[i].shape)
            # wxyuvt[i] = wxyuvt[i].reshape(bs, -1, h, w, kernel_l)
            # self.t0 = f_ori
            # self.t1 = fxyt[i]
            # self.t2 = fuvt[i]
            # self.t3 = wxyuvt[i]
            # print(wxyuvt[i])
            # print('wxyuvt',wxyuvt[i].shape)
            # print('c',c.shape)

            c = c.view(-1)
            index = (
                torch.LongTensor([(0 if i < h * w else 1) for i in range(h * w * bs)]),
                torch.LongTensor([0 for i in range(h * w * bs)]),
                torch.LongTensor([i % h for i in range(h * w * bs)]),
                torch.LongTensor([i % w for i in range(h * w * bs)]),
                torch.LongTensor([(kernel_l // 2) for i in range(h * w * bs)])
            )
            wxyuvt[i] = wxyuvt[i].index_put(index, c)

            # print('w',wxyuvt[i].shape)
            # for h1 in range(h):
            #    for w1 in range(w):
            #        wxyuvt[i][:,h1,w1,6,6]=c[:,h1,w1]
            # eq(6)
            #if True:
            if first_frame:
                if i == 0:
                    lxyt[i] = unfold(accL)
                else:
                    lxyt[i] = unfold(lxyt[i - 1])
                # lxyt[i] = lxyt[i].view(bs, -1, kernel_height, kernel_width, h, w)
                lxyt[i] = lxyt[i].view(bs, C, kernel_l, h, w)
                # print(lxyt[i].shape)
                lxyt[i] = lxyt[i].permute(0, 1, 3, 4, 2)
                lxyt[i] = torch.sum(lxyt[i] * wxyuvt[i], dim=4) / (1e-10 + torch.sum(wxyuvt[i], dim=4))
                # print(lxyt[i])
                # print(lxyt[i].shape)
            else:
                if i != K - 1:
                    if i == 0:
                        lxyt[i] = unfold(accL)
                    else:
                        lxyt[i] = unfold(lxyt[i - 1])
                    lxyt[i] = lxyt[i].view(bs, C, kernel_l, h, w)
                    # print(lxyt[i].shape)
                    lxyt[i] = lxyt[i].permute(0, 1, 3, 4, 2)
                    lxyt[i] = torch.sum(lxyt[i] * wxyuvt[i], dim=4) / (1e-10 + torch.sum(wxyuvt[i], dim=4))
                    # print(i, lxyt[i].shape)
                else:
                    lxyt[i] = unfold(lxyt[i - 1])
                    lxyt[i] = lxyt[i].view(bs, C, kernel_l, h, w)
                    # print(lxyt[i].shape)
                    lxyt[i] = lxyt[i].permute(0, 1, 3, 4, 2)
                    # print('self.previous_fk',self.previous_fk.shape)
                    # print('mv',mv.shape)
                    warp_fuvt = unfold(self.warper2d(self.previous_fk, mv))
                    # warp_fuvt = warp_fuvt.view(bs, -1, kernel_height, kernel_width, h, w)
                    warp_fuvt = warp_fuvt.view(bs, 8, kernel_l, h, w)
                    # print(warp_fuvt.shape)
                    # print(warp_fuvt.shape)
                    # print(fxyt[i].shape)
                    # print(torch.sum((fxyt[i] - warp_fuvt) ** 2, dim=1).shape)
                    w_xyuvt = torch.exp(-b * torch.sum((fxyt[i] - warp_fuvt) ** 2, dim=1))
                    # w_xyuvt = w_xyuvt.reshape(bs, -1, h, w, kernel_l)
                    # print(w_xyuvt.shape)
                    w_xyuvt = w_xyuvt.unsqueeze(-1).permute(0, 4, 2, 3, 1)
                    # print('w_xyuvt',w_xyuvt.shape)
                    # print('lxyt[i]',lxyt[i].shape)

                    previous_o = unfold(self.previous_o)
                    previous_o = previous_o.view(bs, C, kernel_l, h, w)
                    # print(previous_o.shape)
                    previous_o = previous_o.permute(0, 1, 3, 4, 2)
                    lxyt[i] = (torch.sum(lxyt[i] * wxyuvt[i], dim=4) + torch.sum(previous_o * w_xyuvt,
                                                                                 dim=4)) / (
                                      1e-10 + torch.sum(wxyuvt[i], dim=4) + torch.sum(w_xyuvt, dim=4))

        self.previous_fk = f_ori
        self.previous_o = lxyt[K - 1]
        self.out_a = ta
        return lxyt[K - 1]


if __name__ == '__main__':
    l = 128
    batch = 2
    clip = 5
    data = {}
    data['image'] = torch.rand(batch, 3, l, l, clip).cuda()
    data['albedo'] = torch.rand(batch, 3, l, l, clip).cuda()
    data['normal'] = torch.rand(batch, 3, l, l, clip).cuda()
    data['pbr'] = torch.rand(batch, 2, l, l, clip).cuda()
    data['depth'] = torch.rand(batch, 1, l, l, clip).cuda()
    data['mv'] = torch.rand(batch, 2, l, l, clip).cuda()

    network = Network().cuda()
    output, a, b = network(data)
    # print(output.shape)


