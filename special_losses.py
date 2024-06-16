import torch.nn as nn
import torch
import numpy as np
import scipy.ndimage.filters as filters

from torch.nn.parameter import Parameter


def get_dirac(size):
    dirac = np.zeros(((size, size)))

    dirac[size//2, size//2] = 1

    return dirac


def get_gaussian2d(size, sigma):
    ret = filters.gaussian_filter(get_dirac(size), sigma)
    return ret


def gaussian_glaplace(size, sigma):
    return filters.gaussian_laplace(get_dirac(size), sigma)


class ConvFilter(nn.Module):

    def __init__(self, num_outputs, size, weights):
        super().__init__()

        self.num_outputs = num_outputs


        weights = np.expand_dims(weights, axis=0)
        weights= np.stack([weights]*self.num_outputs)

        self.weights =torch.FloatTensor(weights)

        self.padding = (size - 1) // 2


        self.weights = Parameter(self.weights, requires_grad=False)

    def forward(self, x):
        x = nn.functional.conv2d(x, (self.weights), groups=self.num_outputs, padding=self.padding )

        return x


class GaussianFilter(ConvFilter):
    def __init__(self, num_outputs, size, sigma):
        weights = get_gaussian2d(size, sigma)
        super(GaussianFilter, self).__init__(num_outputs, size, weights)


class GLFilter(ConvFilter):
    def __init__(self, num_outputs, size, sigma):
        weights = gaussian_glaplace(size, sigma)
        super().__init__(num_outputs, size, weights)






def get_data(results):
    results = results.data.cpu().numpy()

    results = results[0, :, :, :]

    if results.shape[0] == 1:
        results = results[0, :, :]
    else:
        results = results.transpose([1, 2, 0])

    return results.clip(0, 1)

class EdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = torch.nn.L1Loss()
        num_input = 3

        self.prefilter = GaussianFilter(3, 11, 1.5)
        self.LoG = GLFilter(3, 11 , 1.5)

    def forward(self, input, target):
        input = self.prefilter(input)
        target = self.prefilter(target)

        input = self.LoG(input)
        target = self.LoG(target)

        return self.criterion(input, target)



class RelativeEdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = torch.nn.L1Loss()
        num_input = 3

        self.prefilter = GaussianFilter(3, 11, 1.5)
        self.LoG = GLFilter(3, 11 , 1.5)

    def forward(self, input, target):
        base = target +.01

        input = self.prefilter(input)
        target = self.prefilter(target)

        input = self.LoG(input)
        target = self.LoG(target)

        return self.criterion(input/base, target/base)



class RelativeL1(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = torch.nn.L1Loss()

    def forward(self, input, target):
        base = target +.01

        return self.criterion(input/base, target/base)


class LossCombo(nn.Module):
    def __init__(self, monitor_writer, *losses):
        super().__init__()
        self.monitor_writer = monitor_writer
        pass

        self.losses = []
        self.losses_names = []
        self.factors = []

        for name, loss, factor in losses:
            self.losses.append(loss)
            self.losses_names.append(name)
            self.factors.append(factor)

            self.add_module(name, loss)

    def multi_gpu(self):
        pass
        #self.losses = [nn.DataParallel(x) for x in self.losses]

    def forward(self, input, target, additional_losses):
        loss_results = []
        for idx, loss in enumerate(self.losses):
            loss_results.append(loss(input, target))

        for name, loss_result, factor in zip(self.losses_names, loss_results, self.factors):
            #print(loss_result)
            self.monitor_writer.add_scalar(name, loss_result*factor)

        for name, loss_result, factor in additional_losses:
            loss_result = loss_result.mean()
            #print(loss_result)
            self.monitor_writer.add_scalar(name, loss_result*factor)


        total_loss = sum([factor*loss_result for factor, loss_result in zip(self.factors, loss_results)]) + sum([factor*loss_result.mean() for name, loss_result, factor in additional_losses])
        self.monitor_writer.add_scalar("total_loss", total_loss)

        return total_loss


class MonitorWriter:
    def __init__(self, writer, tensorboard_graph_every, tensorboard_every, denoiser):
        self.tensorboard_graph_every = tensorboard_graph_every
        self.tensorboard_every = tensorboard_every
        self.writer_count = 0

        self.writer = writer

        self.draw_prefix = ""

        self.train = True

        self.denoiser = denoiser

    def next_step(self):
        self.writer_count += 1

    def set_prefix(self, prefix):
        self.draw_prefix = prefix

    def add_scalar(self, name, loss):
        if self.train and self.writer_count % self.tensorboard_graph_every == 0:
            name = self.draw_prefix+ name
            value = loss.data.cpu().numpy()
            self.writer.add_scalar(name, value, self.writer_count)
            value = float(value)


    def add_image(self, name, img):

        if self.train and self.writer_count % self.tensorboard_every == 0:
            self.writer.add_image(self.draw_prefix + name, img, self.writer_count)


