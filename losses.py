from torch.nn import functional as func
import numpy as np 
import torch
import settings
from special_losses import EdgeLoss, RelativeEdgeLoss

l1_norm = torch.nn.L1Loss()
edge_loss = EdgeLoss().cuda()
rel_edge_loss = RelativeEdgeLoss().cuda()

def LoG(img):
	weight = [
		[0, 0, 1, 0, 0],
		[0, 1, 2, 1, 0],
		[1, 2, -16, 2, 1],
		[0, 1, 2, 1, 0],
		[0, 0, 1, 0, 0]
	]
	weight = np.array(weight)

	weight_np = np.zeros((1, 1, 5, 5))
	weight_np[0, 0, :, :] = weight
	weight_np = np.repeat(weight_np, img.shape[1], axis=1)
	weight_np = np.repeat(weight_np, img.shape[0], axis=0)
	weight = torch.from_numpy(weight_np).type(torch.FloatTensor).to('cuda:0')
	return func.conv2d(img, weight, padding=1)


def HFEN_L2Norm(output, target):
	return torch.sum(torch.pow(LoG(output) - LoG(target), 2)) / torch.sum(torch.pow(LoG(target), 2))

def HFEN_L1(output, target):
	return l1_norm(LoG(output), LoG(target))

def HFEN_L1Norm(output, target):
	return l1_norm(LoG(output), LoG(target)) / LoG(target).norm()

def get_temporal_data(output, target):
	final_output = output.clone()
	final_target = target.clone()
	final_output.fill_(0)
	final_target.fill_(0)

	for i in range(1, settings.recursion_step):
		final_output[:, :, :, :, i] = output[:, :, :, :, i] - output[:, :, :, :, i-1]
		final_target[:, :, :, :, i] = target[:, :, :, :, i] - target[:, :, :, :, i-1]

	return final_output, final_target

def warper2d(history, flow):
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

    warped = torch.nn.functional.grid_sample(history, grid, align_corners=True)
    return warped

def get_temporal_data_warp(output, target, mv):
	final_output = output.clone()
	final_target = target.clone()
	final_output.fill_(0)
	final_target.fill_(0)

	for i in range(1, settings.recursion_step):
		final_output[..., i] = output[..., i] - warper2d(output[..., i-1], mv[..., i])
		final_target[..., i] = target[..., i] - warper2d(target[..., i-1], mv[..., i])

	return final_output, final_target

def loss_func(output, temporal_output, target, temporal_target):
	ls = l1_norm(output, target)
	lt = l1_norm(temporal_output, temporal_target)
	lp = edge_loss(output, target)

	return 0.8 * ls + 0.1 * lt + 0.1 * lp, ls, lt, lp

# RAE loss
def rae_loss(outputs, data):
    weight = settings.loss_weight
    loss_final = ls_final = lt_final = lp_final = 0
    gts = data['gt']
    mvs = data['mv']

    t_outputs, t_gts = get_temporal_data_warp(outputs, gts, mvs)

    for j in range(settings.recursion_step):
        output = outputs[..., j]
        gt = gts[..., j]
        t_output = t_outputs[..., j]
        t_gt = t_gts[..., j]
        l, ls, lt, lp = loss_func(output, t_output, gt, t_gt)
        loss_final += l * weight[j]
        ls_final += ls
        lt_final += lt
        lp_final += lp

    return loss_final, ls_final, lt_final, lp_final

# loss of Neural Temporal Adaptive Sampling and Denoising
def ntasd_loss(outputs, data):
    gts = data['gt']    

    last_frame = settings.recursion_step - 1

    deltax = outputs[..., last_frame] - outputs[..., last_frame-1]
    deltay = gts[..., last_frame] - gts[..., last_frame-1]
    temporal_loss = l1_norm(deltax, deltay)

    
    output = outputs[..., last_frame]
    gt = gts[..., last_frame]
    spatial_loss = l1_norm(output, gt)
    
    loss_final = spatial_loss + temporal_loss

    return loss_final, spatial_loss, temporal_loss

def Relative_L1(im, ref):
    loss = ( torch.abs(im-ref) / (0.01 + torch.abs(ref.detach())) ).mean()
    return loss

# loss of Deep Adaptive Sampling for Low Sample Count Rendering
def dasr_loss(outputs, data):
    gts = data['gt']    

    last_frame = settings.recursion_step - 1

    deltax = outputs[..., last_frame] - outputs[..., last_frame-1]
    deltay = gts[..., last_frame] - gts[..., last_frame-1]
    temporal_loss = l1_norm(deltax, deltay)
    
    output = outputs[..., last_frame]
    gt = gts[..., last_frame]
    spatial_loss = Relative_L1(output, gt)
    gradient_loss = rel_edge_loss(output, gt)
    
    loss_final = 0.8 * spatial_loss + 0.2 * temporal_loss + gradient_loss

    return loss_final, spatial_loss, temporal_loss, gradient_loss


# Sample-based Monte Carlo Denoising using a Kernel-Splatting Network
class RelativeMSE(torch.nn.Module):
    """Relative Mean-Squared Error.

    :math:`0.5 * \\frac{(x - y)^2}{y^2 + \epsilon}`

    Args:
        eps(float): small number to avoid division by 0.
    """
    def __init__(self, eps=1e-2):
        super(RelativeMSE, self).__init__()
        self.eps = eps

    def forward(self, im, ref):
        """Evaluate the metric.

        Args:
            im(th.Tensor): image.
            ref(th.Tensor): reference.
        """
        mse = torch.pow(im-ref, 2)
        loss = mse/(torch.pow(ref, 2) + self.eps)
        loss = 0.5*torch.mean(loss)
        return loss


class SMAPE(torch.nn.Module):
    """Symmetric Mean Absolute error.

    :math:`\\frac{|x - y|} {|x| + |y| + \epsilon}`

    Args:
        eps(float): small number to avoid division by 0.
    """

    def __init__(self, eps=1e-2):
        super(SMAPE, self).__init__()
        self.eps = eps

    def forward(self, im, ref):
        # NOTE: the denominator is used to scale the loss, but does not
        # contribute gradients, hence the '.detach()' call.
        loss = (torch.abs(im-ref) / (
            self.eps + torch.abs(im.detach()) + torch.abs(ref.detach()))).mean()

        return loss


class TonemappedMSE(torch.nn.Module):
    """Mean-squared error on tonemaped images.

    Args:
        eps(float): small number to avoid division by 0.
    """

    def __init__(self, eps=1e-2):
        super(TonemappedMSE, self).__init__()
        self.eps = eps  # avoid division by zero

    def forward(self, im, ref):
        im = _tonemap(im)
        ref = _tonemap(ref)
        loss = torch.pow(im-ref, 2)
        loss = 0.5*torch.mean(loss)
        return loss


class TonemappedRelativeMSE(torch.nn.Module):
    """Relative mean-squared error on tonemaped images.

    Args:
        eps(float): small number to avoid division by 0.
    """
    def __init__(self, eps=1e-2):
        super(TonemappedRelativeMSE, self).__init__()
        self.eps = eps  # avoid division by zero

    def forward(self, im, ref):
        im = _tonemap(im)
        ref = _tonemap(ref)
        mse = torch.pow(im-ref, 2)
        loss = mse/(torch.pow(ref, 2) + self.eps)
        loss = 0.5*torch.mean(loss)
        return loss


def _tonemap(im):
    """Helper Reinhards tonemapper.

    Args:
        im(th.Tensor): image to tonemap.

    Returns:
        (th.Tensor) tonemaped image.
    """
    im = torch.clamp(im, min=0)
    return im / (1+im)
# Sample-based Monte Carlo Denoising using a Kernel-Splatting Network