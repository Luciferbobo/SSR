import os
import cv2
import math
import numpy as np


def tonemap(matrix, gamma=2.2):
  return np.clip(matrix ** (1.0/gamma), 0, 1)


# http://filmicworlds.com/blog/filmic-tonemapping-operators/
def UnchartedCurve(color):
    A = 0.22
    B = 0.3
    C = 0.1
    D = 0.2
    E = 0.01
    F = 0.3

    color = ( ( color * ( A * color + C * B ) + D * E ) / ( color * ( A * color + B ) + D * F ) ) - ( E / F )
    color = np.clip(color, 0.0, np.max(color))
    
    return color

def HdrToLinear_Uncharted(color):
    color = UnchartedCurve(color)
    y = UnchartedCurve(11.2)
    color = color / y

    color = np.clip(color, 0.0, np.max(color))

    return color

def FilmicTonemap(color):
    color = HdrToLinear_Uncharted(color)
    color = pow(color, 1/2.2)
    color = np.clip(color, 0, 1)
    return color


def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        # return float('inf')
        return 0.0
    return 20 * math.log10(255.0 / math.sqrt(mse))


def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def rmse(img1, img2):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    num = (img1 - img2)**2
    denom = img2**2 + 1.0e-2
    relMse = np.divide(num, denom)
    relMseMean = 0.5*np.mean(relMse)
    return relMseMean


def calculate_rmse(img1, img2):
    '''calculate RMSE
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    return rmse(img1, img2)


def save_output_png(input, path, name, mode='clip', cmap=None):
    if not os.path.exists(path):
        os.mkdir(path)
    img = input.detach().squeeze()
    if len(img.size()) == 3:
        img = img.permute(1, 2, 0)
    img = img.cpu().numpy()
    if mode == 'clip':
        img = img * 255.0
        img = np.clip(img, 0, 255)
    elif mode == 'norm':
        img -= np.min(img)
        img /= np.max(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(os.path.join(path, name + '.png'), img)


def save_output_exr(input, path, name, cmap=None):
    if not os.path.exists(path):
        os.mkdir(path)
    img = input.detach().squeeze()
    if len(img.size()) == 3:
        img = img.permute(1, 2, 0)
    img = img.cpu().numpy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(os.path.join(path, name + '.exr'), img)