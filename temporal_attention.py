import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.utils.checkpoint import checkpoint
from einops import rearrange, repeat

def norm(norm_type, out_ch):
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm2d(out_ch, affine=True)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm2d(out_ch, affine=False)
    else:
        raise NotImplementedError('Normalization layer [{:s}] is not found'.format(norm_type))
    return layer


def act(act_type, inplace=True, neg_slope=0.2, n_prelu=1):
    # helper selecting activation
    # neg_slope: for lrelu and init of prelu
    # n_prelu: for p_relu num_parameters
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('Activation layer [{:s}] is not found'.format(act_type))
    return layer


def sequential(*args):
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)

# conv norm activation
def conv_block(in_ch, out_ch, kernel_size, stride=1, dilation=1, padding=0, padding_mode='zeros', norm_type=None,
               act_type='relu', groups=1, inplace=True):
    c = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding,
                  padding_mode=padding_mode, groups=groups)
    n = norm(norm_type, out_ch) if norm_type else None
    a = act(act_type, inplace) if act_type else None
    return sequential(c, n, a)

class WinAtten(nn.Module):
    def __init__(self, ch, block_size=8, num_heads=4, bias=False):
        super(WinAtten, self).__init__()
        self.block_size = block_size
        self.num_heads = num_heads
        self.head_ch = ch // num_heads
        assert ch % num_heads == 0, "ch should be divided by # heads"

        # relative positional embedding: row and column embedding each with dimension 1/2 head_ch
        self.rel_h = nn.Parameter(torch.randn(1, block_size, 1, self.head_ch//2), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn(1, 1, block_size, self.head_ch//2), requires_grad=True)

        self.q_conv = nn.Conv2d(14, ch, kernel_size=1, bias=bias)
        self.k_conv = nn.Conv2d(14, ch, kernel_size=1, bias=bias)
        self.v_conv = nn.Conv2d(3, ch, kernel_size=1, bias=bias)

        self.reset_parameters()

    def forward(self, noisy, curr_aux, prev_aux):
        q = self.q_conv(curr_aux)
        b, c, h, w, block, heads = *q.shape, self.block_size, self.num_heads

        q = rearrange(q, 'b c (h k1) (w k2) -> (b h w) (k1 k2) c', k1=block, k2=block)
        q *= self.head_ch ** -0.5  # b*#blocks, flattened_query, c

        k = self.k_conv(prev_aux)
        k = F.unfold(k, kernel_size=block, stride=block, padding=0)
        k = rearrange(k, 'b (c a) l -> (b l) a c', c=c)

        v = self.v_conv(noisy)
        v = F.unfold(v, kernel_size=block, stride=block, padding=0)
        v = rearrange(v, 'b (c a) l -> (b l) a c', c=c)

        # b*#blocks*#heads, flattened_vector, head_ch
        q, v = map(lambda i: rearrange(i, 'b a (h d) -> (b h) a d', h=heads), (q, v))
        # positional embedding
        k = rearrange(k, 'b (k1 k2) (h d) -> (b h) k1 k2 d', k1=block, h=heads)
        k_h, k_w = k.split(self.head_ch//2, dim=-1)
        k = torch.cat([k_h+self.rel_h, k_w+self.rel_w], dim=-1)
        k = rearrange(k, 'b k1 k2 d -> b (k1 k2) d')

        # b*#blocks*#heads, flattened_query, flattened_neighborhood
        sim = torch.einsum('b i d, b j d -> b i j', q, k)
        attn = F.softmax(sim, dim=-1)
        # b*#blocks*#heads, flattened_query, head_ch
        out = torch.einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h w n) (k1 k2) d -> b (n d) (h k1) (w k2)', b=b, h=(h//block), w=(w//block), k1=block, k2=block)
        return out

    def reset_parameters(self):
        init.kaiming_normal_(self.q_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.k_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.v_conv.weight, mode='fan_out', nonlinearity='relu')
        init.normal_(self.rel_h, 0, 1)
        init.normal_(self.rel_w, 0, 1)

class TemporalAttention(nn.Module):
    def __init__(self, ch, block_size=8, num_heads=4, checkpoint=True):
        super(TemporalAttention, self).__init__()
        self.checkpoint = checkpoint
        self.attention = WinAtten(ch, block_size=block_size, num_heads=num_heads)
        self.feed_forward = nn.Sequential(
            conv_block(32, 16, kernel_size=3, padding=1, padding_mode='reflect', act_type='relu'),
            conv_block(16, 3, kernel_size=3, padding=1, padding_mode='reflect', act_type='relu')
        )

    def forward(self, x):
        out = self.attention(x[0], x[1], x[2])
        out = self.feed_forward(out)

        return out