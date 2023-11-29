from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
# x=torch.rand((10,3,224,224))
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce


def trans_ms(f, k):
    return reduce(f, 'b c (h1 h2) (w1 w2) -> b c h1 w1', 'max', h2=k, w2=k)
    # return reduce(f, 'b c (h1 h2) (w1 w2) -> b c h1 w1', 'mean', h2=k, w2=k)


def trans_local(f, k):
    return rearrange(f, 'b c (h hp) (w wp) -> b (c h w) hp wp', hp=k, wp=k)


def trans_batch(f):
    return rearrange(f, 'b c h w -> b (c h w)')


def trans_channel(f):
    return rearrange(f, 'b c h w -> b c (h w)')


def trans_att(fm, eps=1e-6):
    am = torch.pow(torch.abs(fm), 2)
    am = torch.sum(am, dim=1, keepdim=True)
    norm = torch.norm(am, dim=(2, 3), keepdim=True)
    am = torch.div(am, norm + eps)
    return am


def batch_l2(f_s, f_t):
    f_s = trans_batch(f_s)
    f_t = trans_batch(f_t)
    G_s = torch.mm(f_s, rearrange(f_s, 'b chw -> chw  b'))
    G_t = torch.mm(f_t, rearrange(f_t, 'b chw -> chw  b'))
    norm_G_s = F.normalize(G_s, p=2, dim=1)
    norm_G_t = F.normalize(G_t, p=2, dim=1)
    return F.mse_loss(norm_G_s, norm_G_t)


def batch_kl(preds_S, preds_T):
    N, C, H, W = preds_S.shape
    softmax_pred_T = F.softmax(preds_T.view(-1, C * W * H) / 1.0, dim=0)
    logsoftmax = torch.nn.LogSoftmax(dim=0)
    loss = torch.sum(
        softmax_pred_T * logsoftmax(preds_T.view(-1, C * W * H) / 1.0) -
        softmax_pred_T * logsoftmax(preds_S.view(-1, C * W * H) / 1.0)) * (1.0
                                                                           **2)
    return loss / (C * N)


def channel_l2(f_s, f_t):
    f_s = trans_channel(f_s)
    G_s = torch.bmm(f_s, rearrange(f_s, 'b c hw -> b hw  c'))
    norm_G_s = F.normalize(G_s, p=2, dim=2)
    f_t = trans_channel(f_t)
    G_t = torch.bmm(f_s, rearrange(f_t, 'b c hw -> b hw  c'))
    norm_G_t = F.normalize(G_t, p=2, dim=2)
    return F.mse_loss(norm_G_s, norm_G_t) * f_s.size(1)


def channel_kl(preds_S, preds_T):
    N, C, H, W = preds_S.shape
    softmax_pred_T = F.softmax(preds_T.view(-1, W * H) / 1.0, dim=1)
    logsoftmax = torch.nn.LogSoftmax(dim=1)
    loss = torch.sum(
        softmax_pred_T * logsoftmax(preds_T.view(-1, W * H) / 1.0) -
        softmax_pred_T * logsoftmax(preds_S.view(-1, W * H) / 1.0)) * (1.0**2)
    return loss / (C * N)


def globe_l2(f_s, f_t):
    # return F.mse_loss(F.normalize(f_s), F.normalize(f_t))
    return F.mse_loss(f_s, f_t)


def globe_att_l2(f_s, f_t):
    return F.mse_loss(trans_att(f_s), trans_att(f_t))


def local_channel_l2(f_s, f_t):
    return channel_l2(trans_local(f_s, 4), trans_local(f_s, f_t))


def local_channel_kl(f_s, f_t):
    return channel_kl(trans_local(f_s, 4), trans_local(f_s, f_t))


def local_att_channel_l2(f_s, f_t):
    return channel_l2(trans_att(trans_local(f_s, 4)),
                      trans_att(trans_local(f_s, f_t)))


def local_att_channel_kl(f_s, f_t):
    return channel_kl(trans_att(trans_local(f_s, 4)),
                      trans_att(trans_local(f_s, f_t)))


def ms_l2(f_s, f_t):
    return (F.mse_loss(f_s, f_t) + F.mse_loss(trans_ms(f_s, 2), trans_ms(
        f_t, 2)) + F.mse_loss(trans_ms(f_s, 4), trans_ms(f_t, 4))) / 3


def ms_att_l2(f_s, f_t):
    return ms_l2(trans_att(f_s), trans_att(f_t))


class UnifiedLoss(nn.Module):
    def __init__(
        self,
        att=False,
        ms=False,
        local=False,
        batch=False,
        channel=False,
        kl=False,
        **kwargs,
    ):
        super().__init__()
        self.att = att
        self.ms = ms
        self.local = local
        self.batch = batch
        self.channel = channel
        self.kl = kl

    def forward(self, f_s, f_t):
        if self.att:
            f_s, f_t = trans_att(f_s), trans_att(f_t)
        if self.local:
            if self.kl:
                loss = (
                    channel_kl(trans_local(f_s, 2), trans_local(f_t, 2)) +
                    channel_kl(trans_local(f_s, 4), trans_local(f_t, 4)) +
                    channel_kl(trans_local(f_s, 1), trans_local(f_t, 1))) / 3
            else:
                loss = (
                    channel_l2(trans_local(f_s, 2), trans_local(f_t, 2)) +
                    channel_l2(trans_local(f_s, 4), trans_local(f_t, 4)) +
                    channel_l2(trans_local(f_s, 1), trans_local(f_t, 1))) / 3
        elif self.ms:
            f_s, f_t = trans_ms(f_s, 2), trans_ms(f_t, 2)
            if self.batch:
                if self.kl:
                    loss = (batch_kl(trans_ms(f_s, 2), trans_ms(f_t, 2)) +
                            batch_kl(trans_ms(f_s, 4), trans_ms(f_t, 4)) +
                            batch_kl(trans_ms(f_s, 1), trans_ms(f_t, 1))) / 3
                else:
                    loss = (batch_l2(trans_ms(f_s, 2), trans_ms(f_t, 2)) +
                            batch_l2(trans_ms(f_s, 4), trans_ms(f_t, 4)) +
                            batch_l2(trans_ms(f_s, 1), trans_ms(f_t, 1))) / 3
            elif self.channel:
                if self.kl:
                    loss = (channel_kl(trans_ms(f_s, 2), trans_ms(f_t, 2)) +
                            channel_kl(trans_ms(f_s, 4), trans_ms(f_t, 4)) +
                            channel_kl(trans_ms(f_s, 1), trans_ms(f_t, 1))) / 3
                else:
                    loss = (channel_l2(trans_ms(f_s, 2), trans_ms(f_t, 2)) +
                            channel_l2(trans_ms(f_s, 4), trans_ms(f_t, 4)) +
                            channel_l2(trans_ms(f_s, 1), trans_ms(f_t, 1))) / 3
            else:
                loss = (globe_l2(trans_ms(f_s, 2), trans_ms(f_t, 2)) +
                        globe_l2(trans_ms(f_s, 4), trans_ms(f_t, 4)) +
                        globe_l2(trans_ms(f_s, 1), trans_ms(f_t, 1))) / 3
        else:
            if self.batch:
                if self.kl:
                    loss = batch_kl(f_s, f_t)
                else:
                    loss = batch_l2(f_s, f_t)
            elif self.channel:
                if self.kl:
                    loss = channel_kl(f_s, f_t)
                else:
                    loss = channel_l2(f_s, f_t)
            else:
                loss = globe_l2(f_s, f_t)

        return loss


def f_loss(f_s,
           f_t,
           att=False,
           ms=False,
           local=False,
           batch=False,
           channel=False,
           kl=False):
    if att:
        f_s, f_t = trans_att(f_s), trans_att(f_t)
    if local:
        if kl:
            loss = (channel_kl(trans_local(f_s, 2), trans_local(f_t, 2)) +
                    channel_kl(trans_local(f_s, 4), trans_local(f_t, 4)) +
                    channel_kl(trans_local(f_s, 1), trans_local(f_t, 1))) / 3
        else:
            loss = (channel_l2(trans_local(f_s, 2), trans_local(f_t, 2)) +
                    channel_l2(trans_local(f_s, 4), trans_local(f_t, 4)) +
                    channel_l2(trans_local(f_s, 1), trans_local(f_t, 1))) / 3
    elif ms:
        f_s, f_t = trans_ms(f_s, 2), trans_ms(f_t, 2)
        if batch:
            if kl:
                loss = (batch_kl(trans_ms(f_s, 2), trans_ms(f_t, 2)) +
                        batch_kl(trans_ms(f_s, 4), trans_ms(f_t, 4)) +
                        batch_kl(trans_ms(f_s, 1), trans_ms(f_t, 1))) / 3
            else:
                loss = (batch_l2(trans_ms(f_s, 2), trans_ms(f_t, 2)) +
                        batch_l2(trans_ms(f_s, 4), trans_ms(f_t, 4)) +
                        batch_l2(trans_ms(f_s, 1), trans_ms(f_t, 1))) / 3
        elif channel:
            if kl:
                loss = (channel_kl(trans_ms(f_s, 2), trans_ms(f_t, 2)) +
                        channel_kl(trans_ms(f_s, 4), trans_ms(f_t, 4)) +
                        channel_kl(trans_ms(f_s, 1), trans_ms(f_t, 1))) / 3
            else:
                loss = (channel_l2(trans_ms(f_s, 2), trans_ms(f_t, 2)) +
                        channel_l2(trans_ms(f_s, 4), trans_ms(f_t, 4)) +
                        channel_l2(trans_ms(f_s, 1), trans_ms(f_t, 1))) / 3
        else:
            loss = (globe_l2(trans_ms(f_s, 2), trans_ms(f_t, 2)) +
                    globe_l2(trans_ms(f_s, 4), trans_ms(f_t, 4)) +
                    globe_l2(trans_ms(f_s, 1), trans_ms(f_t, 1))) / 3
    else:
        if batch:
            if kl:
                loss = batch_kl(f_s, f_t)
            else:
                loss = batch_l2(f_s, f_t)
        elif channel:
            if kl:
                loss = channel_kl(f_s, f_t)
            else:
                loss = channel_l2(f_s, f_t)
        else:
            loss = globe_l2(f_s, f_t)
    return loss


class HintLoss(nn.Module):
    """Fitnets: hints for thin deep nets, ICLR 2015"""
    def __init__(self):
        super(HintLoss, self).__init__()
        self.crit = nn.MSELoss()

    def forward(self, f_s, f_t):
        return self.crit(f_s, f_t)


if __name__ == '__main__':
    f_s = torch.randn(64, 64, 8, 8)
    f_t = torch.randn(64, 64, 8, 8)
    print(f_loss(f_s, f_t))
    print(f_loss(f_s, f_t, channel=True))
    print(f_loss(f_s, f_t, channel=True, kl=True))
    print(f_loss(f_s, f_t, batch=True))
    print(f_loss(f_s, f_t, batch=True, kl=True))

    print(f_loss(f_s, f_t, ms=True))
    print(f_loss(f_s, f_t, ms=True, channel=True))
    print(f_loss(f_s, f_t, ms=True, channel=True, kl=True))
    print(f_loss(f_s, f_t, ms=True, batch=True))
    print(f_loss(f_s, f_t, ms=True, batch=True, kl=True))

    print(f_loss(f_s, f_t, local=True))
    print(f_loss(f_s, f_t, local=True, kl=True))

    print(f_loss(f_s, f_t, att=True))
    print(f_loss(f_s, f_t, att=True, channel=True))
    print(f_loss(f_s, f_t, att=True, channel=True, kl=True))
    print(f_loss(f_s, f_t, att=True, batch=True))
    print(f_loss(f_s, f_t, att=True, batch=True, kl=True))

    print(f_loss(f_s, f_t, att=True, ms=True))
    print(f_loss(f_s, f_t, att=True, ms=True, channel=True))
    print(f_loss(f_s, f_t, att=True, ms=True, channel=True, kl=True))
    print(f_loss(f_s, f_t, att=True, ms=True, batch=True))
    print(f_loss(f_s, f_t, att=True, ms=True, batch=True, kl=True))

    print(f_loss(f_s, f_t, att=True, local=True))
    print(f_loss(f_s, f_t, att=True, local=True, kl=True))
