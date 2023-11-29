from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask


def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask


def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt


def cosine_similarity(a, b, eps=1e-8):
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + eps)


def pearson_correlation(a, b, eps=1e-8):
    return cosine_similarity(a - a.mean(1).unsqueeze(1),
                             b - b.mean(1).unsqueeze(1), eps)


def inter_class_relation(y_s, y_t):
    return 1 - pearson_correlation(y_s, y_t).mean()


def intra_class_relation(y_s, y_t):
    return inter_class_relation(y_s.transpose(0, 1), y_t.transpose(0, 1))


def inter_loss(z_s, z_t):
    y_s = z_s.softmax(dim=1)
    y_t = z_t.softmax(dim=1)
    return inter_class_relation(y_s, y_t)


def intra_loss(z_s, z_t):
    y_s = z_s.softmax(dim=1)
    y_t = z_t.softmax(dim=1)
    return intra_class_relation(y_s, y_t)


def kl_loss_origin(out_s, out_t, temperature):
    return F.kl_div(F.log_softmax(out_s / temperature, dim=1),
                    F.softmax(out_t / temperature, dim=1),
                    reduction='batchmean') * temperature * temperature


def tckd_trans(logits_student, logits_teacher, target, temperature):
    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
    log_pred_student = torch.log(pred_student)
    return log_pred_student, pred_teacher


def nckd_trans(logits_student, logits_teacher, target, temperature):
    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
    log_pred_student = torch.log(pred_student)
    pred_teacher_part2 = F.softmax(logits_teacher / temperature -
                                   1000.0 * gt_mask,
                                   dim=1)
    log_pred_student_part2 = F.log_softmax(logits_student / temperature -
                                           1000.0 * gt_mask,
                                           dim=1)
    return log_pred_student_part2, pred_teacher_part2


def kd_loss(z_s,
            z_t,
            target,
            temperature,
            intra_class=False,
            tc=False,
            nt=False,
            kl=True,
            cos=False):
    log_pred_student, pred_teacher = tckd_trans(z_s, z_t, target, temperature)
    log_pred_student_part2, pred_teacher_part2 = nckd_trans(
        z_s, z_t, target, temperature)
    if intra_class == True:
        if kl == True:
            loss = kl_loss(z_s.transpose(0, 1), z_t.transpose(0, 1), 4)
        else:
            loss = inter_class_relation(z_s.transpose(0, 1),
                                        z_t.transpose(0, 1))
    elif tc == True:
        if kl == True:
            loss = kl_loss(log_pred_student, pred_teacher, temperature)
        else:
            loss = inter_class_relation(log_pred_student, pred_teacher)
    elif nt == True:
        if kl == True:
            loss = kl_loss(log_pred_student_part2, pred_teacher_part2,
                           temperature)
        else:
            loss = inter_class_relation(log_pred_student_part2,
                                        pred_teacher_part2)
    else:
        if kl == True:
            loss = kl_loss(z_s, z_t, temperature)
        else:
            loss = inter_class_relation(z_s, z_t)
    return loss


# class DistillKL(nn.Module):
#     """Distilling the Knowledge in a Neural Network"""
#     def __init__(self, T):
#         super(DistillKL, self).__init__()
#         self.T = T

#     def forward(self, y_s, y_t, target):

#         return kd_loss(y_s,
#                        y_t,
#                        target,
#                        self.T,
#                        intra_class=opt.use_intra,
#                        tc=opt.use_tc,
#                        nt=opt.use_nt,
#                        kl=opt.use_kl,
#                        cos=opt.use_cos)


class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T=4):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t, target=None):
        p_s = F.log_softmax(y_s / self.T, dim=1)
        p_t = F.softmax(y_t / self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**
                                                         2) / y_s.shape[0]
        return loss
