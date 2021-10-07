import torch
import torch.nn.functional as F
import torch.nn as nn
from models.vggloss import VGG19


class losses_computer():
    def __init__(self, opt):
        self.opt = opt
        
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="mean")
        if not opt.no_labelmix:
            self.labelmix_function = torch.nn.MSELoss()

    def loss(self, x, label, for_real):
        #--- balancing classes ---
        weight_map = get_class_balancing(self.opt, x, label)
        #--- n+1 loss ---
        target = get_n1_target(self.opt, x, label, for_real)
        loss = F.cross_entropy(x, target, reduction='none')
        if for_real:
            loss = torch.mean(loss * weight_map[:, 0, :, :])
        else:
            loss = torch.mean(loss)
        return loss

    def loss_labelmix(self, mask, output_D_mixed, output_D_fake, output_D_real):
        mixed_D_output = mask*output_D_real+(1-mask)*output_D_fake
        return self.labelmix_function(mixed_D_output, output_D_mixed)
    
    def loss_adv(self, y, for_real):
        if for_real:
            y_true = torch.full(y.size(), 1.0, device=y.device)
        else:
            y_true = torch.full(y.size(), 0.0, device=y.device)
        return self.bce_loss(y, y_true)


def get_class_balancing(opt, x, label):
    if not opt.no_balancing_inloss:
        class_occurence = torch.sum(label, dim=(0, 2, 3))
        num_of_classes = (class_occurence > 0).sum()
        coefficients = torch.reciprocal(class_occurence) * torch.numel(label) / (num_of_classes * label.shape[1])
        integers = torch.argmax(label, dim=1, keepdim=True)
        weight_map = coefficients[integers]
    else:
        weight_map = torch.ones_like(x[:, :, :, :])
    return weight_map


def get_n1_target(opt, x, label, target_is_real):
    # returns 0 or 1 tensor with dimensions x.shape
    targets = get_target_tensor(opt, x, target_is_real)
    # number of classes, e.g. 20
    num_of_classes = label.shape[1]
    # gets a tensor of dims [B, W, H]
    integers = torch.argmax(label, dim=1)
    targets = targets[:, 0, :, :] * num_of_classes
    integers += targets.long()
    integers = torch.clamp(integers, min=num_of_classes-1) - num_of_classes + 1
    return integers


def get_target_tensor(opt, x, target_is_real):
    if target_is_real:
        return torch.cuda.FloatTensor(1).fill_(1.0).requires_grad_(False).expand_as(x)
    else:
        return torch.cuda.FloatTensor(1).fill_(0.0).requires_grad_(False).expand_as(x)


class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = VGG19()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss
