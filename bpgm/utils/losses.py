from torch import nn
import torch

from model.models import Vgg19

    
LABEL_REAL, LABEL_FAKE = 1.0, 0.0


class VGGLoss(nn.Module):
    def __init__(self, layids = None):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19()
        self.vgg.cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        self.layids = layids

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        if self.layids is None:
            self.layids = list(range(len(x_vgg)))
        for i in self.layids:
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


class AdversarialLoss:
    def __init__(self):
        self.loss = nn.BCEWithLogitsLoss(reduction="mean")

    def __call__(self, y_hat, label):
        if label not in [LABEL_REAL, LABEL_FAKE]:
            raise Exception("Invalid label is passed to adversarial loss")
        
        y_true = torch.full(y_hat.size(), label, device="cuda:0")
        return self.loss(y_hat, y_true)
    