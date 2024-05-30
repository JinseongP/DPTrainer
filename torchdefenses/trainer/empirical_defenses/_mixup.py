from numpy as np

import torch
import torch.nn as nn

from ..advtrainer import AdvTrainer


class Mixup(AdvTrainer):
    def __init__(self, rmodel, alpha=1.0, mixup_trainer=None):
        super().__init__("Mixup", rmodel)
        self.mixup_trainer = mixup_trainer
        self.alpha = alpha
        
    def calculate_cost(self, train_data):
        r"""
        Overridden.
        """
        images, labels = train_data
        images = images.to(self.device)
        labels = labels.to(self.device)
        
        images_mixed. y_a, y_b, lam = self.get_mixup_data(images, labels)
        
        logits_mixed = self.rmodel(images_mixed)
        cost = self.mixup_criterion(nn.CrossEntropyLoss(), logits_mixed, y_a, y_b, lam)
        
        self.record_dict["CALoss"] = cost.item()

        return cost


    # https://github.com/facebookresearch/mixup-cifar10/blob/main/train.py
    def get_mixup_data(self, x, y, use_cuda=True):
        '''Returns mixed inputs, pairs of targets, and lambda'''
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1

        batch_size = x.size()[0]
        if use_cuda:
            index = torch.randperm(batch_size).cuda()
        else:
            index = torch.randperm(batch_size)

        if self.mixup_trainer is not None:
            x_ref, y_ref = iter(self.mixup_trainer).next()
        else:
            x_ref, y_ref = x, y
            
        mixed_x = lam * x + (1 - lam) * x_ref[index, :]
        y_a, y_b = y, y_ref[index]
        return mixed_x, y_a, y_b, lam
    
    def get_mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
