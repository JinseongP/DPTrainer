import torch
import torch.nn as nn

from torchattacks import FFGSM

from ..advtrainer import AdvTrainer


class Fast(AdvTrainer):
    r"""
    Fast adversarial training in 'Fast is better than free: Revisiting adversarial training'
    [https://arxiv.org/abs/2001.03994]

    Attributes:
        self.rmodel : rmodel.
        self.device : device where rmodel is.
        self.optimizer : optimizer.
        self.scheduler : scheduler (Automatically updated).
        self.max_epoch : total number of epochs.
        self.max_iter : total number of iterations.
        self.epoch : current epoch starts from 1 (Automatically updated).
        self.iter : current iters starts from 1 (Automatically updated).
            * e.g., is_last_batch = (self.iter == self.max_iter)
        self.record_dict : dictionary for printing records.

    Arguments:
        rmodel (nn.Module): rmodel to train.
        eps (float): strength of the attack or maximum perturbation.
        alpha (float): alpha in the paper.
    """
    def __init__(self, rmodel, eps, alpha):
        super().__init__("Fast", rmodel)
        self.atk = FFGSM(rmodel, eps, alpha)

    def calculate_cost(self, train_data):
        r"""
        Overridden.
        """
        images, labels = train_data
        images = images.to(self.device)
        labels = labels.to(self.device)

        adv_images = self.atk(images, labels)
        logits_adv = self.rmodel(adv_images)

        cost = nn.CrossEntropyLoss()(logits_adv, labels)
        self.record_dict["CALoss"] = cost.item()
        return cost
