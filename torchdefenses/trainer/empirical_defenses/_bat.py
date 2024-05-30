import torch
import torch.nn as nn
import torch.nn.functional as F

from torchattacks import PGD, TPGD

from ..advtrainer import AdvTrainer


class BAT(AdvTrainer):
    r"""
    Adversarial training in 'Bridged Adversarial Training'
    [https://arxiv.org/abs/2108.11135]

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
        alpha (float): step size.
        steps (int): number of steps.
        beta (float): trade-off regularization parameter.
        m (int): number of bridges.
        loss (str): inner maximization loss. [Options: 'ce', 'kl'] (Default: 'ce')
    """
    def __init__(self, rmodel, eps, alpha, steps, beta, m, loss='ce'):
        super().__init__("BAT", rmodel)
        if loss == 'ce':
            self.atk = PGD(rmodel, eps, alpha, steps)
        elif loss == 'kl':
            self.atk = TPGD(rmodel, eps, alpha, steps)
        else:
            raise ValueError(type + " is not a valid type. [Options: 'ce', 'kl']")
        self.beta = beta
        self.m = m

    def calculate_cost(self, train_data):
        r"""
        Overridden.
        """
        images, labels = train_data
        images = images.to(self.device)
        labels = labels.to(self.device)

        logits_clean = self.rmodel(images)
        probs_clean = F.softmax(logits_clean, dim=1)

        loss_natural = nn.CrossEntropyLoss()(logits_clean, labels)

        adv_images = self.atk(images, labels)
        pert = (adv_images-images).detach()

        loss_robust = 0
        probs_pre = probs_clean
        for i in range(self.m):
            logits_adv = self.rmodel(images + pert*(i+1)/self.m)
            log_probs_adv = F.log_softmax(logits_adv, dim=1)
            loss_robust += nn.KLDivLoss(reduction='batchmean')(log_probs_adv, probs_pre)
            probs_pre = F.softmax(logits_adv, dim=1)

        cost = loss_natural + self.beta*loss_robust

        self.record_dict["Loss"] = cost.item()
        self.record_dict["CELoss"] = loss_natural.item()
        self.record_dict["KLLoss"] = loss_robust.item()

        return cost
