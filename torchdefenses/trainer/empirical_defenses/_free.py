import torch
import torch.nn as nn

from ..advtrainer import AdvTrainer


class Free(AdvTrainer):
    r"""
    Free adversarial training in 'Adversarial Training for Free!'
    [https://arxiv.org/pdf/1904.12843.pdf]

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
        m (int) : hop steps.
    """
    def __init__(self, rmodel, eps, m):
        super().__init__("Free", rmodel)
        self.eps = eps
        self.m = m
        self.m_accumulated = 0
        self.repeated_data = None
        self.delta = 0

    def _do_iter(self, train_data):
        r"""
        Overridden.
        """
        if self.m_accumulated % self.m == 0:
            self.repeated_data = train_data
        self.m_accumulated += 1

        images, labels = self.repeated_data
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        images.requires_grad = True

        logits_adv = self.rmodel(images + self.delta)
        cost = nn.CrossEntropyLoss()(logits_adv, labels)

        self.optimizer.zero_grad()
        cost.backward()

        grad = images.grad.detach()

        self.optimizer.step()

        self.delta = self.delta + self.eps*grad.sign()
        self.delta = torch.clamp(self.delta.data, -self.eps, self.eps)

        self.record_dict["CALoss"] = cost.item()