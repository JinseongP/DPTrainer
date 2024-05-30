import torch
import torch.nn as nn

from torchattacks import PGD

from .advtrainer import AdvTrainer


class ATMART(AdvTrainer):
    r"""
    Attributes:
        self.rmodel : rmodel.
        self.device : device where rmodel is.
        self.optimizer : optimizer.
        self.scheduler : scheduler (Automatically updated).
        self.curr_epoch : current epoch starts from 1 (Automatically updated).
        self.curr_iter : current iters starts from 1 (Automatically updated).

    Arguments:
        rmodel (nn.Module): rmodel to train.
        eps (float): strength of the attack or maximum perturbation.
        alpha (float): step size.
        steps (int): number of steps.
        random_start (bool): using random initialization of delta.
    """
    def __init__(self, rmodel, eps, alpha, steps, random_start=True, n_classes=10, rand=False):
        super().__init__(rmodel)
        self.atk = PGD(rmodel, eps, alpha, steps, random_start)
        self.n_classes = n_classes
        self.rand = rand

    def calculate_cost(self, train_data, reduction='mean'):
        r"""
        Overridden.
        """
        images, labels = train_data
        images = images.to(self.device)
        labels = labels.to(self.device)

        adv_images = self.atk(images, labels)
        logits_adv = self.rmodel(adv_images)

        class A:
            def sample(self):
                return 1
        m = A()
        if self.rand:
            m = torch.distributions.beta.Beta(torch.tensor([0.5]), torch.tensor([0.5]))
        cost_ce = m.sample()*nn.CrossEntropyLoss()(logits_adv, labels)
        cost_bce = 0
        for i in range(self.n_classes):
            cost_bce += m.sample()*nn.BCEWithLogitsLoss()(logits_adv[:, i], (labels==i).float())/self.n_classes
        self.add_record_item('CALoss', cost_ce.mean().item())
        self.add_record_item('BCELoss', cost_bce.mean().item())
        
        cost = cost_ce + cost_bce

        return cost