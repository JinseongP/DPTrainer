import torch
import torch.nn as nn
import torch.nn.functional as F

from torchattacks import PGD, TPGD

from ..advtrainer import AdvTrainer


class ATRADES(AdvTrainer):
    r"""
    AT + TRADES

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
        eta (float): the ratio between the loss of AT and TRADES. (eta=1 is equal to AT)
        inner_loss (str): inner loss to maximize ['ce', 'kl'].
    """
    def __init__(self, rmodel, eps, alpha, steps, beta, eta, inner_loss='ce'):
        super().__init__("TRADES", rmodel)
        if inner_loss == 'ce':
            self.atk = PGD(rmodel, eps, alpha, steps)
        elif inner_loss == 'kl':
            self.atk = TPGD(rmodel, eps, alpha, steps)
        else:
            raise ValueError("Not valid inner loss.")
        self.beta = beta
        self.eta = eta

    def calculate_cost(self, train_data):
        r"""
        Overridden.
        """
        images, labels = train_data
        images = images.to(self.device)
        labels = labels.to(self.device)

        adv_images = self.atk(images, labels)

        # TRADES
        logits_clean = self.rmodel(images)
        loss_ce = nn.CrossEntropyLoss()(logits_clean, labels)

        logits_adv = self.rmodel(adv_images)
        probs_clean = F.softmax(logits_clean, dim=1)
        log_probs_adv = F.log_softmax(logits_adv, dim=1)
        loss_kl = nn.KLDivLoss(reduction='batchmean')(log_probs_adv, probs_clean)

        cost_trades = loss_ce + self.beta * loss_kl

        # AT
        cost_at = nn.CrossEntropyLoss()(logits_adv, labels)

        # Combine
        cost = self.eta*cost_at + (1-self.eta)*cost_trades

        self.record_dict["Loss"] = cost.item()
        self.record_dict["ATLoss"] = cost_at.item()
        self.record_dict["TRLoss"] = cost_trades.item()

        return cost
