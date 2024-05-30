import torch
import torch.nn as nn
import torch.nn.functional as F

from torchattacks import PGD
from ..advtrainer import AdvTrainer

#better copy해두고 ctrl z 실시하서 better2 살리
class Better(AdvTrainer):
    def __init__(self, rmodel, eps, alpha, random_start=False, M=1):
        super().__init__("Better", rmodel)
        self.eps = eps
        self.alpha = alpha
        self.random_start = random_start
        self.atk = None
        if M > 1:
            self.atk = PGD(rmodel, eps=eps, alpha=alpha, steps=M-1, random_start=False)

    def calculate_cost(self, train_data):
        r"""
        Overridden.
        """
        images, labels = train_data
        images = images.to(self.device)
        labels = labels.to(self.device)
        logits = self.rmodel(images)

        cost = nn.CrossEntropyLoss()(logits, labels)
        self.record_dict["CALoss"] = cost.item()
        return cost

    # Update Weight
    def _update_weight(self, *input):
        images, labels = tuple(*input)
        images, labels = images.to(self.device), labels.to(self.device)
        if self.random_start:
            # Starting at a uniformly random point
            images = images + torch.empty_like(images).uniform_(-self.eps, self.eps)
            images = torch.clamp(images, min=0, max=1)
        images.requires_grad = True
        
        # SAM, ASAM, GSAM ...
        if self.minimizer is not None:
            if self.atk is not None:
                adv_images = self.atk(images, labels)
                adv_images.requires_grad = True
                self.calculate_cost([adv_images, labels]).backward()
                grad1 = adv_images.grad.detach().clone()
                adv_images.grad.fill_(0)
                if self.clip_grad_norm:
                    torch.nn.utils.clip_grad_norm_(self.rmodel.parameters(), self.clip_grad_norm)
                self.minimizer.ascent_step()
            else:
                self.calculate_cost([images, labels]).backward()
                grad1 = images.grad.detach().clone()
                images.grad.fill_(0)
                if self.clip_grad_norm:
                    torch.nn.utils.clip_grad_norm_(self.rmodel.parameters(), self.clip_grad_norm)
                self.minimizer.ascent_step()
            
            adv_images = images + self.alpha*grad1.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()
            adv_images.requires_grad = True
            
            # BridgedSAM
            if hasattr(self.minimizer, 'middle_step'):
                for i in range(self.minimizer.N-1):
                    if self.atk is not None:
                        adv_images = self.atk(adv_images, labels)
                        adv_images.requires_grad = True
                    self.calculate_cost([adv_images, labels]).backward()
                    grad2 = adv_images.grad.detach().clone()
                    adv_images.grad.fill_(0)
                    if self.clip_grad_norm:
                        torch.nn.utils.clip_grad_norm_(self.rmodel.parameters(), self.clip_grad_norm)
                    self.minimizer.middle_step()

                    adv_images = adv_images + self.alpha*grad2.sign()
                    delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
                    adv_images = torch.clamp(images + delta, min=0, max=1).detach()
                    adv_images.requires_grad = True

            self.calculate_cost([adv_images, labels]).backward()
            if self.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(self.rmodel.parameters(), self.clip_grad_norm)
            self.minimizer.descent_step()
        else:
            cost = self.calculate_cost([adv_images, labels])
            self.optimizer.zero_grad()
            cost.backward()
            self.optimizer.step()

class Better2(AdvTrainer):
    def __init__(self, rmodel, eps, alpha, random_start=False, M=1, loss='AT', beta=None):
        super().__init__("Better2", rmodel)
        self.eps = eps
        self.alpha = alpha
        self.random_start = random_start
        self.atk = None
        if M > 1:
            self.atk = PGD(rmodel, eps=eps, alpha=alpha, steps=M-1, random_start=False)
        if loss=='AT':
            self.calculate_cost = self.calculate_cost_AT
        elif loss=='TRADES':
            self.calculate_cost = self.calculate_cost_TRADES
        else:
            raise ValueError('Not applicable loss')
        self.beta = beta
            
    def calculate_cost_TRADES(self, train_data):
        r"""
        Overridden.
        """
        images, adv_images, labels = train_data
        adv_images = adv_images.to(self.device)
        labels = labels.to(self.device)

        logits_clean = self.rmodel(images)
        loss_ce = nn.CrossEntropyLoss()(logits_clean, labels)

        logits_adv = self.rmodel(adv_images)
        probs_clean = F.softmax(logits_clean, dim=1)
        log_probs_adv = F.log_softmax(logits_adv, dim=1)
        loss_kl = nn.KLDivLoss(reduction='batchmean')(log_probs_adv, probs_clean)

        cost = loss_ce + self.beta * loss_kl

        self.record_dict["Loss"] = cost.item()
        self.record_dict["CELoss"] = loss_ce.item()
        self.record_dict["KLLoss"] = loss_kl.item()
        return cost
    
    def calculate_cost_AT(self, train_data):
        r"""
        Overridden.
        """
        images, _, labels = train_data
        images = images.to(self.device)
        labels = labels.to(self.device)
        logits = self.rmodel(images)

        cost = nn.CrossEntropyLoss()(logits, labels)
        self.record_dict["CALoss"] = cost.item()
        return cost

    # Update Weight
    def _update_weight(self, *input):
        images, labels = tuple(*input)
        images, labels = images.to(self.device), labels.to(self.device)
        adv_images = images.detach().clone()
        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, min=0, max=1)
        adv_images.requires_grad = True
        
        # SAM, ASAM, GSAM ...
        if self.minimizer is not None:
            if self.atk is not None:
                adv_images = self.atk(adv_images, labels)
                adv_images.requires_grad = True
                self.calculate_cost([images, adv_images, labels]).backward()
                grad1 = adv_images.grad.detach().clone()
                adv_images.grad.fill_(0)
                if self.clip_grad_norm:
                    torch.nn.utils.clip_grad_norm_(self.rmodel.parameters(), self.clip_grad_norm)
                self.minimizer.ascent_step()
            else:
                self.calculate_cost([images, adv_images, labels]).backward()
                grad1 = adv_images.grad.detach().clone()
                adv_images.grad.fill_(0)
                if self.clip_grad_norm:
                    torch.nn.utils.clip_grad_norm_(self.rmodel.parameters(), self.clip_grad_norm)
                self.minimizer.ascent_step()
            
            adv_images = adv_images + self.alpha*grad1.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()
            adv_images.requires_grad = True
            
            # BridgedSAM
            if hasattr(self.minimizer, 'middle_step'):
                for i in range(self.minimizer.N-1):
                    if self.atk is not None:
                        adv_images = self.atk(adv_images, labels)
                        adv_images.requires_grad = True
                    self.calculate_cost([images, adv_images, labels]).backward()
                    grad2 = adv_images.grad.detach().clone()
                    adv_images.grad.fill_(0)
                    if self.clip_grad_norm:
                        torch.nn.utils.clip_grad_norm_(self.rmodel.parameters(), self.clip_grad_norm)
                    self.minimizer.middle_step()

                    adv_images = adv_images + self.alpha*grad2.sign()
                    delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
                    adv_images = torch.clamp(images + delta, min=0, max=1).detach()
                    adv_images.requires_grad = True

            self.calculate_cost([images, adv_images, labels]).backward()
            if self.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(self.rmodel.parameters(), self.clip_grad_norm)
            self.minimizer.descent_step()
        else:
            cost = self.calculate_cost([images, adv_images, labels])
            self.optimizer.zero_grad()
            cost.backward()
            self.optimizer.step()
