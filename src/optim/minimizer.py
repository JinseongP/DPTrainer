import torch
from collections import defaultdict
from typing import Callable, List, Optional, Union        

from opacus.optimizers import DPOptimizer
from opacus.optimizers.optimizer import _generate_noise


"""
Minimizers for DP
DPSAT: reuse the perturbed gradients of previous iteration.
DPSATMomentum: reuse the momentum of the previous gradients, i.e., reuse the direction w.r.t the previous weight parameters.
                Refer to Park, Jinseong, et al. "Fast sharpness-aware training for periodic time series classification and forecasting." 
                Applied Soft Computing (2023): 110467. (https://www.sciencedirect.com/science/article/pii/S1568494623004854).

Modified from https://github.com/SamsungLabs/ASAM    
"""

class DPSAT:
    def __init__(self, optimizer, model, rho=0.5, augmult=16):
        self.optimizer = optimizer
        self.model = model
        self.rho = rho 
        self.augmult = augmult
        self.state = defaultdict(dict)
        self.skip_calculate_cost = True

    def step(self, cost_fn, *inputs):
        # cost = cost_fn(*inputs)
        # cost.backward()
        self.ascent_step()

        for aug in range(self.augmult):
            cost = cost_fn(*inputs)
            cost.backward()
            with torch.no_grad():
                if aug == 0:
                    for n, grad_sample in enumerate(self.optimizer.grad_samples):
                        self.state["augmult"][n] = grad_sample.clone().detach()
                else:
                    for n, grad_sample in enumerate(self.optimizer.grad_samples):
                        self.state["augmult"][n] += grad_sample.clone().detach()
            self.optimizer.zero_grad()
        with torch.no_grad():
            for n, p in enumerate(self.optimizer.params):
                p.grad_sample = self.state["augmult"][n] / self.augmult   
                    
        self.descent_step()
        # self.update_records("loss_p", cost.item())

    @torch.no_grad()
    def ascent_step(self):
        rho = self.rho
        grads = []
        self.grads_ascent = []

        for n, p in self.model.named_parameters():
            prev_grad = self.state[p].get("prev_grad")
            if prev_grad is None:
                prev_grad = torch.zeros_like(p)
                self.state[p]["prev_grad"] = prev_grad
            self.grads_ascent.append(self.state[p]["prev_grad"].flatten())
            grads.append(torch.norm(prev_grad, p=2))
        grad_norm = torch.norm(torch.stack(grads), p=2) + 1.e-16

        for n, p in self.model.named_parameters():
#             if p.grad is None:
#                 continue
            eps = self.state[p].get("prev_grad")
            eps.mul_(rho / grad_norm)
            self.state[p]["eps"] = eps
            p.add_(eps)
   
        self.optimizer.zero_grad()        
        
    @torch.no_grad()
    def descent_step(self):
        for n, p in self.model.named_parameters(): ## weight to original
            if p.grad is None:
                continue
            p.sub_(self.state[p]["eps"])

        if self.optimizer.pre_step(): ## is last batch -> p.grad_sample to p.grad 
            for n, p in self.model.named_parameters():
                if p.grad is None:
                    continue
                prev_grad = p.grad.clone().detach()
                self.state[p]["prev_grad"] = prev_grad
            self.optimizer.original_optimizer.step()
            self.optimizer.zero_grad()
        else:
            self.optimizer.zero_grad()
        
class DPSATMomentum(DPSAT):
    @torch.no_grad()
    def ascent_step(self):
        rho = self.rho
        grads = []
        for n, p in self.model.named_parameters():
            prev_p = self.state[p].get("prev")
            if prev_p is None:
                prev_p = torch.clone(p).detach()
                self.state[p]["prev"] = prev_p
            grads.append(torch.norm(prev_p - p, p=2))
        grad_norm = torch.norm(torch.stack(grads), p=2) + 1.e-16

        for n, p in self.model.named_parameters():
            prev_p = self.state[p].get("prev")
            eps = prev_p - p
            eps.mul_(rho / grad_norm)
            self.state[p]["eps"] = eps
            p.add_(eps)
             
        self.optimizer.zero_grad()        
        
    @torch.no_grad()
    def descent_step(self):
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            p.sub_(self.state[p]["eps"])
        if self.optimizer.pre_step(): ## is last batch -> p.grad_sample to p.grad 
            for n, p in self.model.named_parameters():
                if p.grad is None:
                    continue
                prev_p = torch.clone(p).detach()
                self.state[p]["prev"] = prev_p
            self.optimizer.original_optimizer.step()
            self.optimizer.zero_grad()
        else:
            self.optimizer.zero_grad()


class DPSATSpan(DPSAT):
    def __init__(self, optimizer, model, rho=0.5, span=16, sampling="positive", augmult=16):
        super().__init__(optimizer, model, rho, augmult)
        self.span = span
        self.sampling = sampling # ["positive", "rand", "onehot"]

    def step(self, cost_fn, *inputs):
        for aug in range(self.augmult):
            self.ascent_step(index=aug)
            cost = cost_fn(*inputs)
            cost.backward()
            with torch.no_grad():
                if aug == 0:
                    for n, grad_sample in enumerate(self.optimizer.grad_samples):
                        self.state["augmult"][n] = grad_sample.clone().detach()
                else:
                    for n, grad_sample in enumerate(self.optimizer.grad_samples):
                        self.state["augmult"][n] += grad_sample.clone().detach()
            self.optimizer.zero_grad()
        with torch.no_grad():
            for n, p in enumerate(self.optimizer.params):
                p.grad_sample = self.state["augmult"][n] / self.augmult   
                    
            self.descent_step()
        # self.update_records("loss_p", cost.item())

    @torch.no_grad()
    def ascent_step(self, index):
        if self.sampling == "positive":
            coef = torch.rand(self.span)
            coef /= torch.sum(coef)
        elif self.sampling == "rand":
            coef = 2 * torch.rand(self.span) - 1
            coef /= torch.sum(coef)
        elif self.sampling == "onehot":
            coef = torch.zeros(self.span)
            coef[index] = 1
        else:
            raise NotImplementedError

        rho = self.rho
        grads = []
        for n, p in self.model.named_parameters():
            prev_p = self.state[p].get("prev")
            if prev_p is None:
                prev_p = torch.clone(p).detach()
                self.state[p]["prev"] = []
                for index_span in range(self.span):
                    self.state[p]["prev"].append(prev_p)
                span_p = prev_p
            else:
                if self.sampling == "onehot":
                    span_p = prev_p[index]
                else:
                    for index_span in range(self.span):
                        if index_span == 0:
                            span_p = coef[index_span] * prev_p[index_span]
                        else:
                            span_p += coef[index_span] * prev_p[index_span]
            self.state[p]["span"] = span_p
            grads.append(torch.norm(span_p - p, p=2))
        grad_norm = torch.norm(torch.stack(grads), p=2) + 1.e-16
        # print("9999",grad_norm)

        for n, p in self.model.named_parameters():
            eps_prev = self.state[p].get("eps")
            if eps_prev is not None:
                self.state[p]["eps_prev"] = torch.clone(eps_prev).detach()
            span_p = self.state[p].get("span")
            eps = span_p - p
            eps.mul_(rho / grad_norm)
            self.state[p]["eps"] = eps
            # print("1111",index,torch.sum(p))
            if index>0:
                p.sub_(self.state[p]["eps_prev"])
            p.add_(eps)
            # print("2222",index,torch.sum(p))

        self.optimizer.zero_grad()        
        

    @torch.no_grad()
    def descent_step(self):
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            p.sub_(self.state[p]["eps"])
            # print("3333",torch.sum(p))

        if self.optimizer.pre_step(): ## is last batch -> p.grad_sample to p.grad 
            for n, p in self.model.named_parameters():
                if p.grad is None:
                    continue
                prev_p = torch.clone(p).detach()
                self.state[p]["prev"].append(prev_p)
                if len(self.state[p]["prev"])>self.span: 
                   self.state[p]["prev"].pop(0)  ## remove the past
            self.optimizer.original_optimizer.step()
            self.optimizer.zero_grad()
        else:
            self.optimizer.zero_grad()



class DPSAM(DPSAT): ##TODO: fix code according Opacus
    def step(self, cost_fn, *inputs):
        raise NotImplementedError

        cost = cost_fn(*inputs)
        cost.backward()
        self.ascent_step()

        for aug in range(self.augmult):
            cost = cost_fn(*inputs)
            cost.backward()
            with torch.no_grad():
                if aug == 0:
                    for n, grad_sample in enumerate(self.optimizer.grad_samples):
                        self.state["augmult"][n] = grad_sample.clone().detach()
                else:
                    for n, grad_sample in enumerate(self.optimizer.grad_samples):
                        self.state["augmult"][n] += grad_sample.clone().detach()
            self.optimizer.zero_grad()
        with torch.no_grad():
            for n, p in enumerate(self.optimizer.params):
                p.grad_sample = self.state["augmult"][n] / self.augmult   
                    
        self.descent_step()

    @torch.no_grad()
    def ascent_step(self):
        grad_norm = []
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            grad_norm.append(torch.norm(p.grad, p=2))
        grad_norm = torch.norm(torch.stack(grad_norm), p=2) + 1.e-16

        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            self.state[p]["eps"] = self.rho / grad_norm * p.grad.clone().detach()
            p.add_(self.state[p]["eps"])
        self.optimizer.zero_grad()


"""
The below minimizers are not modified for DP versions.
"""

class ASAM:

    def __init__(self, optimizer, model, rho=0.5, eta=0.01):
        self.optimizer = optimizer
        self.model = model
        self.rho = rho
        self.eta = eta
        self.state = defaultdict(dict)

    @torch.no_grad()
    def ascent_step(self):
        wgrads = []
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            t_w = self.state[p].get("eps")
            if t_w is None:
                t_w = torch.clone(p).detach()
                self.state[p]["eps"] = t_w
            if 'weight' in n:
                t_w[...] = p[...]
                t_w.abs_().add_(self.eta)
                p.grad.mul_(t_w)
            wgrads.append(torch.norm(p.grad, p=2))
        wgrad_norm = torch.norm(torch.stack(wgrads), p=2) + 1.e-16
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            t_w = self.state[p].get("eps")
            if 'weight' in n:
                p.grad.mul_(t_w)
            eps = t_w
            eps[...] = p.grad[...]
            eps.mul_(self.rho / wgrad_norm)
            p.add_(eps)
        self.optimizer.zero_grad()

    @torch.no_grad()
    def descent_step(self):
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            p.sub_(self.state[p]["eps"])
        self.optimizer.step()
        self.optimizer.zero_grad()


class SAM(ASAM):
    @torch.no_grad()
    def ascent_step(self):
        grads = []
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            grads.append(torch.norm(p.grad, p=2))
        grad_norm = torch.norm(torch.stack(grads), p=2) + 1.e-16
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            eps = self.state[p].get("eps")
            if eps is None:
                eps = torch.clone(p).detach()
                self.state[p]["eps"] = eps
            eps[...] = p.grad[...]
            eps.mul_(self.rho / grad_norm)
            p.add_(eps)
        self.optimizer.zero_grad()

class GSAM:
    def __init__(self, optimizer, model, rho=0.5, alpha=0.1,
                 decay=True, rho_min=0, lr_max=None, lr_min=0):
        self.optimizer = optimizer
        self.model = model
        self.rho = rho
        self.state = defaultdict(dict)
        self.alpha = alpha
        
        self.decay = decay
        assert self.decay and (lr_max is not None)
        self.rho_min = rho_min
        self.lr_max = lr_max
        self.lr_min = lr_min
        
    @torch.no_grad()
    def ascent_step(self):
        grads = []
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            self.state["ascent"][n] = p.grad.clone().detach()
            grads.append(torch.norm(p.grad, p=2))
        grad_norm = torch.norm(torch.stack(grads), p=2) + 1.e-16
        self.ascent_norm = grad_norm.clone().detach()
        
        if self.decay:
            lr = self.optimizer.param_groups[0]['lr']
            rho = self.rho_min + (self.rho - self.rho_min)*(lr - self.lr_min)/(self.lr_max - self.lr_min)
        else:
            rho = self.rho
        
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            eps = self.state[p].get("eps")
            if eps is None:
                eps = torch.clone(p).detach()
                self.state[p]["eps"] = eps
            eps[...] = p.grad[...]
            eps.mul_(rho / grad_norm)
            p.add_(eps)
        self.optimizer.zero_grad()
        
    @torch.no_grad()
    def descent_step(self):
        self.state["descent"] = {}
        grads = []
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            self.state["descent"][n] = p.grad.clone().detach()
            grads.append(torch.norm(p.grad, p=2))
        grad_norm = torch.norm(torch.stack(grads), p=2) + 1.e-16
        self.descent_norm = grad_norm
        
        descents = self.state["descent"]
        ascents = self.state["ascent"]
        
        inner_prod = self.inner_product(descents, ascents)
        
        # get cosine
        cosine = inner_prod / (self.ascent_norm * self.descent_norm + 1.e-16)
        
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            vertical = ascents[n] - cosine * self.ascent_norm * descents[n] / (self.descent_norm + 1.e-16)
            p.grad.sub_(self.alpha*vertical)

        self.optimizer.step()
        self.optimizer.zero_grad()

    def inner_product(self, u, v):
        value = 0
        for key in v.keys():
            value += torch.dot(u[key].flatten(),v[key].flatten())
        return value        