import os
import warnings
from collections import OrderedDict
from datetime import datetime
from copy import deepcopy

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.optim import *
from torch.optim.lr_scheduler import *
import torch.nn.functional as F

from opacus.utils.batch_memory_manager import BatchMemoryManager
from opacus import PrivacyEngine
import torchattacks

from ._rm import RecordManager
from .. import RobModel
from ..optim import *
from ._trainer import Trainer
from ..utils import *

def _data(loader):
    while True:
        yield from loader

def inner_product(u, v):
    value = 0
    for key in v.keys():
        v_unsqueeze = v[key].expand_as(u[key])
        value += torch.dot(u[key].flatten(),v_unsqueeze.flatten())
    return value


r"""
Base class for DP trainers.
"""

class DpTrainer(Trainer):

    def __init__(self, name, rmodel):
        super(DpTrainer, self).__init__(name, rmodel)
        self._flag_record_rob = False
        self.state = defaultdict(dict)

    def record_rob(self, train_loader, val_loader,
                   eps=None, alpha=None, steps=None, std=None,
                   n_train_limit=None, n_val_limit=None):
        self._flag_record_rob = True
        self._record_rob_keys = ['Clean']
        self._train_loader_rob = get_subloader(train_loader, n_train_limit)
        self._val_loader_rob = get_subloader(val_loader, n_val_limit)
        self._init_record_keys = self._init_record_keys_with_rob

    def _init_record_keys_with_rob(self, record_type):
        # Add Epoch and Iter to Record_keys
        keys = ["Epoch"]
        if record_type == "Iter":
            keys = ["Epoch", "Iter"]

        # Add Custom Record_dict keys to Record_keys
        for key in self.record_dict.keys():
            keys.append(key)

        # Add robust keys
        # keys += [key + '(Tr)' for key in self._record_rob_keys]
        keys += [key + '(Val)' for key in self._record_rob_keys]

        # Add lr to Record_keys
        keys.append("lr")
        return keys

    # Update Records
    def _update_record(self, record, lr):
        if self._flag_record_rob:
            rob_record = []
            for loader in [self._val_loader_rob]:
                rob_record.append(self.rmodel.eval_accuracy(loader,device=self.device))
            return record + rob_record + [lr]

        return record + [lr]

    def _init_minimizer(self, minimizer, augmult=None):
        if minimizer == None:
            self.minimizer=None
        elif not isinstance(minimizer, str):
            self.minimizer = minimizer
            self.minimizer.augmult = augmult
        else:
            exec("self.minimizer = " + minimizer.split("(")[0] + "(self.optimizer, self.rmodel," + minimizer.split("(")[1])
            self.minimizer.augmult = augmult



    def fit(self, train_loader, optimizer, max_epoch,
            start_epoch=0, end_epoch=None, public_loader=None, extender=None,
            scheduler=None, scheduler_type=None, minimizer=None, is_ema=False, augmult=None,
            save_path=None, save_best=None, save_type="Epoch", 
            save_overwrite=False, record_type="Epoch", augmentation=None, normalize=None, regularizer=None, **kwargs):

        # Init record and save dicts
        self.rm = None
        self.start_time = None
        self.record_dict = OrderedDict()
        self.curr_record_dict = None
        self.best_record_dict = None
        self.save_dict = OrderedDict()

        # Set Epoch and Iterations
        self.max_epoch = max_epoch
        self.train_loader = train_loader

        # Check Save and Record Values
        self._check_valid_options(save_type)
        self._check_valid_options(record_type)

        # Init Optimizer and Schduler
        self.augmult = augmult
        self.is_ema = is_ema
        if self.is_ema:
            self.ema = ExponentialMovingAverage(self.rmodel.parameters(), decay=0.999)

        self._init_optimizer(optimizer)
        self._init_schdeuler(scheduler, scheduler_type)
        self._init_minimizer(minimizer, self.augmult)

        if public_loader is not None:
            public_loader = _data(public_loader)
            self.check_cal_public = True
        else:
            self.check_cal_public = False
        
        self.extender = extender
        if self.extender and "Ours" in self.extender:
            self.extender_minimizer = SAM(self.optimizer, self.rmodel, rho = kwargs["rhoMax"])
            self.rhoMax =  kwargs["rhoMax"]
            self.rhoMin = kwargs["rhoMin"]
            self.adaptive = kwargs["adaptive"]

        self.augmentation = augmentation
        self.normalize = normalize    
        self.regularizer = regularizer

        # Check Save Path
        if save_path is not None:
            if save_path[-1] == "/":
                save_path = save_path[:-1]
            # Save Initial Model
            self._check_path(save_path, overwrite=save_overwrite)
            self._check_path(save_path+"/last.pth", overwrite=save_overwrite, file=True)
            self._check_path(save_path+"/best.pth", overwrite=save_overwrite, file=True)
            self._check_path(save_path+"/record.csv", overwrite=save_overwrite, file=True)
            self._check_path(save_path+"/summary.txt", overwrite=save_overwrite, file=True)

            if save_type in ["Epoch", "Iter"]:
                self._check_path(save_path+"/epoch_iter/", overwrite=save_overwrite)
                self._save_dict_to_file(save_path, 0)
        else:
            raise ValueError("save_path should be given for save_type != None.")

        # Check Save Best
        if save_best is not None:
            if record_type not in ['Epoch', 'Iter']:
                raise ValueError("record_type should be given for save_best = True.")
            if save_path is None:
                raise ValueError("save_path should be given for save_best != None.")

        # Training Start
        for epoch in range(self.max_epoch):
            # Update Current Epoch
            self.epoch = epoch+1

            # If start_epoch is given, update schduler steps.
            if self.epoch < start_epoch:
                if self.scheduler_type == "Epoch":
                    self._update_scheduler()
                elif self.scheduler_type == "Iter":
                    for _ in range(self.max_iter):
                        self._update_scheduler()
                else:
                    pass
                continue

            if end_epoch is not None:
                if self.epoch == end_epoch:
                    break

            self.losses = 0
            self.losses_p = 0

            self.correct = 0
            self.total = 0
        
                
            with BatchMemoryManager(
                data_loader=train_loader, 
                max_physical_batch_size=self.max_physical_batch_size, 
                optimizer=self.optimizer
            ) as memory_safe_data_loader:

                self.max_iter = len(memory_safe_data_loader)
                for i, train_data in enumerate(memory_safe_data_loader):

                    # Init Record and Save dict
                    self._init_record_dict()
                    self._init_save_dict()
                    
                    if self.start_time is None:
                        self.start_time = datetime.now()

                    # Update Current Iteration
                    self.iter = i+1

                    # Set Train Mode
                    self.rmodel = self.rmodel.to(self.device)
                    self.rmodel.train()

                    try:
                        self._do_iter(train_data)
                        warnings.warn(
                            "_do_iter() is deprecated, use _update_weight() instead",
                            DeprecationWarning
                        )
                    except:
                        # Calulate Cost and Update Weight
                        if self.extender is not None:
                            self._update_weight_with_public(train_data, public_loader=public_loader)
                        else:
                            self._update_weight(train_data)

                    # Check Last Batch
                    is_last_batch = (self.iter == self.max_iter)

                    self.losses += self.record_dict["CALoss"]
                    self.record_dict["CALoss"] = self.losses/(i+1)
                    
                    # if self.minimizer:
                    #     self.losses_p += self.record_dict["CALoss^p"]
                    #     self.record_dict["CALoss^p"] = self.losses_p/(i+1)
                    
                    self.record_dict["Acc(Tr)"] = 100 * float(self.correct) / self.total
                    self.rmodel.eval()

                    # Print Training Information

                    def manage_record():
                        if record_type is not None:
                            if self.rm is None:
                                self.record_keys = self._init_record_keys(record_type)
                                self.rm = RecordManager(self.record_keys, start_time=self.start_time)
                                print("["+self.name+"]")
                                print("Training Information.")
                                print("-Epochs:", self.max_epoch)
                                print("-Optimizer:", self.optimizer)
                                print("-Scheduler:", self.scheduler)
                                print("-Save Path:", save_path)
                                print("-Save Type:", str(save_type))
                                print("-Record Type:", str(record_type))
                                print("-Device:", self.device)
                                if save_best is not None:
                                    self._check_save_best(save_best, self.record_keys)
                            else:
                                self.rm.progress()

                            iter_record = self._get_record_from_record_dict()
                            if record_type == "Epoch" and is_last_batch:
                                lr = self.optimizer.param_groups[0]['lr']
                                curr_record = self._update_record([self.epoch]+iter_record, lr=lr)
                                self._update_record_and_best(self.record_keys, curr_record,
                                                            save_best, save_path)
                            elif record_type == "Iter":
                                lr = self.optimizer.param_groups[0]['lr']
                                curr_record = self._update_record([self.epoch, self.iter]+iter_record, lr=lr)
                                self._update_record_and_best(self.record_keys, curr_record,
                                                            save_best, save_path)
                            else:
                                pass
                            
                            if save_path is not None:
                                self.rm.to_csv(save_path+"/record.csv", verbose=False)
                                # self._save_dict_to_file(save_path, is_last=True)
                        else:
                            pass
  
                        # Save Model
                        if save_type == "Epoch" and is_last_batch:
                            self._save_dict_to_file(save_path, self.epoch)
                        elif save_type == "Iter":
                            self._save_dict_to_file(save_path, self.epoch, self.iter)
                        elif save_path is not None and is_last_batch:
                            self._save_dict_to_file(save_path, is_last=True)
                        else:
                            pass

                        # Scheduler Step
                        if (self.scheduler_type == "Epoch") and is_last_batch:
                            self._update_scheduler()
                        elif self.scheduler_type == "Iter":
                            self._update_scheduler()
                        else:
                            pass

                    if self.is_ema:
                        with self.ema.average_parameters():
                            manage_record()
                    else:
                        manage_record()

            # print("Number of batches : {}\n".format(i+1))

        # Print Summary
        if record_type is not None:
            if self.is_ema:
                with self.ema.average_parameters():
                    self.rm.summary(save_path+"/summary.txt")
                    self.rm.to_csv(save_path+"/record.csv", verbose=False)
                    self._save_dict_to_file(save_path, is_last=True)
            else:
                self.rm.summary(save_path+"/summary.txt")
                self.rm.to_csv(save_path+"/record.csv", verbose=False)
                self._save_dict_to_file(save_path, is_last=True)
            
    # Update Weight
    def _update_weight(self, *input):
        # SAM, ASAM, GSAM ...
        self.optimizer.zero_grad()
        if self.minimizer is not None:
            self.minimizer.step(lambda x: self.calculate_cost(x), *input)
            
        else:
            for aug in range(self.augmult):
                cost = self.calculate_cost(*input)
                cost.backward()
                with torch.no_grad():
                    if aug == 0:
                        for n, grad_sample in enumerate(self.optimizer.grad_samples):
                            self.state["augmult"][n] = grad_sample.clone().detach()
                        # for n, p in self.rmodel.named_parameters():###1. 
                        #     if p.grad is None:
                        #         continue
                        #     self.state["augmult"][n] = p.grad_sample.clone().detach()
                    else:
                        for n, grad_sample in enumerate(self.optimizer.grad_samples):
                            self.state["augmult"][n] += grad_sample.clone().detach()
                        # for n, p in self.rmodel.named_parameters():###1. 
                        #     if p.grad is None:
                        #         continue
                        #     self.state["augmult"][n] += p.grad_sample.clone().detach()
                # if aug is not self.augmult-1:
                self.optimizer.zero_grad()
            with torch.no_grad():
                for n, p in enumerate(self.optimizer.params):
                    p.grad_sample = self.state["augmult"][n] / self.augmult      

            self.optimizer.step()


        if self.is_ema:
            self.ema.update()

    # Update Weight
    def _update_weight_with_public(self, *input, **kwargs):

        public_loader = kwargs['public_loader'] #*public_input
        
        # SAM, ASAM, GSAM ...
        self.optimizer.zero_grad()

        if self.minimizer is not None:
            raise NotImplementedError
        
            if 'DPSAT' not in str(self.minimizer.__class__):
                self.calculate_cost(*input).backward()
                self.skip_calculate_cost = False
            else:
                self.skip_calculate_cost = True
            self.minimizer.ascent_step()

            self.calculate_ascent_cost(*input).backward()
            
            if self.augmult:
                with torch.no_grad():
                    for p in self.optimizer.params:
                        grad_sample = self.optimizer._get_flat_grad_sample(p)
                        p.grad_sample = grad_sample.reshape((-1,self.augmult) + grad_sample.shape[1:]).mean(axis=1)

            self.minimizer.descent_step()
            
        else:
            if self.check_cal_public:
                # store gradients of public data first

                if self.extender=="Ours-Clip":
                    self.optimizer.zero_grad()

                    public_input = next(public_loader)
                    public_cost = self.calculate_cost(public_input, use_public=True)
                    public_cost.backward()               

                    self.optimizer.clip_and_accumulate()
                    for n, p in enumerate(self.optimizer.params):
                        p.grad /= len(public_input[0])
                        self.state["pub_grad"][p] = p.grad.clone().detach()
                        # print(torch.sum(self.state["pub_grad"][p]))
                    self.optimizer.zero_grad()


                if self.extender=="Ours-Ind" or self.extender=="Ours-GSAM":
                    rho = self.rhoMin + (self.rhoMax - self.rhoMin) * (1 - (self.epoch / self.max_epoch))
                    for aug in range(self.augmult):
                        public_input = next(public_loader)
                        public_cost = self.calculate_cost(public_input, use_public=True)
                        public_cost.backward()               

                        if self.adaptive:
                            self.eta = 0.01
                            with torch.no_grad():
                                wgrads = []
                                for n, p in self.rmodel.named_parameters():
                                    if p.grad is None:
                                        continue
                                    # t_w = self.state[p].get("pub_eps_"+str(aug))
                                    # if t_w is None:
                                    t_w = torch.clone(p).detach()
                                    self.state["pub_eps_"+str(aug)][p] = t_w
                                    if 'weight' in n:
                                        t_w[...] = p[...]
                                        t_w.abs_().add_(self.eta)
                                        p.grad.mul_(t_w)
                                    wgrads.append(torch.norm(p.grad, p=2))
                                wgrad_norm = torch.norm(torch.stack(wgrads), p=2) + 1.e-16
                                for n, p in self.rmodel.named_parameters():
                                    if p.grad is None:
                                        continue
                                    t_w = self.state["pub_eps_"+str(aug)].get(p)
                                    if 'weight' in n:
                                        p.grad.mul_(t_w)
                                    eps = t_w
                                    eps[...] = p.grad[...]
                                    eps.mul_(rho / wgrad_norm)

                        else:
                            with torch.no_grad():
                                grads = []
                                for n, p in self.rmodel.named_parameters():
                                    if p.grad is None:
                                        continue
                                    grads.append(torch.norm(p.grad, p=2))
                                grad_norm = torch.norm(torch.stack(grads), p=2) + 1.e-16
                                self.state["grad_norm_"+str(aug)] = grad_norm
                                for n, p in self.rmodel.named_parameters():
                                    if p.grad is None:
                                        continue
                                    self.state["pub_eps_"+str(aug)][p] = torch.clone(p.grad).detach() * (rho / grad_norm)
                                    #min(self.optimizer.max_grad_norm, grad_norm) / grad_norm ##TODO: rhoMax
                                
                        self.optimizer.zero_grad()

                else:
                    public_input = next(public_loader)
                    if self.extender=="Ours":
                        public_cost = self.calculate_cost(public_input, use_public=True)
                        public_cost.backward()
                        self.extender_minimizer.ascent_step()

                    public_cost = self.calculate_cost(public_input, use_public=True)
                    public_cost.backward()
                    
                    with torch.no_grad():
                        grads = []
                        for n, p in self.rmodel.named_parameters():
                            if p.grad is None:
                                continue
                            grads.append(torch.norm(p.grad, p=2))
                        grad_norm = torch.norm(torch.stack(grads), p=2) + 1.e-16

                        for n, p in self.rmodel.named_parameters():
                            if p.grad is None:
                                continue
                            self.state[p]["grad"] = torch.clone(p.grad).detach() #* min(self.optimizer.max_grad_norm, grad_norm) / grad_norm

                self.optimizer.zero_grad()
                self.check_cal_public = False

            # use private data
            if self.augmult:
                for aug in range(self.augmult):
                    if self.extender=="Ours-Ind":
                        with torch.no_grad():
                            for n, p in self.rmodel.named_parameters():
                                if p.grad is None:
                                    continue
                                if aug > 0: # we cannot subtract for the first augmult
                                    p.sub_(self.state["pub_eps_"+str(aug-1)][p])
                                p.add_(self.state["pub_eps_"+str(aug)][p]) ## move i th ascent and and remove i-1 th descent
                    
                    cost = self.calculate_cost(*input)
                    cost.backward()
                    
                    if self.extender=="Ours-GSAM":         
                        self.state["descent"] = {}
                        grads = []
                        for n, p in enumerate(self.optimizer.params):
                            self.state["descent"][p] = p.grad_sample.clone().detach()
                            grads.append(torch.norm(p.grad_sample, p=2))
                        grad_norm = torch.norm(torch.stack(grads), p=2) + 1.e-16


                        self.descent_norm = grad_norm
                        self.ascent_norm = self.state["grad_norm_"+str(aug)]
                        
                        ascents = self.state["pub_eps_"+str(aug)]
                        descents = self.state["descent"]
                        inner_prod = inner_product(descents, ascents)

                        # self.ascent_norm = grad_norm
                        # self.descent_norm = self.state["grad_norm_"+str(aug)]
                        
                        # descents = self.state["pub_eps_"+str(aug)]
                        # ascents = self.state["descent"]
                        # inner_prod = inner_product(ascents, descents)
                        
                        # get cosine
                        cosine = inner_prod / (self.ascent_norm * self.descent_norm + 1.e-16)
                        
                        rho = self.rhoMin + (self.rhoMax - self.rhoMin) * (1 - (self.epoch / self.max_epoch))
                        for n, p in enumerate(self.optimizer.params):
                            vertical = ascents[p] - cosine * self.ascent_norm * descents[p] / (self.descent_norm + 1.e-16)
                            p.grad_sample.sub_(rho*vertical)

                    with torch.no_grad():
                        if aug == 0:
                            for n, grad_sample in enumerate(self.optimizer.grad_samples):
                                self.state["augmult"][n] = grad_sample.clone().detach()
                            # for n, p in self.rmodel.named_parameters():###1. 
                            #     if p.grad is None:
                            #         continue
                            #     self.state["augmult"][n] = p.grad_sample.clone().detach()
                        else:
                            for n, grad_sample in enumerate(self.optimizer.grad_samples):
                                self.state["augmult"][n] += grad_sample.clone().detach()
                            # for n, p in self.rmodel.named_parameters():###1. 
                            #     if p.grad is None:
                            #         continue
                            #     self.state["augmult"][n] += p.grad_sample.clone().detach()
                    # if aug is not self.augmult-1:
                    self.optimizer.zero_grad()
                with torch.no_grad():
                    for n, p in enumerate(self.optimizer.params):
                        p.grad_sample = self.state["augmult"][n] / self.augmult


                    # for grad_sample in self.optimizer.grad_samples:
                    #     p.grad_sample = self.state["augmult"][n] / self.augmult
                    #     if self.extender=="DOPE-SGD" or self.extender=="Ours": 
                    #         p.grad_sample.sub_(self.state[p]["grad"])

            else:
                cost = self.calculate_cost(*input)
                cost.backward()

            if self.extender=="DOPE-SGD": #Effectively Using Public Data in Privacy Preserving Machine Learning
                with torch.no_grad():
                    for p in self.optimizer.params:
                        p.grad_sample.sub_(self.state[p]["grad"])

            is_last_batch =  self.optimizer.pre_step() # clip + noise addition
            if self.extender =="Ours-Ind":
                # Calcuate ascent direction in mini-batch with diff each public mini-batch
                with torch.no_grad():
                    for n, p in self.rmodel.named_parameters():
                        if p.grad is None:
                            continue
                        p.sub_(self.state["pub_eps_"+str(self.augmult-1)][p])

            if is_last_batch:
                with torch.no_grad():
                    if self.extender =="Mirror-GD":
                        beta= torch.cos(torch.Tensor([3.1415*(self.epoch/self.max_epoch)/2 /10])).cuda()
                        for n, p in self.rmodel.named_parameters():
                            if p.grad is None:
                                continue
                            """
                            #### uzn
                            p.grad *= max(1, torch.linalg.norm(self.state[p]["grad"])/torch.linalg.norm(p.grad))
                            print("beta:", beta, "private:", torch.linalg.norm(p.grad), "public:", torch.linalg.norm(self.state[p]["grad"]))
                            ####
                            """
                            p.grad.mul_(beta)
                            p.grad.add_((1-beta)*self.state[p]["grad"])
                        
                    elif self.extender=="DOPE-SGD": #Effectively Using Public Data in Privacy Preserving Machine Learning
                        for n, p in self.rmodel.named_parameters():###1. 
                            if p.grad is None:
                                continue
                            p.grad.add_(self.state[p]["grad"])

                    elif self.extender =="Ours":
                        for n, p in self.rmodel.named_parameters():
                            if p.grad is None:
                                continue
                            p.sub_(self.extender_minimizer.state[p]["eps"])
                            p.grad.add_(self.state[p]["grad"])
                    elif self.extender == "Ours-Clip":
                        for n, p in self.rmodel.named_parameters():
                            if p.grad is None:
                                continue
                            p.grad.add_(self.state["pub_grad"][p])
                    elif self.extender=="DPMD":
                        #Public Data-Assisted Mirror Descent for Private Model Training
                        raise NotImplementedError
                    
                    elif self.extender =="AdaDPS":
                        raise NotImplementedError
                    elif "Ours" in self.extender:
                        pass
                    else:
                        raise NotImplementedError("Choose proper extender.")

                self.optimizer.original_optimizer.step()
                self.check_cal_public = True

        if self.is_ema:
            self.ema.update()

    # def calculate_ascent_cost(self, train_data, use_public=False):
    #     r"""
    #     Overridden.
    #     """
    #     images, labels = train_data
    #     images = images.to(self.device)
    #     labels = labels.to(self.device)
    #     augmentation = DataAugmentation()
    #     images = augmentation(images)
    #     logits = self.rmodel(images)
    #     cost = nn.CrossEntropyLoss()(logits, labels)
    #     self.record_dict["CALoss^p"] = cost.item()

    #     if self.skip_calculate_cost:
    #         _, pre = torch.max(logits.data, 1)
    #         self.total += pre.size(0)
    #         self.correct += (pre == labels).sum()        

    #     return cost   
    
    def calculate_cost(self, train_data, use_public=False):
        r"""
        Overridden.
        """
        images, labels = train_data

        images = images.to(self.device)
        labels = labels.to(self.device)

        if self.augmult and self.augmentation=="adv":# and self.epoch>10:
            augmentation = DataAugmentation()
            images = augmentation(images)
            standard_model = deepcopy(self.rmodel.model)#.to_standard_module()
            standard_model.remove_hooks() # not to store individual gradients
            atk = torchattacks.PGD(standard_model, eps=0.01/255, alpha=0.01/255, steps=1, random_start=True)
            # If inputs were normalized, then
            atk.set_normalization_used(mean=self.normalize['mean'], std=self.normalize['std'])
            atk.set_mode_targeted_by_label(quiet=True)
            target_labels = (labels) # loss descending images towards original labels
            adv_images = atk(images, target_labels)
            logits = self.rmodel(adv_images)
            cost = nn.CrossEntropyLoss()(logits, labels)

        elif self.augmult and self.regularizer:#MAN
            augmentation = DataAugmentation()
            images = augmentation(images)
            logits, tmp_var_ = self.rmodel(images) # the model should be wrapped with OpenCLIPMAN
            probs = F.softmax(logits, dim=1)
            onehot_labels = torch.nn.functional.one_hot(labels)
            cost = nn.CrossEntropyLoss()(logits, labels)
            ## for MAN
            loss_var_algo = 0.0
            for var_val in tmp_var_:
                loss_var_algo += torch.mean(var_val)
            cost += self.regularizer * loss_var_algo

        elif self.augmult:# and self.extender is None:
            augmentation = DataAugmentation()
            images = augmentation(images)
            logits = self.rmodel(images)
            probs = F.softmax(logits, dim=1)
            onehot_labels = torch.nn.functional.one_hot(labels)
            cost = nn.CrossEntropyLoss()(logits, labels)

        # elif self.augmult and not use_public:
        #     augmentation = DataAugmentation()
        #     images = augmentation(images)
        #     logits = self.rmodel(images)
        #     probs = F.softmax(logits, dim=1)
        #     cost = nn.CrossEntropyLoss()(logits, labels)
        else:
            logits = self.rmodel(images)
            cost = nn.CrossEntropyLoss()(logits, labels)

        _, pre = torch.max(logits.data, 1)
        self.total += pre.size(0)
        self.correct += (pre == labels).sum()

        self.record_dict["CALoss"] = cost.item()
        return cost        


    def calculate_augmult(self, images, labels, augmult=2):
        """
        Implementation of Unlocking High-Accuracy Differentially Private Image Classification through Scale
        based on https://github.com/deepmind/jax_privacy/blob/95c5b00ae3a4f50e319ccc4e3a014cb0811864e5/jax_privacy/experiments/image_classification/forward.py
        """
        augmentation = DataAugmentation()

        # `images` has shape [NHWC] -> [NKHWC] (K is augmult)
        images = images.unsqueeze(1).expand(-1, augmult, -1, -1, -1)
        labels = labels.unsqueeze(1).expand(-1, augmult,)

        reshaped_images  = images.reshape((-1,) + images.shape[2:])
        reshaped_images = augmentation(reshaped_images)
        reshaped_labels  = labels.reshape((-1,)+ labels.shape[2:])

        logits = self.rmodel(reshaped_images)
        cost = nn.CrossEntropyLoss(reduction='none')(logits, reshaped_labels)
        
        cost = cost.reshape(images.shape[:2])
        cost = torch.mean(cost, axis=1)
        
        logits = logits.reshape(images.shape[:2] + logits.shape[1:])
        selected_logits = logits[:, 0, :]

        return selected_logits, torch.mean(cost)