import torch

from ._rm import RecordManager
from .. import RobModel
from ..optim import *
from ._trainer import Trainer
from ..utils import *

r"""
Base class for  trainers.

"""


class Standard(Trainer):
    r"""
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
    """
    def __init__(self, name, rmodel):
        super(Standard, self).__init__(name, rmodel)
        self._flag_record_rob = False

    def _init_record_keys_with_rob(self, record_type):
        # Add Epoch and Iter to Record_keys
        keys = ["Epoch"]
        if record_type == "Iter":
            keys = ["Epoch", "Iter"]

        # Add Custom Record_dict keys to Record_keys
        for key in self.record_dict.keys():
            keys.append(key)

        # Add robust keys
        keys += [key + '(Tr)' for key in self._record_rob_keys]
        keys += [key + '(Val)' for key in self._record_rob_keys]

        # Add lr to Record_keys
        keys.append("lr")
        return keys
    
    def record_rob(self, train_loader, val_loader,
                   eps=None, alpha=None, steps=None, std=None,
                   n_train_limit=None, n_val_limit=None):
        self._flag_record_rob = True
        self._record_rob_keys = ['Clean']
        self._train_loader_rob = get_subloader(train_loader, n_train_limit)
        self._val_loader_rob = get_subloader(val_loader, n_val_limit)
        self._init_record_keys = self._init_record_keys_with_rob

    # Update Records
    def _update_record(self, record, lr):
        if self._flag_record_rob:
            rob_record = []
            for loader in [self._train_loader_rob, self._val_loader_rob]:
                rob_record.append(self.rmodel.eval_accuracy(loader,device=self.device))
            return record + rob_record + [lr]

        return record + [lr]

    def calculate_cost(self, train_data,reduction="mean"):
        r"""
        Overridden.
        """
        images, labels = train_data
        images = images.to(self.device)
        labels = labels.to(self.device)
        logits = self.rmodel(images)

        cost = nn.CrossEntropyLoss(reduction=reduction)(logits, labels)
        if reduction != "none":
            self.record_dict["CALoss"] = cost.item()
        else:
            self.record_dict["CALoss"] = cost.item()

        return cost
