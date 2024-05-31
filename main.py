import os
import warnings
import argparse
import numpy as np
import pandas as pd
import torch
import csv

from torch.optim import *
import torchvision.transforms as transforms
import importlib
from PIL import Image

from opacus import PrivacyEngine
from opacus.validators import ModuleValidator

import src
import src.trainer as tr
from src.utils import *

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()

# Differtial Privacy args
parser.add_argument('--max_grad_norm', type=float, default=0.1, help='gradient clipping paramter C')
parser.add_argument('--epsilon', type=float, default=3.0, help='privacy budget epsilon')
parser.add_argument('--delta', type=float, default=1e-5, help='privacy budget epsilon')

# Optimization args
parser.add_argument('--optimizer', type=str, default='SGD(lr=1, momentum=0)', help='optimizer with arguments')
parser.add_argument('--epochs', type=int, default=30, help='number of training epochs')
parser.add_argument('--batch_size', type=int, default=4096, help='batch_size')
parser.add_argument('--max_physical_batch_size', type=int, default=1024, help='number of max_physical_batch_size (for distributed training in DP)')
parser.add_argument('--minimizer', type=str, default=None, help="[None]")
parser.add_argument('--rho', type=float, default=0.0, help='perturbation radius of sharpness-aware training. rho=0.0 for DPSGD.')
parser.add_argument('--augmult', type=int, default=16, help='number of augmentations')

# Dataset and dataloader args
parser.add_argument('--data', type=str, default='CIFAR10', help="['CIFAR10' 'FashionMNIST' 'MNIST' 'SVHN' 'CIFAR100']")
parser.add_argument('--model_name', type=str, default='WRN16-4_WS', help= "['DPNASNet_CIFAR','DPNASNet_FMNIST','DPNASNet_MNIST','Handcrafted_CIFAR','Handcrafted_MNIST','ResNet10', 'WRN16-4_WS', 'WRN40-4_WS']")
parser.add_argument('--normalization', default=True)
parser.add_argument('--n_class', type=int, default=10, help='number of classification class')

# Saving args
parser.add_argument('--path', type=str, default="./saved/")
parser.add_argument('--name', type=str, default="saved_name")
parser.add_argument('--result_csv', type=str, default="./saved/result.csv")
parser.add_argument('--memo', type=str)

# GPU args
parser.add_argument('--use_gpu', default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')

# In-sample args
parser.add_argument('--warm', default=True)
parser.add_argument('--pretrained_dir', type=str, default="./saved/cifar-10-warmup/last.pth")
parser.add_argument('--public_data_dir', type=str, default="./data/cifar-10-edm/cifar10_data_sampled_weight0.npz")
parser.add_argument('--public_indices_dir', type=str, default="./data/cifar-10-edm/cifar10_data_sampled_index.pt")
parser.add_argument('--extender', type=str, default=None,  help='[None, DOPE-SGD, Mirror-GD]')
parser.add_argument('--public_batch_size', type=int, default=64, help='public batch_size')

if __name__ == '__main__':
    args = parser.parse_args()
    for arg_name, arg_value in vars(args).items():
        if arg_value == "None":
            setattr(args, arg_name, None)
        if arg_value == "True":
            setattr(args, arg_name, True)
        if arg_value == "False":
            setattr(args, arg_name, False)

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    if args.use_gpu:
        torch.cuda.set_device(args.gpu)
    print('args:', args)
    SAVE_PATH = args.path + args.name

    ### FOR MODELING
    if args.normalization==True:
        if args.data == "CIFAR10":
            ## obtained from synthetic images
            # NORMALIZE = {'mean':[0.4930, 0.4875, 0.4548],
            #              'std':[0.2475, 0.2445, 0.2637]}
            NORMALIZE = {'mean':[0.4914, 0.4822, 0.4465],
                         'std':[0.2023, 0.1994, 0.2010]}
            args.n_class = 10

        elif args.data == "CIFAR100":
            NORMALIZE = {'mean':[0.5071, 0.4867, 0.4408],
                         'std':[0.2675, 0.2565, 0.2761]}
            args.n_class = 100

        elif "MNIST" in args.data: #MNIST, FMNIST
            NORMALIZE = {'mean':[0.1307],
                         'std':[0.3081]}
            
        elif args.data == "SVHN":
            NORMALIZE = {'mean':[0.4377, 0.4438, 0.4728],
                         'std':[0.1980, 0.2010, 0.1970]}
        else:
            raise NotImplementedError("Choose proper dataset.")

    ### Data loader
    if args.extender is not None:
        public_indices = torch.load(args.public_indices_dir)
        val_info = [int(i) for i in range(50000) if i in public_indices]
        data = src.Datasets(data_name=args.data, train_transform = transforms.ToTensor(),val_info=val_info)
        train_loader, _, test_loader = data.get_loader(batch_size=args.batch_size, batch_size_val=args.public_batch_size,
                                                                   drop_last_train=False, num_workers=16)

        public_data = np.load(args.public_data_dir)
        images = public_data['samples'][-50000:]
        images = [Image.fromarray(np.array(img)) for img in images]
        labels = torch.Tensor(public_data['labels'][-50000:]).type('torch.LongTensor')
        labels = torch.argmax(labels, dim=1)
        public_transform  = transforms.Compose([
                        transforms.ToTensor(),
                        ])
        public_dataset = PtDataset(images, labels, transform=public_transform)
        public_loader = DataLoader(public_dataset, batch_size=args.public_batch_size, shuffle=True, 
                                drop_last=False, num_workers=16, pin_memory=True)
        print("Private", len(train_loader)*args.batch_size, "Public:", len(public_loader)*args.public_batch_size)
    elif args.warm:
        public_indices = torch.load(args.public_indices_dir)
        val_info = [int(i) for i in range(50000) if i in public_indices]
        data = src.Datasets(data_name=args.data, train_transform = transforms.ToTensor(),val_info=val_info)
        train_loader, public_loader, test_loader = data.get_loader(batch_size=args.batch_size, batch_size_val=args.public_batch_size,
                                                                   drop_last_train=False, num_workers=16)
    else:    
        data = src.Datasets(data_name=args.data, train_transform = transforms.ToTensor())
        train_loader, test_loader = data.get_loader(batch_size=args.batch_size, drop_last_train=False, num_workers=16)

    ### Model & Optimizer
    ### Load model
    model = src.utils.load_model(model_name=args.model_name, n_classes=args.n_class).cuda() # Load model
    model = ModuleValidator.fix(model)
    if "WRN" in args.model_name:
        set_groupnorm_num_groups(model, num_groups=16)
    
    ModuleValidator.validate(model, strict=False)

    pytorch_total_params = sum(p.numel() for p in model.parameters())

    print("model params: {:.4f}M".format(pytorch_total_params/1000000))

    ### Define optimizer
    optimizer = args.optimizer
    print("Optimizer with :", optimizer)
    exec("optimizer = " + optimizer.split("(")[0] + "(model.parameters()," + optimizer.split("(")[1])

    ### Load PrivacyEngine from Opacus
    privacy_engine = PrivacyEngine()

    model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        epochs=args.epochs,
        target_epsilon=args.epsilon,
        target_delta=args.delta,
        max_grad_norm=args.max_grad_norm,
    )

    optimizer.target_epsilon = args.epsilon
    optimizer.target_delta = args.delta

    print(f"Using sigma={optimizer.noise_multiplier} and C={args.max_grad_norm}")

    rmodel = src.RobModel(model, n_classes=args.n_class, normalize=NORMALIZE).cuda()        
    
    if args.warm:
        print(f"Load pretrained model: {args.pretrained_dir}")
        rmodel.load_dict(args.pretrained_dir)

    ### Start Training
    trainer = tr.DpTrainer(args.name,rmodel)
    trainer.max_physical_batch_size = args.max_physical_batch_size
    trainer.record_rob(train_loader, test_loader)

    if args.warm:
        if args.extender is not None:
            trainer.fit(train_loader=train_loader, max_epoch=args.epochs, start_epoch=0,
                optimizer=optimizer,public_loader=public_loader,extender=args.extender,
                scheduler=None, scheduler_type="Epoch",
                minimizer=None, is_ema=True, augmult=args.augmult,
                save_path=SAVE_PATH, save_best={"Clean(Val)":"HB"},
                save_type=None, save_overwrite=True, record_type="Epoch")
        else:
            trainer.fit(train_loader=train_loader, max_epoch=args.epochs, start_epoch=0,
                optimizer=optimizer, 
                scheduler=None, scheduler_type="Epoch",
                minimizer=None, is_ema=True, augmult=args.augmult,
                save_path=SAVE_PATH, save_best={"Clean(Val)":"HB"},
                save_type=None, save_overwrite=True, record_type="Epoch")            
    else:
        trainer.fit(train_loader=train_loader, max_epoch=args.epochs, start_epoch=0,
        optimizer=optimizer,
        scheduler=None, scheduler_type="Epoch",
        minimizer=None, is_ema=True, augmult=args.augmult,
        save_path=SAVE_PATH, save_best={"Clean(Val)":"HB"},
        save_type=None, save_overwrite=True, record_type="Epoch")            
    ### Evaluation
    rmodel.load_dict(SAVE_PATH+'/last.pth')
    last = rmodel.eval_accuracy(test_loader)

    rmodel.load_dict(SAVE_PATH+'/best.pth')
    best = rmodel.eval_accuracy(test_loader)
    print("LAST: ", last, "BEST: ", best)
    
    df = pd.DataFrame({'name': [args.name], 'last': [last], 'best': [best], 'memo': [args.memo], 'args': [args]})
    if not os.path.exists(args.result_csv):
        df.to_csv(args.result_csv, index=False, mode='w', encoding='utf-8-sig')
    else:
        df.to_csv(args.result_csv, index=False, mode='a', encoding='utf-8-sig', header=False)



