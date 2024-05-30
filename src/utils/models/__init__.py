from .lenet import LeNet, LeNetPP
from .preactresnet import PreActBlock, PreActResNet
from .resnet import ResBasicBlock, ResNet
from .resnetWS import ResBasicBlockWS, ResNetWS, ResNet9_WS
from .densenet import DenseNet, Bottleneck
from .vgg import VGG
from .wideresnet import WideResNet
from .wideresnetWS import WideResNetWS
from .dpmodels import SampleConvNet, CNNS
from .dpnas import DPNASNet_CIFAR, DPNASNet_CIFAR100, DPNASNet_FMNIST, DPNASNet_MNIST
from .dpnasWS import DPNASNet_CIFAR_WS, DPNASNet_CIFAR100_WS, DPNASNet_FMNIST_WS, DPNASNet_MNIST_WS

from .handcrafteddp import CNNS
from .dpresnet import DPResBasicBlock, DPResNet

from .openclip import OpenCLIP, OpenCLIPMAN
    
def load_model(model_name, n_classes, **kwargs):
    if model_name == "LeNet":
        return LeNet(n_classes)
    
    if model_name == "LeNetPP":
        return LeNetPP(n_classes)
    
    elif model_name == "MNIST_ATES":
        return MNIST_ATES(n_classes)
    
    elif model_name == "MNIST_DAT":
        return MNIST_DAT(n_classes)
    
    elif model_name == "MNIST_Fast":
        return MNIST_Fast(n_classes)

    elif model_name == "WRN16-4":
        model = WideResNet(depth=16, num_classes=n_classes, 
                           widen_factor=4, dropRate=0.0)
        
    elif model_name == "WRN16-4_WS":
        model = WideResNetWS(depth=16, num_classes=n_classes, 
                           widen_factor=4, dropRate=0.0)
        
    elif model_name == "WRN40-4_WS":
        model = WideResNetWS(depth=40, num_classes=n_classes, 
                           widen_factor=4, dropRate=0.0)
        
    elif model_name == "WRN28-10":
        model = WideResNet(depth=28, num_classes=n_classes, 
                           widen_factor=10, dropRate=0.0)
        
    elif model_name == "WRN34-10":
        model = WideResNet(depth=34, num_classes=n_classes,
                           widen_factor=10, dropRate=0.0)
        
    elif model_name == "PRN18":
        model = PreActResNet(PreActBlock, num_blocks=[2,2,2,2],
                             num_classes=n_classes)
        
    elif model_name == "ResNet10":
        model = ResNet(ResBasicBlock, [1, 1, 1, 1], n_classes, in_channels=1)

    elif model_name == "ResNet10-Channel3":
        model = ResNet(ResBasicBlock, [1, 1, 1, 1], n_classes, in_channels=3) 

    elif model_name == "ResNet10-Channel3_WS":
        model = ResNetWS(ResBasicBlockWS, [1, 1, 1, 1], n_classes, in_channels=3) 

    elif model_name == "ResNet9_WS":
        model = ResNet9_WS(3, n_classes)

    elif model_name == "ResNet18":
        model = ResNet(ResBasicBlock, [2, 2, 2, 2], n_classes)
        
    elif model_name == "ResNet34":
        model = ResNet(ResBasicBlock, [3, 4, 6, 3], n_classes)
        
    elif model_name == "ResNet50":
        model = ResNet(ResBasicBlock, [3, 4, 6, 3], n_classes)
        
    elif model_name == "ResNet101":
        model = ResNet(ResBasicBlock, [3, 4, 23, 3], n_classes)
    
    elif model_name == "ResNet152":
        model = ResNet(ResBasicBlock, [3, 8, 36, 3], n_classes)
        
    elif model_name == "DenseNet121":
        model = DenseNet(Bottleneck, [6,12,24,16], growth_rate=32)

    elif model_name == "DenseNet169":
        model = DenseNet(Bottleneck, [6,12,32,32], growth_rate=32)

    elif model_name == "DenseNet201":
        model = DenseNet(Bottleneck, [6,12,48,32], growth_rate=32)

    elif model_name == "DenseNet161":
        model = DenseNet(Bottleneck, [6,12,36,24], growth_rate=48)
        
    elif model_name == "VGG11":
        model = VGG('VGG11', n_classes)
        
    elif model_name == "VGG13":
        model = VGG('VGG13', n_classes)
        
    elif model_name == "VGG16":
        model = VGG('VGG16', n_classes)
        
    elif model_name == "VGG19":
        model = VGG('VGG19', n_classes)
    
    elif model_name == 'SampleConvNet':
        model = SampleConvNet()
    
    elif model_name == 'SampleConvNet':
        model = SampleConvNet()
    
    elif model_name == 'SampleConvNet':
        model = SampleConvNet()
    
    elif model_name == 'SampleConvNet':
        model = SampleConvNet()
        
    elif model_name == 'DPNASNet_CIFAR':
        model = DPNASNet_CIFAR() 
        
    elif model_name == 'DPNASNet_CIFAR100':
        model = DPNASNet_CIFAR100()    
        
    elif model_name == 'DPNASNet_FMNIST':
        model = DPNASNet_FMNIST()     
        
    elif model_name == 'DPNASNet_MNIST':
        model = DPNASNet_MNIST()   
        
    elif model_name == 'DPNASNet_CIFAR_WS':
        model = DPNASNet_CIFAR_WS() 
        
    elif model_name == 'DPNASNet_CIFAR100_WS':
        model = DPNASNet_CIFAR100_WS()    
        
    elif model_name == 'DPNASNet_FMNIST_WS':
        model = DPNASNet_FMNIST_WS()     
        
    elif model_name == 'DPNASNet_MNIST_WS':
        model = DPNASNet_MNIST_WS()      

    elif model_name == 'Handcrafted_CIFAR':
        model = CNNS["cifar10"](3, input_norm="GroupNorm", num_groups=81, size=None)
        
    elif model_name == 'Handcrafted_MNIST':
        model = CNNS["mnist"](1, input_norm="GroupNorm", num_groups=81, size=None)  
        
    elif model_name == "DPResNet18":
        model = DPResNet(DPResBasicBlock, [2, 2, 2, 2], n_classes)

    elif "OpenCLIPMAN" in model_name:
        model = OpenCLIPMAN(model_name, n_classes)

    elif "OpenCLIP" in model_name:
        model = OpenCLIP(model_name, n_classes)
        
    else:
        raise ValueError('Invalid model name.')
        
    print(model_name, "is loaded.")
        
    return model
