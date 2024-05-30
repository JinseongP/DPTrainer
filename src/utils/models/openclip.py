import torch
import torch.nn as nn
import open_clip

# Define your custom layers to add on top of the CLIP model
class OpenCLIP(nn.Module):
    def __init__(self, model_name, num_classes):
        super(OpenCLIP, self).__init__()
        if model_name == "ViT-B-16-OpenCLIP":
            self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained='laion2b_s34b_b88k')
        else:
            print("ADD models-")
        # Define custom layers to add on top of the CLIP model
        clip_output_dim = self.clip_model.visual.output_dim  # Get the output dimension of the CLIP model
        self.additional_layers = nn.Sequential(
            # nn.Linear(clip_output_dim, num_classes),
            nn.Linear(clip_output_dim, 512),
            nn.ReLU(), #nn.Tanh()
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)  
        )

        for param in self.clip_model.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        x = torch.nn.functional.interpolate(x, size=self.clip_model.visual.image_size, mode='bilinear', align_corners=False)

        with torch.no_grad():
            x = self.clip_model.encode_image(x)
        x = self.additional_layers(x)
        return x

class OpenCLIPMAN(nn.Module):
    def __init__(self, model_name, num_classes):
        super(OpenCLIPMAN, self).__init__()
        if "ViT-B-16-OpenCLIP" in model_name:
            self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained='laion2b_s34b_b88k')
        else:
            print("ADD models-")
        # Define custom layers to add on top of the CLIP model
        clip_output_dim = self.clip_model.visual.output_dim  # Get the output dimension of the CLIP model
        self.layer1  = nn.Sequential(
            nn.Linear(clip_output_dim, 512),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU())
        self.layer3 = nn.Linear(256, num_classes)  

        for param in self.clip_model.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        x = torch.nn.functional.interpolate(x, size=self.clip_model.visual.image_size, mode='bilinear', align_corners=False)

        with torch.no_grad():
            x = self.clip_model.encode_image(x)
        x = self.layer1(x)
        act1 = torch.mean(x**2,axis = [1])
        x = self.layer2(x)
        act2 = torch.mean(x**2,axis = [1])
        x = self.layer3(x)
        var_list = [act1, act2]

        return x, var_list
