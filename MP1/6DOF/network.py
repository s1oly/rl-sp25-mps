import torch
import torch.nn as nn
from torchvision import models
from utils import make_rotation_matrix
from utils import rotation_6d_to_matrix
from absl import flags
FLAGS = flags.FLAGS


class SimpleModel(nn.Module):
    def __init__(self, num_classes):
        super(SimpleModel, self).__init__()
        self.resnet = models.resnet18(weights='IMAGENET1K_V1')
        self.resnet.fc = nn.Identity()
        self.num_classes = num_classes
        num_outputs = self.num_classes + 6 + 3 # was 9 previously for rot matrix
        self.one = 1

    # For adding iterative feedback,
    # need to add feed back of rotation prediction + translation 

        self.head = nn.Sequential(
            nn.Linear(512 + 4 + 6 + 3, 256), 
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_classes) # num_outputs
        )

        self.heads_list = nn.ModuleList([nn.Sequential(
            nn.Linear(512 + 4 + 6 + 3, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256,6 + 3) # num_outputs
        ) for i in range(13)])
       

    def forward(self, image, bbox, r_current = None, t_current = None):
        if(r_current is None):
            r_current = torch.zeros(image.shape[0], 6, device=image.device) # shape needs to include batch size, don't know how
        if(t_current is None):
            t_current = torch.zeros(image.shape[0], 3, device=image.device)
        x = self.resnet(image)
        features = torch.cat((x, bbox, r_current, t_current), dim=1)
        x = self.head(features)
        # logits, R, t = torch.split(x, [self.num_classes, 9*self.one, 3*self.one], dim=1)
        logits = x
        outputs = torch.stack([heads(features) for heads in self.heads_list], dim = 1)
        # R = outputs[:, :, :9]
        rot_6d = outputs[:, :, :6]
        t = outputs[:, :, 6:] # was 9, is now 6

        return logits, rot_6d, t

    def process_output(self, outs):
        with torch.no_grad():
            # logits, R, t = outs 
            logits, rot_6d, t = outs 
            cls = logits.argmax(dim=1)
            batch_size = rot_6d.shape[0]  # was R
            # R = make_rotation_matrix(R[torch.arange(batch_size), cls].reshape(-1, 3, 3))
            R = rotation_6d_to_matrix(rot_6d[torch.arange(batch_size), cls].reshape(-1,6))
            t = t[torch.arange(batch_size), cls].reshape(-1, 3, 1)
            return cls, R, t
    
