
#%% set the background
from __future__ import print_function, division
import torch
from torchvision import datasets, transforms
import os
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from CAM import *

#%% put the model into the CAM
model_file_path= 'YOUR MODEL FOLDER PATH / MODEL TO LOAD.pt'
model = torch.load(model_load_path)
model=model.eval()

#%% initialize the CAM module
cam = CAM(model)

#%% test it for one image
# Load an image
img = Image.open('PATH TO YOUR TEST IMAGE')

# define Augmentation steps.
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
PIL_tops = transforms.Compose([transforms.Resize(224),transforms.CenterCrop(224)])
tensor_tops = transforms.Compose([transforms.ToTensor(), normalize])
cropped_img = PIL_tops(img)
trans_img = tensor_tops(cropped_img)

# predict
out=model(trans_img.unsqueeze(0).cuda())

# plot it
img_output = cam.visualize(11,cropped_img,alpha=0.8)
img_output = np.array(img_output)
plt.subplot(121)
plt.imshow(cropped_img)
plt.subplot(122)
plt.imshow(img_output)
plt.show()
