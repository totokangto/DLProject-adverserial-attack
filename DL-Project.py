# -*- coding: utf-8 -*-


import os
import math
import csv
import pickle
from urllib import request
import scipy.stats as st

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
import torchvision
from torchvision import datasets, models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score

device = torch.device("cuda:0")

import gdown

ids = [
    '1XHEWIiTv9Czjn9RJ6IHu_fWXfrteks5l',
    '1gwa_5bTO3dchDlC3WZ3nt_0VvwBkvFfi',
    '1DCUuuy20k-dNbUnCyi0HI9O7qy8hOpE5'
]

outputs = [
    'train.csv',
    'test.csv',
    'img.zip'
]

for i, o in zip(ids, outputs):
    gdown.download(id=i, output=o, quiet=False)

file_name = "img.zip"
output_dir = "img"
os.system("unzip "+file_name)


##load image metadata (Image_ID, true label, and target label)
def load_ground_truth(fname):
    image_id_list = []
    label_ori_list = []
    label_tar_list = []

    df = pd.read_csv(fname)
    for _, row in df.iterrows():
        image_id_list.append( row['ImageId'] )
        label_ori_list.append( int(row['TrueLabel']) - 1 )
        label_tar_list.append( int(row['TargetClass']) - 1 )
    gt = pickle.load(request.urlopen('https://gist.githubusercontent.com/yrevar/6135f1bd8dcf2e0cc683/raw/d133d61a09d7e5a3b36b8c111a8dd5c4b5d560ee/imagenet1000_clsid_to_human.pkl'))
    return image_id_list,label_ori_list,label_tar_list, gt

## simple Module to normalize an image
class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = torch.Tensor(mean)
        self.std = torch.Tensor(std)
    def forward(self, x):
        return (x - self.mean.type_as(x)[None,:,None,None]) / self.std.type_as(x)[None,:,None,None]

"""# Training"""

norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
trn = transforms.Compose([transforms.ToTensor(),])
ids, origins, targets, gt = load_ground_truth('train.csv')

batch_size = 20
max_iterations = 10
input_path = 'images/'
epochs = int(np.ceil(len(ids) / batch_size))

img_size = 299
lr = 1.6 #step size
epsilon = 32 # L_inf norm bound


resnet = models.resnet50(weights="IMAGENET1K_V1").eval()
vgg = models.vgg16_bn(weights="IMAGENET1K_V1").eval()

for param in resnet.parameters():
    param.requires_grad = False
for param in vgg.parameters():
    param.requires_grad = False

resnet.to(device)
vgg.to(device)

torch.manual_seed(42)
torch.backends.cudnn.deterministic = True

preds_ls = []
labels_ls =[]
origin_ls = []
preds_res_ls = []
decay_factor = 1.0


torch.cuda.empty_cache()
for k in tqdm(range(epochs), total=epochs):
    batch_size_cur = min(batch_size, len(ids) - k * batch_size)
    X_ori = torch.zeros(batch_size_cur, 3, img_size, img_size).to(device)
    delta = torch.zeros_like(X_ori, requires_grad=True).to(device)
    for i in range(batch_size_cur):
        X_ori[i] = trn(Image.open(input_path + ids[k * batch_size + i] + '.png'))
    ori_idx = origins[k * batch_size:k * batch_size + batch_size_cur]
    labels = torch.tensor(targets[k * batch_size:k * batch_size + batch_size_cur]).to(device)
    g = torch.zeros_like(X_ori).detach().to(device)
    for t in range(max_iterations):
        """
        NI-FGSM

        delta_nes = delta + lr*decay_factor*g
        logits = resnet(norm(X_ori + delta_nes))
        loss = nn.CrossEntropyLoss(reduction='sum')(logits,labels)
        loss.backward()
        grad_c = delta.grad.clone()
        delta.grad.zero_()
        """

        # TI-FGSM
        nsig = 3*(3**(1/2))
        x = np.linspace(-nsig, nsig, 7) 
        rv = st.norm(loc = 0, scale = 3/(3**(1/2))) 
        kern1d = rv.pdf(x) 
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        kernel = kernel.astype(np.float32)
        kernel = kernel.reshape(-1,7,7)
        kernel = kernel[np.newaxis, ...]
        kernel = np.tile(kernel, (3, 1, 1, 1))
        stacked_kernel = torch.from_numpy(kernel).to(device)
        grad_c = F.conv2d(grad_c, stacked_kernel, stride=1, padding="same", groups=3)

        # MI-FGSM
        g = decay_factor*g + grad_c/torch.norm(grad_c,p=1)
        delta.data = delta.data - lr * torch.sign(g)

        delta.data = delta.data.clamp(-epsilon / 255,epsilon / 255)

    X_pur = norm(X_ori + delta)
    preds = torch.argmax(vgg(X_pur), dim=1)
    preds_res = torch.argmax(resnet(X_pur), dim=1)
    preds_ls.append(preds.cpu().numpy())
    labels_ls.append(labels.cpu().numpy())
    origin_ls.append(ori_idx)
    preds_res_ls.append(preds_res.cpu().numpy())



df = pd.DataFrame({
    'origin': [a for b in origin_ls for a in b],
    'pred': [a for b in preds_ls for a in b],
    'label': [a for b in labels_ls for a in b]
})

df1 = df[df['pred']==df['origin']]
print("pred = origin",len(df1))




print("accuracy_score :",accuracy_score(df['label'], df['pred']))

"""* This performance will not be reproduced with Colab. Please don't worry and do your best.

"""

df.to_csv('submission.csv')

