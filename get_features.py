from torchvision.models import resnet152
import torchvision.transforms as transforms
import sys
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import h5py
import copy

import ipdb

json_file = sys.argv[1]
batchsize = 50
device = "cuda"

scaler = transforms.Scale((224,224))
totensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

# transform = transforms.Compose([
#     transforms.Resize(size=(224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])

def get_image(file_name):
    img = Image.open(file_name)
    img = totensor(scaler(img))
    if img.shape != (3, 224, 224):
        if len(img.shape) == 2:
            img = img.unsqueeze(0)

        img = img.repeat_interleave(3, axis=0)

    img = normalize(img).unsqueeze(0)
    return img

model = resnet152(pretrained=True)
for param in model.parameters():
     param.requires_grad = False

df = pd.read_json(json_file, orient="records", lines=True)
uniq_files = df[["filename", "id"]]
uniq_files = uniq_files.drop_duplicates()
uniq_files["image"] = 0

# for i, v in uniq_files.iterrows():
#     v["image"] = get_image(v["filename"])

uniq_files["image"] = uniq_files.filename.apply(get_image)
# uniq_files2 = uniq_files[uniq_files.image.apply(np.shape) != (1, 224, 224, 3)]
# uniq_files = uniq_files.drop(uniq_files[uniq_files.image.apply(np.shape) != (1, 224, 224, 3)].index)

# features1 = h5py.File('images.hdf5', 'w')
# features1.create_dataset('images', data=torch.cat(uniq_files.image.tolist(), axis=0))
# features1.create_dataset('ids', data=uniq_files.id)
# features1.close()


model1 = copy.deepcopy(nn.Sequential(*list(model.children())[:-1]))

model1.to(device)
model1.eval()

all_features1 = np.zeros((0, 2048))
for i in range(0,len(uniq_files) - len(uniq_files) % batchsize,batchsize):
    els = uniq_files.iloc[i:i+batchsize].image
    els = torch.cat(els.tolist(), dim=0)
    els = els.to(device)
    all_features1 = np.concatenate((all_features1, model1(els).cpu().numpy().reshape(batchsize, -1)), axis=0)

if len(uniq_files) % batchsize != 0:
    els = uniq_files.iloc[i+batchsize:].image
    els = torch.cat(els.tolist(), dim=0)
    els = els.to(device)
    all_features1 = np.concatenate((all_features1, model1(els).cpu().numpy().reshape(-1, 2048)), axis=0)

del model1

features1 = h5py.File('../../features1.hdf5', 'w')
features1.create_dataset('features', data=all_features1)
features1.create_dataset('ids', data=uniq_files.id)
features1.close()

del all_features1



# TOO BIG

# model1 = copy.deepcopy(nn.Sequential(*list(model.children())[:-2]))
# model1.to(device)
# model1.eval()

# all_features2 = np.zeros((0, 2048, 7, 7))
# for i in range(0,len(uniq_files) - len(uniq_files) % batchsize,batchsize):
#     els = uniq_files.iloc[i:i+batchsize].image
#     els = torch.cat(els.tolist(), dim=0)
#     els = els.to(device)
#     all_features2 = np.concatenate((all_features2, model1(els).cpu().numpy()), axis=0)

# if len(uniq_files) % batchsize != 0:
#     els = uniq_files.iloc[i+batchsize:].image
#     els = torch.cat(els.tolist(), dim=0)
#     els = els.to(device)
#     all_features2 = np.concatenate((all_features2, model1(els).cpu().numpy()), axis=0)

# features2 = h5py.File('features2.hdf5', 'w')
# features2.create_dataset('features', data=all_features2)
# features2.create_dataset('ids', data=uniq_files.id)
# features2.close()
