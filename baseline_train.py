# As usual, a bit of setup
import time, os, json
import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from model import Model

import ipdb

# from coco_utils import load_coco_data, sample_coco_minibatch, decode_captions

# # Load COCO data from disk; this returns a dictionary
# # We'll work with dimensionality-reduced features for this notebook, but feel
# # free to experiment with the original features by changing the flag below.
# data = load_coco_data(pca_features=True)

# # Print out all the keys and values from the data dictionary
# for k, v in data.items():
#     if type(v) == np.ndarray:
#         print(k, type(v), v.shape, v.dtype)
#     else:
#         print(k, type(v), len(v))

# trn = pd.DataFrame([{"id":data["train_image_idxs"][i], "captions":data["train_captions"][i,:], "features":data["train_features"][data["train_image_idxs"][i],:]} for i in range(len(data["train_image_idxs"]))])
# val = pd.DataFrame([{"id":data["val_image_idxs"][i], "captions":data["val_captions"][i,:], "features":data["val_features"][data["val_image_idxs"][i],:]} for i in range(len(data["val_image_idxs"]))])

# trn.to_json("train.json", orient="records", lines=True, force_ascii=False)
# val.to_json("val.json", orient="records", lines=True, force_ascii=False)

# outputs ->
# val_captions <class 'numpy.ndarray'> (195954, 17) int32
# train_urls <class 'numpy.ndarray'> (82783,) <U63
# val_features <class 'numpy.ndarray'> (40504, 512) float32
# train_captions <class 'numpy.ndarray'> (400135, 17) int32
# train_image_idxs <class 'numpy.ndarray'> (400135,) int32
# idx_to_word <class 'list'> 1004
# word_to_idx <class 'dict'> 1004
# val_urls <class 'numpy.ndarray'> (40504,) <U63
# train_features <class 'numpy.ndarray'> (82783, 512) float32
# val_image_idxs <class 'numpy.ndarray'> (195954,) int32



# Look at the data
# batch_size = 3
# captions, features, urls = sample_coco_minibatch(data, batch_size=batch_size)
# for i, (caption, url) in enumerate(zip(captions, urls)):
#     plt.imshow(image_from_url(url))
#     plt.axis('off')
#     caption_str = decode_captions(caption, data['idx_to_word'])
#     plt.title(caption_str)
#     plt.show()

# greedy sampling
# for split in ['train', 'val']:
#     minibatch = sample_coco_minibatch(small_data, split=split, batch_size=1)
#     gt_captions, features, urls = minibatch
#     gt_captions = decode_captions(gt_captions, data['idx_to_word'])

#     sample_captions = small_rnn_model.sample_greedily(features)
#     sample_captions = decode_captions(sample_captions, data['idx_to_word'])

#     for gt_caption, sample_caption, url in zip(gt_captions, sample_captions, urls):
#         plt.imshow(image_from_url(url))
#         plt.title('%s\n%s\nGT:%s' % (split, sample_caption, gt_caption))
#         plt.axis('off')
#         plt.show()




    # # Data loading code
    # traindir = os.path.join(args.data, 'train')
    # valdir = os.path.join(args.data, 'val')
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])

    # train_loader = torch.utils.data.DataLoader(
    #     datasets.ImageFolder(traindir, transforms.Compose([
    #         transforms.RandomSizedCrop(224),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         normalize,
    #     ])),
    #     batch_size=args.batch_size, shuffle=True,
    #     num_workers=args.workers, pin_memory=True)

    # val_loader = torch.utils.data.DataLoader(
    #     datasets.ImageFolder(valdir, transforms.Compose([
    #         transforms.Scale(256),
    #         transforms.CenterCrop(224),
    #         transforms.ToTensor(),
    #         normalize,
    #     ])),
    #     batch_size=args.batch_size, shuffle=False,
    #     num_workers=args.workers, pin_memory=True)~

BATCHSIZE = 8
EMBED_DIM = 300
HIDDEN_DIM = 350
LEARNING_RATE = 1e-4
device = "cpu"

class Data(Dataset):
    """"""
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        ex = self.df.iloc[idx]

        captions = torch.LongTensor(ex.captions)
        ids = torch.LongTensor([ex.id])
        # input_mask = torch.tensor(feats.input_mask, dtype=torch.long)
        image_features = torch.FloatTensor(ex.features)

        return image_features, captions, ids

# trn = pd.read_json("train.json", orient="records", lines=True)
# trn_loader = DataLoader(dataset=Data(trn), batch_size=BATCHSIZE)

# val = pd.read_json("val.json", orient="records", lines=True)
# val_loader = DataLoader(dataset=Data(val), batch_size=BATCHSIZE)

trn = pd.read_json("val.json", orient="records", lines=True)
trn_loader = DataLoader(dataset=Data(trn), batch_size=BATCHSIZE)

# model = BaselineModel(EMBED_DIM, HIDDEN_DIM, image_dim=512, batch_size=BATCHSIZE, vocab_size=1004, start_token=1, end_token=2, pad_token=0, unk_token=3, max_seq_len=17)
model = TransformerModel(hidden_dim=128, image_dim=512, head_count=8, batch_size=8, vocab_size=1004, start_token=1, end_token=2, pad_token=0, unk_token=3, max_seq_len=16, n_layer=4)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

model.train()
# model.eval()
losses = []
for image_features, captions, ids in trn_loader:
    image_features = image_features.to(device)
    captions = captions.to(device)

    loss = model(image_features, captions=captions)
    # answers = model(image_features)

    loss.backward()
    print(loss.item())
    losses.append(loss.item())
    optimizer.step()
    model.zero_grad()

print("Final Loss")
print(sum(losses)/len(losses))
