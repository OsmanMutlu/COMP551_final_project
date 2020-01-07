import time, os, json
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from model import TransformerModel, TransformerModel2
import h5py
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score

from optim import BertAdam
import torchvision.transforms as transforms
from PIL import Image

import ipdb

# from subprocess import Popen, PIPE, STDOUT

# from coco_caption.pycocotools.coco import COCO
# from coco_caption.pycocoevalcap.eval import COCOEvalCap

# from coco_utils import load_coco_data, sample_coco_minibatch, decode_captions

# # Load COCO data from disk; this returns a dictionary
# # We'll work with dimensionality-reduced features for this notebook, but feel
# # free to experiment with the original features by changing the flag below.
# data = load_coco_data(pca_features=True)


# trn = pd.DataFrame([{"id":data["train_image_idxs"][i], "captions":data["train_captions"][i,:], "features":data["train_features"][data["train_image_idxs"][i],:]} for i in range(len(data["train_image_idxs"]))])
# val = pd.DataFrame([{"id":data["val_image_idxs"][i], "captions":data["val_captions"][i,:], "features":data["val_features"][data["val_image_idxs"][i],:]} for i in range(len(data["val_image_idxs"]))])

# trn.to_json("train.json", orient="records", lines=True, force_ascii=False)
# val.to_json("val.json", orient="records", lines=True, force_ascii=False)


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

BATCHSIZE = 512
HIDDEN_DIM = 768
LEARNING_RATE = 1e-4
EPOCH = 30
MAX_SEQ_LEN = 16
device = torch.device("cuda")
train_folder_path = "dataset2/train2014/"
val_folder_path = "dataset2/val2014/"
# result_file = "test_val_small_results.json"

# class Data(Dataset):
#     """"""
#     def __init__(self, df):
#         self.df = df

#     def __len__(self):
#         return len(self.df)

#     def __getitem__(self, idx):
#         ex = self.df.iloc[idx]

#         captions = torch.LongTensor(ex.captions)
#         ids = torch.LongTensor([ex.id])
#         # input_mask = torch.tensor(feats.input_mask, dtype=torch.long)
#         features = torch.FloatTensor(ex.features)

#         return image, captions, ids

def get_image(file_name, scaler, totensor, normalize):
    img = Image.open(file_name)
    img = totensor(scaler(img))
    if img.shape != (3, 224, 224):
        if len(img.shape) == 2:
            img = img.unsqueeze(0)

        img = img.repeat_interleave(3, axis=0)

    img = normalize(img).unsqueeze(0)
    return img

class Data1(Dataset):
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
        features1 = torch.FloatTensor(ex.features)

        return features1, captions, ids

class Data2(Dataset):
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

        return captions, ids

class Data3(Dataset):
    """"""
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        ex = self.df.iloc[idx]

        # ids = torch.LongTensor([ex.id])
        ids = torch.LongTensor([ex.id])
        # input_mask = torch.tensor(feats.input_mask, dtype=torch.long)

        return ids

# class Data(Dataset):
#     """"""
#     def __init__(self, filename):
#         self.filename = filename
#         self.f = open(filename, "r")

#     def __len__(self):
#         if "val" in self.filename:
#             return 12312321
#         elif "train" in self.filename:
#             return 123123

#     def __getitem__(self, idx):
#         line = self.f.readline()
#         assert(line != "")

#         ex = json.loads(line)

#         captions = torch.LongTensor(ex.captions)
#         ids = torch.LongTensor([ex.id])
#         # input_mask = torch.tensor(feats.input_mask, dtype=torch.long)
#         image = torch.FloatTensor(np.array(ex.image))

#         return image, captions, ids


# trn = pd.read_json("train.json", orient="records", lines=True)
# trn_loader = DataLoader(dataset=Data(trn), batch_size=BATCHSIZE)

# val = pd.read_json("val.json", orient="records", lines=True)
# val_loader = DataLoader(dataset=Data(val), batch_size=BATCHSIZE)


# trn = pd.read_json("new_train.json", orient="records", lines=True)
# hd5 = h5py.File("train_features1.hdf5", "r")
# trn = trn.merge(pd.DataFrame({"id":hd5["ids"], "features":hd5["features"]}), on="id")
# trn_loader = DataLoader(dataset=Data1(trn), batch_size=BATCHSIZE)

# model = TransformerModel(hidden_dim=128, feature_dim=2048, head_count=8, batch_size=8, vocab_size=1004, start_token=1, end_token=2, pad_token=0, unk_token=3, max_seq_len=MAX_SEQ_LEN, n_layer=4, device=device)



# TODO : get test
# TODO : Maybe divide val further regarding 2017 split

# Load Vocab
vocab = json.load(open("/home/omutlu/comp551_final_project/dataset/coco_captioning/coco2014_vocab.json", "r"))["idx_to_word"]

def get_sent_from_vocab(sentence):
    return " ".join([vocab[word] for word in sentence])

def clean_sent(sent, pad_index=0, end_index=2, start_index=1):
    sent = np.array(sent)
    sent = sent[sent != pad_index]
    sent = sent[sent != end_index]
    return sent

trn = pd.read_json("new_train.json", orient="records", lines=True)
trn_loader = DataLoader(dataset=Data2(trn), batch_size=BATCHSIZE, drop_last=True)

# Load pretrained embeddings
embeds_file = h5py.File("embeddings.hdf5", "r")
pre_embeds = np.array(embeds_file["embeds"], dtype=np.float32)
embeds_file.close()

model = TransformerModel2(hidden_dim=HIDDEN_DIM, feature_dim=2048, head_count=8, batch_size=BATCHSIZE, vocab_size=1004, start_token=1, end_token=2, pad_token=0, unk_token=3, max_seq_len=MAX_SEQ_LEN, n_layer=4, device=device, pretrained_embeds=pre_embeds)

val = pd.read_json("test_val_small.json", orient="records", lines=True)
val_loader = DataLoader(dataset=Data3(val), batch_size=BATCHSIZE)

# coco_val = COCO("coco-caption/annotations/captions_val_small2014.json")

# model.load_state_dict(torch.load("model.pt"))
model.to(device)

model = torch.nn.DataParallel(model)

# optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

num_train_steps = len(trn) / BATCHSIZE * EPOCH

# This optimizer code is taken, BertAdam included, from https://github.com/huggingface/transformers
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'gamma', 'beta']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
]
optimizer = BertAdam(optimizer_grouped_parameters,
                     lr=LEARNING_RATE,
                     warmup=0.1,
                     t_total=num_train_steps)

model.train()

# losses = []
# for features, captions, ids in trn_loader:
#     features = features.to(device)
#     captions = captions.to(device)

#     loss = model(features, captions=captions)
#     # answers = model(image_features)

#     loss.backward()
#     print(loss.item())
#     losses.append(loss.item())
#     optimizer.step()
#     model.zero_grad()

scaler = transforms.Scale((224,224))
totensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])

orders = torch.cat([torch.arange(MAX_SEQ_LEN, dtype=torch.long, device=device).unsqueeze(0) for _ in range(BATCHSIZE)], dim=0)

print("Started Training") # TODO : tqdm here
best_score = 0.0
for i in range(EPOCH):
    iteration = 1
    losses = []
    model.train()
    for captions, ids in trn_loader:
        iteration += 1
        captions = captions.to(device)

        ids = ids.squeeze(1)
        filenames = [trn[trn.id == a.item()].iloc[0].filename for a in ids]
        filenames = [train_folder_path + filename for filename in filenames]
        image_features = torch.cat([get_image(filename, scaler, totensor, normalize) for filename in filenames], axis=0).to(device)
        loss = model(image_features, captions=captions, orders=orders)

        loss.mean().backward()
        losses.append(loss.mean().item())
        optimizer.step()
        model.zero_grad()

        if iteration % 2 == 0:
            all_meteor = 0
            model.eval()
            # result_json = []
            for ids in val_loader:

                ids = ids.squeeze(1)
                filenames = [val[val.id == a.item()].iloc[0].filename for a in ids]
                captions = [val[val.id == a.item()].iloc[0].captions for a in ids]
                filenames = [val_folder_path + filename for filename in filenames]
                ipdb.set_trace()
                image_features = torch.cat([get_image(filename, scaler, totensor, normalize) for filename in filenames], axis=0).to(device)
                if len(image_features) == BATCHSIZE:
                    pred_captions = model(image_features, generate=MAX_SEQ_LEN, orders=orders)
                else:
                    pred_captions = model(image_features, generate=MAX_SEQ_LEN)

                pred_sentences = [get_sent_from_vocab(clean_sent(sent)) for sent in pred_captions]
                # result_json.extend([{"image_id":image_id,"caption":caption} for image_id, caption in zip(ids, pred_sentences)])

                for j,refs in enumerate(captions):
                    all_meteor += meteor_score([get_sent_from_vocab(clean_sent(ref[1:])) for ref in refs], pred_sentences[j]) # ref[1:] -> ignore start index

            # with open(result_file, "w") as f:
            #     json.dump(result_json, f)

            # coco_val_res = coco.loadRes(result_file)
            # coco_eval = COCOEvalCap(coco_val, coco_val_res)
            # coco_eval.params['image_id'] = cocoRes.getImgIds() # since we use subset of val
            # coco_eval.evaluate()
            # coco_eval.eval.items()

            # p = Popen("python2 eval_coco.py", shell=True, stdout=PIPE, stderr=STDOUT)
            # p.wait()
            # result_dict = p.stdout.read()

            # curr_scores = []
            # for metric, score in result_dict.items():
            #     curr_scores.append(score)
            #     print('%s: %.3f'%(metric, score))

            # curr_score = sum(curr_scores)/len(curr_scores)
            curr_score = all_meteor / len(val)
            print("Overal score : %.4f" %curr_score)
            print("***")


            if curr_score > best_score:
                best_score = curr_score
                model = model.module if hasattr(model, 'module') else model  # To handle multi gpu
                torch.save(model.state_dict(), "model2.pt")


    print("Epoch " + str(i+1))
    print("Loss : %.4f" %(sum(losses)/len(losses)))


    all_meteor = 0
    # result_json = []
    model.eval()
    for ids in val_loader:

        ids = ids.squeeze(1)
        filenames = [val[val.id == a.item()].iloc[0].filename for a in ids]
        captions = [val[val.id == a.item()].iloc[0].captions for a in ids]
        filenames = [val_folder_path + filename for filename in filenames]
        image_features = torch.cat([get_image(filename, scaler, totensor, normalize) for filename in filenames], axis=0).to(device)
        if len(image_features) == BATCHSIZE:
            pred_captions = model(image_features, generate=MAX_SEQ_LEN, orders=orders)
        else:
            pred_captions = model(image_features, generate=MAX_SEQ_LEN)

        pred_sentences = [get_sent_from_vocab(clean_sent(sent)) for sent in pred_captions]
        # result_json.extend([{"image_id":image_id,"caption":caption} for image_id, caption in zip(ids, pred_sentences)])

        for j,refs in enumerate(captions):
            all_meteor += meteor_score([get_sent_from_vocab(clean_sent(ref[1:])) for ref in refs], pred_sentences[j]) # ref[1:] -> ignore start index

    # with open(result_file, "w") as f:
    #     json.dump(result_json, f)


    # coco_val_res = coco.loadRes(result_file)
    # coco_eval = COCOEvalCap(coco_val, coco_val_res)
    # coco_eval.params['image_id'] = cocoRes.getImgIds() # since we use subset of val
    # coco_eval.evaluate()
    # coco_eval.eval.items()

    # curr_scores = []
    # for metric, score in result_dict.items():
    #     curr_scores.append(score)
    #     print('%s: %.3f' %(metric, score))

    # curr_score = sum(curr_scores)/len(curr_scores)
    curr_score = all_meteor / len(val)
    print("Overal score : %.4f" %curr_score)
    print("------------------------------")


    if curr_score > best_score:
        best_score = curr_score
        model = model.module if hasattr(model, 'module') else model  # To handle multi gpu
        torch.save(model.state_dict(), "model2.pt")

    # all_bleu = 0
    # model.eval()
    # for ids in val_loader:

    #     ids = ids.squeeze(1)
    #     filenames = [val[val.id == a.item()].iloc[0].filename for a in ids]
    #     captions = [val[val.id == a.item()].iloc[0].captions for a in ids]
    #     pred_captions = model([val_folder_path + filename for filename in filenames])
    #     pred_sentences = [get_sent_from_vocab(clean_sent(sent)) for sent in pred_captions]

    #     # TODO : use vocab to get sentences as string
    #     for j,refs in enumerate(captions):
    #         all_bleu += sentence_bleu([get_sent_from_vocab(clean_sent(ref[1:])) for ref in refs], pred_sentences[j]) # ref[1:] -> ignore start index

    # curr_bleu = all_bleu / len(val)
    # print("Bleu score : %.4f" %curr_bleu)


    # if curr_bleu > best_bleu:
    #     best_bleu = curr_bleu
    #     model = model.module if hasattr(model, 'module') else model  # To handle multi gpu
    #     torch.save(model.state_dict(), "model.pt")
