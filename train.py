import time, os, json
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from model import TransformerModel2
import h5py

# NOTE : Turns out nltk scorers and COCO evaluations are completely different from each other.
# from nltk.translate.bleu_score import sentence_bleu
# from nltk.translate.meteor_score import meteor_score

from optim import BertAdam
import torchvision.transforms as transforms
from PIL import Image
from bleu_scorer import BleuScorer

import ipdb

# from subprocess import Popen, PIPE, STDOUT

# from coco_caption.pycocotools.coco import COCO
# from coco_caption.pycocoevalcap.eval import COCOEvalCap

# from coco_utils import load_coco_data, sample_coco_minibatch, decode_captions

MULTI_GPU = True
BATCHSIZE = 256
VAL_BATCHSIZE = 64
HIDDEN_DIM = 768
LEARNING_RATE = 1e-4
EPOCH = 10
MAX_SEQ_LEN = 49
device = torch.device("cuda:0")
train_folder_path = "dataset2/train2014/"
val_folder_path = "dataset2/val2014/"
# result_file = "reg_16_test_val_small_results.json"
# model_name = "reg_16_model.pt"
result_file = "reg_49_test_val_small_results.json"
model_name = "reg_49_model.pt"

if not MULTI_GPU:
    VAL_BATCHSIZE = BATCHSIZE


def get_image(file_name, scaler, totensor, normalize):
    img = Image.open(file_name)
    img = totensor(scaler(img))
    if img.shape != (3, 224, 224):
        if len(img.shape) == 2:
            img = img.unsqueeze(0)

        img = img.repeat_interleave(3, axis=0)

    img = normalize(img).unsqueeze(0)
    return img

class Data2(Dataset):
    """"""
    def __init__(self, df, pad_token=0):
        self.df = df
        self.pad_token = pad_token

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        ex = self.df.iloc[idx]

        captions = torch.LongTensor(ex.captions + [self.pad_token] * (MAX_SEQ_LEN - 16))
        ids = torch.LongTensor([ex.id])

        return captions, ids

class Data3(Dataset):
    """"""
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        ex = self.df.iloc[idx]

        ids = torch.LongTensor([ex.id])

        return ids

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

# NOTE : No need for val_model if using single gpu. In multi-gpu for some reason after forward is done, it throws an error when "gathering" from gpu's.
model = TransformerModel2(hidden_dim=HIDDEN_DIM, feature_dim=2048, head_count=8, batch_size=BATCHSIZE, vocab_size=1004, start_token=1, end_token=2, pad_token=0, unk_token=3, max_seq_len=MAX_SEQ_LEN, n_layer=8, device=device, pretrained_embeds=pre_embeds)
if MULTI_GPU:
    val_model = TransformerModel2(hidden_dim=HIDDEN_DIM, feature_dim=2048, head_count=8, batch_size=VAL_BATCHSIZE, vocab_size=1004, start_token=1, end_token=2, pad_token=0, unk_token=3, max_seq_len=MAX_SEQ_LEN, n_layer=8, device=device, pretrained_embeds=pre_embeds)

val = pd.read_json("test_val_small.json", orient="records", lines=True)
val_loader = DataLoader(dataset=Data3(val), batch_size=VAL_BATCHSIZE)

# coco_val = COCO("coco-caption/annotations/captions_val_small2014.json")

# model.load_state_dict(torch.load(model_name))

if MULTI_GPU:
    val_model.to(device)
    model = torch.nn.DataParallel(model, device_ids=[0,1])

model.to(device)

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

scaler = transforms.Scale((224,224))
totensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])

# NOTE : We use this orders for positional embeddings
orders = torch.cat([torch.arange(MAX_SEQ_LEN, dtype=torch.long, device=device).unsqueeze(0) for _ in range(BATCHSIZE)], dim=0)
if MULTI_GPU:
    val_orders = torch.cat([torch.arange(MAX_SEQ_LEN, dtype=torch.long, device=device).unsqueeze(0) for _ in range(VAL_BATCHSIZE)], dim=0)

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

        if MULTI_GPU:
            loss = loss.mean()

        loss.backward()
        losses.append(loss.item())
        optimizer.step()
        model.zero_grad()

        # Give bleu on val every 100 iteration
        if iteration % 100 == 0:
            if MULTI_GPU:
                val_model.load_state_dict(model.module.state_dict())
                val_model.eval()
            else:
                model.eval()

            result_json = []
            bleu_scorer = BleuScorer(n=4)
            for ids in val_loader:

                ids = ids.squeeze(1)
                filenames = [val[val.id == a.item()].iloc[0].filename for a in ids]
                captions = [val[val.id == a.item()].iloc[0].captions for a in ids]
                filenames = [val_folder_path + filename for filename in filenames]
                image_features = torch.cat([get_image(filename, scaler, totensor, normalize) for filename in filenames], axis=0).to(device)

                if MULTI_GPU:
                    if len(image_features) == VAL_BATCHSIZE:
                        pred_captions = val_model(image_features, generate=MAX_SEQ_LEN, orders=val_orders)
                    else:
                        pred_captions = val_model(image_features, generate=MAX_SEQ_LEN)

                else:
                    if len(image_features) == BATCHSIZE:
                        pred_captions = model(image_features, generate=MAX_SEQ_LEN, orders=orders)
                    else:
                        pred_captions = model(image_features, generate=MAX_SEQ_LEN)


                pred_sentences = [get_sent_from_vocab(clean_sent(sent)) for sent in pred_captions]
                result_json.extend([{"image_id":image_id,"caption":caption} for image_id, caption in zip([a.item() for a in ids], pred_sentences)])

                for j,refs in enumerate(captions):
                    bleu_scorer += (pred_sentences[j], [get_sent_from_vocab(clean_sent(ref[1:])) for ref in refs])
                    # all_meteor += meteor_score([get_sent_from_vocab(clean_sent(ref[1:])) for ref in refs], pred_sentences[j]) # ref[1:] -> ignore start index

            curr_score, _ = bleu_scorer.compute_score(option='closest', verbose=0)

            with open(result_file, "w") as f:
                json.dump(result_json, f)

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
            print(curr_score)
            curr_score = sum(curr_score) / 4
            print("Overal score : %.4f" %curr_score)
            print("***")


            if curr_score > best_score:
                best_score = curr_score
                model_to_save = model.module if hasattr(model, 'module') else model  # To handle multi gpu
                torch.save(model_to_save.state_dict(), model_name)

            model.train()


    print("Epoch " + str(i+1))
    print("Loss : %.4f" %(sum(losses)/len(losses)))


    bleu_scorer = BleuScorer(n=4)
    result_json = []
    if MULTI_GPU:
        val_model.load_state_dict(model.module.state_dict())
        val_model.eval()
    else:
        model.eval()

    for ids in val_loader:

        ids = ids.squeeze(1)
        filenames = [val[val.id == a.item()].iloc[0].filename for a in ids]
        captions = [val[val.id == a.item()].iloc[0].captions for a in ids]
        filenames = [val_folder_path + filename for filename in filenames]
        image_features = torch.cat([get_image(filename, scaler, totensor, normalize) for filename in filenames], axis=0).to(device)

        if MULTI_GPU:
            if len(image_features) == VAL_BATCHSIZE:
                pred_captions = val_model(image_features, generate=MAX_SEQ_LEN, orders=val_orders)
            else:
                pred_captions = val_model(image_features, generate=MAX_SEQ_LEN)
        else:
            if len(image_features) == BATCHSIZE:
                pred_captions = model(image_features, generate=MAX_SEQ_LEN, orders=orders)
            else:
                pred_captions = model(image_features, generate=MAX_SEQ_LEN)

        pred_sentences = [get_sent_from_vocab(clean_sent(sent)) for sent in pred_captions]
        result_json.extend([{"image_id":image_id,"caption":caption} for image_id, caption in zip([a.item() for a in ids], pred_sentences)])

        for j,refs in enumerate(captions):
            bleu_scorer += (pred_sentences[j], [get_sent_from_vocab(clean_sent(ref[1:])) for ref in refs])
            # all_meteor += meteor_score([get_sent_from_vocab(clean_sent(ref[1:])) for ref in refs], pred_sentences[j]) # ref[1:] -> ignore start index

    curr_score, _ = bleu_scorer.compute_score(option='closest', verbose=0)

    with open(result_file, "w") as f:
        json.dump(result_json, f)


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
    print(curr_score)
    curr_score = sum(curr_score) / 4
    print("Overal score : %.4f" %curr_score)
    print("------------------------------")


    if curr_score > best_score:
        best_score = curr_score
        model = model.module if hasattr(model, 'module') else model  # To handle multi gpu
        torch.save(model.state_dict(), model_name)

