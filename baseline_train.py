import time, os, json
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from model import BaselineModel2
import h5py
# from nltk.translate.bleu_score import sentence_bleu
# from nltk.translate.meteor_score import meteor_score

from optim import BertAdam
from bleu_scorer import BleuScorer

import ipdb

# from subprocess import Popen, PIPE, STDOUT

# from coco_caption.pycocotools.coco import COCO
# from coco_caption.pycocoevalcap.eval import COCOEvalCap

# from coco_utils import load_coco_data, sample_coco_minibatch, decode_captions

BATCHSIZE = 128
HIDDEN_DIM = 768
LEARNING_RATE = 1e-4
EPOCH = 10
device = torch.device("cuda")
train_folder_path = "dataset2/train2014/"
val_folder_path = "dataset2/val2014/"
result_file = "baseline_test_val_small_results.json"
model_name = "baseline_model.pt"

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
trn_loader = DataLoader(dataset=Data2(trn), batch_size=BATCHSIZE)

# Load pretrained embeddings
# embeds_file = h5py.File("embeddings.hdf5", "r")
# pre_embeds = np.array(embeds_file["embeds"], dtype=np.float32)
# embeds_file.close()

model = BaselineModel2(HIDDEN_DIM, HIDDEN_DIM, image_dim=2048, batch_size=BATCHSIZE, vocab_size=1004, start_token=1, end_token=2, pad_token=0, unk_token=3, max_seq_len=16, pretrained_embeds=[], device=device)

val = pd.read_json("test_val_small.json", orient="records", lines=True)
val_loader = DataLoader(dataset=Data3(val), batch_size=BATCHSIZE)

# coco_val = COCO("coco-caption/annotations/captions_val_small2014.json")

# model.load_state_dict(torch.load(model_name))
model.to(device)

# model = torch.nn.DataParallel(model)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


model.train()

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
        loss = model([train_folder_path + filename for filename in filenames], captions=captions)

        loss.backward()
        losses.append(loss.item())
        optimizer.step()
        model.zero_grad()

        if iteration % 300 == 0:
            # all_meteor = 0
            model.eval()
            result_json = []
            for ids in val_loader:

                ids = ids.squeeze(1)
                filenames = [val[val.id == a.item()].iloc[0].filename for a in ids]
                captions = [val[val.id == a.item()].iloc[0].captions for a in ids]
                pred_captions = model([val_folder_path + filename for filename in filenames])
                pred_sentences = [get_sent_from_vocab(clean_sent(sent)) for sent in pred_captions]
                result_json.extend([{"image_id":image_id.item(),"caption":caption, "true_cap":get_sent_from_vocab(clean_sent(refs[0][1:]))} for image_id, caption, refs in zip(ids, pred_sentences, captions)])

                bleu_scorer = BleuScorer(n=4)
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
            # curr_score = all_meteor / len(val)
            print(curr_score)
            curr_score = sum(curr_score) / 4
            print("Overal score : %.4f" %curr_score)
            print("***")


            if curr_score > best_score:
                best_score = curr_score
                model = model.module if hasattr(model, 'module') else model  # To handle multi gpu
                torch.save(model.state_dict(), model_name)

            model.train()


    print("Epoch " + str(i+1))
    print("Loss : %.4f" %(sum(losses)/len(losses)))


    result_json = []
    model.eval()
    for ids in val_loader:

        ids = ids.squeeze(1)
        filenames = [val[val.id == a.item()].iloc[0].filename for a in ids]
        captions = [val[val.id == a.item()].iloc[0].captions for a in ids]
        pred_captions = model([val_folder_path + filename for filename in filenames])
        pred_sentences = [get_sent_from_vocab(clean_sent(sent)) for sent in pred_captions]
        result_json.extend([{"image_id":image_id.item(),"caption":caption, "true_cap":get_sent_from_vocab(clean_sent(refs[0][1:]))} for image_id, caption, refs in zip(ids, pred_sentences, captions)])

        bleu_scorer = BleuScorer(n=4)
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
    # curr_score = all_meteor / len(val)
    print(curr_score)
    curr_score = sum(curr_score) / 4
    print("Overal score : %.4f" %curr_score)
    print("------------------------------")


    if curr_score > best_score:
        best_score = curr_score
        model = model.module if hasattr(model, 'module') else model  # To handle multi gpu
        torch.save(model.state_dict(), model_name)
