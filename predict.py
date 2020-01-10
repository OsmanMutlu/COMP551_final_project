# As usual, a bit of setup
import time, os, json
import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from model import TransformerModel2
import h5py
# from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import torchvision.transforms as transforms
from PIL import Image
from bleu_scorer import BleuScorer

import ipdb

# from coco_caption.pycocotools.coco import COCO
# from coco_caption.pycocoevalcap.eval import COCOEvalCap
# from subprocess import Popen, PIPE, STDOUT

# from coco_utils import load_coco_data, sample_coco_minibatch, decode_captions

BATCHSIZE = 64
HIDDEN_DIM = 768
LEARNING_RATE = 1e-4
EPOCH = 30
MAX_SEQ_LEN = 49
device = torch.device("cuda:1")
train_folder_path = "dataset2/train2014/"
val_folder_path = "dataset2/val2014/"
test_folder_path = "dataset2/test2014/"
result_file = "test_val_small_results.json"
# result_file = "new_test_results.json"
# result_file = "new_val_test_results.json"

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

# TODO : Maybe divide val further regarding 2017 split
# Load Vocab
vocab = json.load(open("/home/omutlu/comp551_final_project/dataset/coco_captioning/coco2014_vocab.json", "r"))["idx_to_word"]

def get_image(file_name, scaler, totensor, normalize):
    img = Image.open(file_name)
    img = totensor(scaler(img))
    if img.shape != (3, 224, 224):
        if len(img.shape) == 2:
            img = img.unsqueeze(0)

        img = img.repeat_interleave(3, axis=0)

    img = normalize(img).unsqueeze(0)
    return img

def get_sent_from_vocab(sentence):
    return " ".join([vocab[word] for word in sentence])

def clean_sent(sent, pad_index=0, end_index=2):
    sent = np.array(sent)
    sent = sent[sent != pad_index]
    sent = sent[sent != end_index]
    return sent

model = TransformerModel2(hidden_dim=HIDDEN_DIM, feature_dim=2048, head_count=8, batch_size=BATCHSIZE, vocab_size=1004, start_token=1, end_token=2, pad_token=0, unk_token=3, max_seq_len=MAX_SEQ_LEN, n_layer=8, device=device)

val = pd.read_json("test_val_small.json", orient="records", lines=True)
# val = pd.read_json("new_val_test.json", orient="records", lines=True)
val_loader = DataLoader(dataset=Data3(val), batch_size=BATCHSIZE)

# coco_val = COCO("coco_caption/annotations/captions_val2014.json")

model.load_state_dict(torch.load("latest_model_49.pt"))
model.to(device)

scaler = transforms.Scale((224,224))
totensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])

orders = torch.cat([torch.arange(MAX_SEQ_LEN, dtype=torch.long, device=device).unsqueeze(0) for _ in range(BATCHSIZE)], dim=0)

bleu_scorer = BleuScorer(n=4)
result_json = []
model.eval()
for ids in val_loader:

    ids = ids.squeeze(1)
    captions = [val[val.id == a.item()].iloc[0].captions for a in ids]
    filenames = [val[val.id == a.item()].iloc[0].filename for a in ids]
    filenames = [val_folder_path + filename for filename in filenames]
    # filenames = [test_folder_path + filename for filename in filenames]
    image_features = torch.cat([get_image(filename, scaler, totensor, normalize) for filename in filenames], axis=0).to(device)
    if len(filenames) == BATCHSIZE:
        pred_captions = model(image_features, orders=orders, generate=MAX_SEQ_LEN)
    else:
        pred_captions = model(image_features, generate=MAX_SEQ_LEN)

    pred_sentences = [get_sent_from_vocab(clean_sent(sent)) for sent in pred_captions]
    result_json.extend([{"image_id":image_id.item(),"caption":caption, "true_cap":get_sent_from_vocab(clean_sent(refs[0][1:]))} for image_id, caption, refs in zip(ids, pred_sentences, captions)])

    for j,refs in enumerate(captions):
        bleu_scorer += (pred_sentences[j], [get_sent_from_vocab(clean_sent(ref[1:])) for ref in refs])

curr_score, _ = bleu_scorer.compute_score(option='closest', verbose=0)

print(curr_score)
print("Bleu : %.4f" %(sum(curr_score) / 4))

with open(result_file, "w") as f:
    json.dump(result_json, f)


# coco_val_res = coco_val.loadRes(result_file)
# coco_eval = COCOEvalCap(coco_val, coco_val_res)
# ipdb.set_trace()
# coco_eval.params['image_id'] = coco_val_res.getImgIds() # since we use subset of val
# coco_eval.evaluate()

# # ipdb.set_trace()

# curr_scores = []
# for metric, score in coco_eval.eval.items():
#     curr_scores.append(score)
#     print('%s: %.3f' %(metric, score))

# curr_score = sum(curr_scores)/len(curr_scores)
# print("Overal score : %.4f" %curr_score)
# print("------------------------------")
