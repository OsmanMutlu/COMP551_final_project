import torch
import torch.nn as nn
import numpy as np
from torchvision.models import resnet152
import torchvision.transforms as transforms
import copy
from PIL import Image

import ipdb

def get_image(file_name, scaler, totensor, normalize):
    img = Image.open(file_name)
    img = totensor(scaler(img))
    if img.shape != (3, 224, 224):
        if len(img.shape) == 2:
            img = img.unsqueeze(0)

        img = img.repeat_interleave(3, axis=0)

    img = normalize(img).unsqueeze(0)
    return img

class BaselineModel(nn.Module):
    def __init__(self, embed_dim, hidden_dim, image_dim=512, batch_size=8, vocab_size=1004, start_token=1, end_token=2, pad_token=0, unk_token=3, max_seq_len=16):
        super(BaselineModel, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.hinit = nn.Linear(image_dim, hidden_dim)
        self.cinit = nn.Linear(image_dim, hidden_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.proj = nn.Linear(hidden_dim, vocab_size)
        self.start_token = start_token
        self.end_token = end_token
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_token)
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size

    def forward(self, image_features, captions=[]):

        h0 = self.hinit(image_features).unsqueeze(0)
        c0 = self.cinit(image_features).unsqueeze(0)

        B, _ = image_features.shape
        if len(captions) > 0: # In Training
            x = captions[:,:-1] # They already have a start token at the beginning
            y = captions[:,1:]

            assert(x.shape[1] == self.max_seq_len)
            assert(y.shape[1] == self.max_seq_len)

            x = self.embed(x)
            x,_ = self.lstm(x, (h0, c0))
            x = self.proj(x)

            loss = self.criterion(x.view(-1, self.vocab_size), y.view(-1))
            return loss

        else: # In Testing

            x = torch.zeros(B,1, dtype=torch.long) + self.start_token

            out = np.zeros((B,0), dtype=int)
            for _ in range(self.max_seq_len):
                x = self.embed(x)
                _, (h0, c0) = self.lstm(x, (h0, c0))
                x = self.proj(h0.squeeze(0))

                x = torch.argmax(x, axis=1).unsqueeze(1)
                x = x.cpu() # TODO : do we need to do something like x.detach().cpu().numpy() ?
                out = np.append(out, x, axis=1)

            return out

class FeedForward(nn.Module):
    def __init__(self, hidden_dim, ff_hidden_dim, pdrop=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(hidden_dim, ff_hidden_dim)
        self.linear2 = nn.Linear(ff_hidden_dim, hidden_dim)
        self.act_func = nn.ReLU() # TODO : ???
        self.dropout1 = nn.Dropout(p=pdrop)
        self.dropout2 = nn.Dropout(p=pdrop)

    def forward(self, x):
        x = self.act_func(self.dropout1(self.linear1(x)))
        x = self.dropout2(self.linear2(x))
        return x

class MaskedSelfAttention(nn.Module):
    def __init__(self, hidden_dim, head_dim, head_count=8, attn_pdrop=0.1, out_pdrop=0.1):
        super(MaskedSelfAttention, self).__init__()

        self.context = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)

        self.head_dim = head_dim
        self.head_count = head_count

        # TODO : if hidden_dim is too small (i.e 512/16 = 32), head_size will be too small (i.e 32/8 = 4) and maybe we can't get sqrt of it
        assert((hidden_dim / head_count) == head_dim)

        self.softmax = nn.Softmax(dim=2)
        self.attn_dropout = nn.Dropout(p=attn_pdrop)
        self.out_dropout = nn.Dropout(p=out_pdrop)

    def forward(self, features, hiddens, attn_mask): # Features are the same as hiddens when doing self attention
        context = self.context(features) # BxSxD
        key = self.key(features) # BxSxD
        value = self.value(hiddens) # BxSxD

        # Split them to multiple heads
        B, S, D = value.shape
        context = context.view((B, S, self.head_count, self.head_dim)).permute(0,2,1,3) # BxNxSxH
        key = key.view((B, S, self.head_count, self.head_dim)).permute(0,2,1,3) # BxNxSxH
        value = value.view((B, S, self.head_count, self.head_dim)).permute(0,2,1,3) # BxNxSxH

        scores = torch.matmul(context, key.transpose(-1,-2)) # BxNxSxS
        scores = scores / np.sqrt(self.head_dim) # Scale them

        # In the Transformer paper, they don't do masking via the usual multiplying by 0.
        # Instead they do it by adding a large negative number. This works because of the softmax in the next step.

        # In the first attention in the decoder, where we do self attention, for a token we mask the words that succeds it.
        # So a matrix, whose upper half is used for masking. Also the padded tokens are completely masked.
        # In the second attention we only use this masking for padded tokens
        scores = scores + attn_mask.unsqueeze(1)

        attn_probs = self.softmax(scores)
        attn_probs = self.attn_dropout(attn_probs)

        out = torch.matmul(attn_probs, value) # BxNxSxH
        out = out.permute(0,2,1,3).contiguous() # BxSxNxH
        out = out.view(B, S, D) # BxSxD
        out = self.out_dropout(out)

        return out

class DecoderSubLayer(nn.Module):
    def __init__(self, hidden_dim, ff_hidden_dim, head_dim, head_count=8):
        super(DecoderSubLayer, self).__init__()

        # Masked Self Attention
        self.masked_attn = MaskedSelfAttention(hidden_dim, head_dim, head_count=head_count)
        self.norm1 = nn.LayerNorm(hidden_dim)

        # Attention on image features
        self.image_attn = MaskedSelfAttention(hidden_dim, head_dim, head_count=head_count)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Feed Forward
        self.feed_forw = FeedForward(hidden_dim, ff_hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

    def forward(self, features, x, self_attn_mask, padded_mask):
        x = self.masked_attn(x, x, self_attn_mask)
        x = self.norm1(x)

        x = self.image_attn(features, x, padded_mask)
        x = self.norm2(x)

        x = self.feed_forw(x)
        x = self.norm3(x)

        return x

# Coded the decoder ourselves
# For fine details, like where to put dropout, We checked Osman's julia implementation at https://github.com/OsmanMutlu/BERT.jl/blob/master/src/model.jl
class Decoder(nn.Module):
    def __init__(self, hidden_dim, ff_hidden_dim, head_dim, head_count=8, n_layer=4, max_seq_len=16):
        super(Decoder, self).__init__()

        layer = DecoderSubLayer(hidden_dim, ff_hidden_dim, head_dim, head_count=head_count)

        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layer)])
        assert((hidden_dim / head_count) == head_dim)

    def forward(self, features, x, self_attn_mask, padded_mask):
        for layer in self.layers:
            x = layer(features, x, self_attn_mask, padded_mask)

        return x

class TransformerModel(nn.Module):
    def __init__(self, hidden_dim=128, feature_dim=2048, head_count=8, batch_size=8, vocab_size=1004, start_token=1, end_token=2, pad_token=0, unk_token=3, max_seq_len=16, n_layer=4, e_pdrop=0.1, device="cuda"):
        super(TransformerModel, self).__init__()

        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embed = nn.Embedding(max_seq_len, hidden_dim)
        self.embed_dropout = nn.Dropout(p=e_pdrop)

        assert(feature_dim % max_seq_len == 0)
        # self.encoder =  resnet152(pretrained=True) # returns BxD -> D is image_dim -> 2048
        # self.encoder = nn.Sequential(*list(self.encoder.children())[:-1])
        # self.encode = nn.Linear(image_dim / head_count, hidden_dim) # Make sure outcome is BxSxH

        ff_hidden_dim = hidden_dim * 4
        head_dim = int(hidden_dim / head_count)
        self.decoder = Decoder(hidden_dim, ff_hidden_dim, head_dim, head_count=head_count, n_layer=n_layer, max_seq_len=max_seq_len)

        self.proj = nn.Linear(hidden_dim, vocab_size)
        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_token)

        self.start_token = start_token
        self.end_token = end_token
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.device = device

    # TODO : Need to think about end_token and pad_tokens -> replace all end_tokens with pad_tokens
    def forward(self, image_features, captions=[]):

        # TODO : take the transformermodel2's code here
        y = captions[:,1:]
        captions = captions[:,:-1]

        self_attn_mask = torch.triu(torch.full((self.batch_size, self.max_seq_len, self.max_seq_len), -10000, dtype=torch.long, device=self.device), 1) # BxSxS
        # TODO : Find a nicer way to do this
        padded_mask = torch.zeros((self.batch_size, self.max_seq_len, self.max_seq_len), dtype=torch.long, device=self.device)
        pads = captions == self.pad_token
        padded_mask[pads] = -10000
        padded_mask = padded_mask.transpose(1,2)
        padded_mask[pads] = -10000
        self_attn_mask = self_attn_mask + padded_mask # -10000 and -20000 do not differ when their exp is taken, so no problem here

        captions = self.embed(captions) + self.pos_embed(torch.cat([torch.arange(self.max_seq_len, dtype=torch.long, device=self.device).unsqueeze(0) for _ in range(self.batch_size)], dim=0))
        captions = self.embed_dropout(captions)

        # image = image.permute(0,3,1,2)
        # image_features = self.encoder(image)
        image_features = image_features.reshape(self.batch_size, self.max_seq_len, -1) # BxSxH
        # image_features = self.encode(image_features) # Making sure sizes match -> TODO : consider this

        captions = self.decoder(image_features, captions, self_attn_mask, padded_mask)
        scores = self.proj(captions)
        loss = self.criterion(scores.view(-1, self.vocab_size), y.reshape(-1))
        return loss

        # TODO : add test time


# class Attention(nn.Module):
#     def __init__(self, features_dim, hidden_dim, attn_dim):
#         super(Attention, self).__init__()
#         self.context = nn.Linear(features_dim, attn_dim) # DxA
#         self.hidden = nn.Linear(hidden_dim, attn_dim) # HxA
#         self.attn = nn.Linear(attn_dim, 1) # Ax1

#         self.tanh = nn.Tanh()
#         self.softmax = nn.Softmax(dim=1)

#     def forward(self, image_features, hidden_states): # BxD, BxSxH -> S might be between 1 or 18
#         context = self.context(image_features)
#         hidden = self.hidden(hidden_states)
#         attn_vector = self.attn(self.tanh(context + hidden)) # BxSx1
#         attn_vector = self.softmax(attn_vector)

class TransformerModel2(nn.Module):
    def __init__(self, hidden_dim=512, feature_dim=2048, head_count=8, batch_size=8, vocab_size=1004, start_token=1, end_token=2, pad_token=0, unk_token=3, max_seq_len=16, n_layer=4, e_pdrop=0.1, device="cuda", pretrained_embeds=[]):
        super(TransformerModel2, self).__init__()

        self.embed = nn.Embedding(vocab_size, hidden_dim)

        # Load pretrained embeddings
        if len(pretrained_embeds) > 0:
            self.embed.weight = nn.Parameter(torch.from_numpy(pretrained_embeds))

        self.pos_embed = nn.Embedding(max_seq_len, hidden_dim)
        self.embed_dropout = nn.Dropout(p=e_pdrop)

        model =  resnet152(pretrained=True)
        # self.encoder = nn.Sequential(*list(self.encoder.children())[:-2], nn.AdaptiveAvgPool2d((4,4)), nn.Linear(feature_dim, hidden_dim))
        get_features = copy.deepcopy(nn.Sequential(*list(model.children())[:-2]))
        del model
        for param in get_features.parameters():
            param.requires_grad = False

        self.get_features = get_features
        self.scaler = transforms.Scale((224,224))
        self.totensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])


        self.pool = nn.AdaptiveAvgPool2d((4,4))
        self.linear = nn.Linear(feature_dim, hidden_dim)
        # self.encode = nn.Linear(image_dim / head_count, hidden_dim) # Make sure outcome is BxSxH

        ff_hidden_dim = hidden_dim * 4
        head_dim = int(hidden_dim / head_count)
        self.decoder = Decoder(hidden_dim, ff_hidden_dim, head_dim, head_count=head_count, n_layer=n_layer, max_seq_len=max_seq_len)

        self.proj = nn.Linear(hidden_dim, vocab_size)
        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_token)

        self.start_token = start_token
        self.end_token = end_token
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.device = device

    # TODO : Need to think about end_token and pad_tokens -> replace all end_tokens with pad_tokens -> We don't actually need to do this?
    def forward(self, filenames, captions=[], orders=[], generate=1):

        if len(orders) == 0:
            orders = torch.cat([torch.arange(self.max_seq_len, dtype=torch.long, device=self.device).unsqueeze(0) for _ in range(len(filenames))], dim=0)

        # TODO : Moving to device might be bad since this happens multiple times when using multi-GPU. Might consider moving these outside to train.py

        # image_features = torch.cat([get_image(filename, self.scaler, self.totensor, self.normalize) for filename in filenames], axis=0).to(self.device)
        image_features = self.get_features(filenames)

        # image = image.permute(0,3,1,2)
        image_features = self.pool(image_features) # Bx2048x4x4
        image_features = self.linear(image_features.permute(0,2,3,1)) # Bx4x4xH
        # TODO : Trying 2d attention
        image_features = image_features.reshape(len(filenames), self.max_seq_len, -1) # BxSxH
        # image_features = self.encode(image_features) # Making sure sizes match -> TODO : consider this

        if generate == 1: # In training
            y = captions[:,1:]
            x = captions[:,:-1]
        else: # In testing
            captions = torch.full((len(filenames), self.max_seq_len), self.pad_token, dtype=torch.long, device=self.device)
            captions[:,0] = self.start_token
            x = copy.copy(captions)
            predictions = np.full((len(filenames), self.max_seq_len), self.pad_token)

        for i in range(generate): # When training only traversed once
            self_attn_mask = torch.triu(torch.full((len(filenames), self.max_seq_len, self.max_seq_len), -10000, dtype=torch.long, device=self.device), 1) # BxSxS
            # TODO : Find a nicer way to do this
            padded_mask = torch.zeros((len(filenames), self.max_seq_len, self.max_seq_len), dtype=torch.long, device=self.device)
            pads = x == self.pad_token
            padded_mask[pads] = -10000
            padded_mask = padded_mask.transpose(1,2)
            padded_mask[pads] = -10000
            self_attn_mask = self_attn_mask + padded_mask # -10000 and -20000 do not differ when their exp is taken, so no problem here

            x = self.embed(x) + self.pos_embed(orders)
            x = self.embed_dropout(x)

            x = self.decoder(image_features, x, self_attn_mask, padded_mask)
            x = self.proj(x) # scores

            if generate > 1: # In testing
                curr_preds = x.argmax(dim=2)[:,i]
                if i < self.max_seq_len - 1:
                    captions[:,i+1] = curr_preds # next input is this timestep's prediction
                    x = copy.copy(captions)

                predictions[:,i] = curr_preds.cpu().numpy()

        if generate == 1: # In training
            loss = self.criterion(x.view(-1, self.vocab_size), y.reshape(-1))
            return loss
        else: # In testing
            return predictions # predictions
