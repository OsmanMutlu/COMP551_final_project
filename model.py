import torch
import torch.nn as nn
import numpy as np

class Model(nn.Module):
    def __init__(self, embed_dim, hidden_dim, image_dim=512, batch_size=8, vocab_size=1004, start_token=1, end_token=2, pad_token=0, unk_token=3, max_seq_len=100):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.hinit = nn.Linear(image_dim, hidden_dim)
        self.cinit = nn.Linear(image_dim, hidden_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.proj = nn.Linear(hidden_dim, vocab_size)
        self.start_token = start_token
        self.end_token = end_token
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_token) # TODO : Does this work ???
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size

    def forward(self, image_features, captions=[]):

        h0 = self.hinit(image_features).unsqueeze(0)
        c0 = self.cinit(image_features).unsqueeze(0)

        B, _ = image_features.shape
        if len(captions) > 0: # In Training
            x = torch.cat((torch.zeros(B, 1, dtype=torch.long) + self.start_token, captions), dim=1)
            y = torch.cat((captions, torch.zeros(B, 1, dtype=torch.long) + self.pad_token), dim=1)

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
