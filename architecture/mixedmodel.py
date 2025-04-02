import torch.nn.functional as F
import torch
from torch import nn

class BERTConcat(nn.Module):
    def __init__(self, in_size, hid_size):
        super(BERTConcat, self).__init__()
        # Bert with last custom layer trainable
        self.bert = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-uncased')
        for param in self.bert.parameters():
            param.requires_grad = False
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hid_size,
            batch_first=True
            # proj_size=1,
        )
        self.proj = nn.Linear(in_features=hid_size+768, out_features=1)

    def forward(self, x, tokens, mask):
        x = x.unsqueeze(-1)
        y_bert = self.bert(tokens, attention_mask=mask)
        y_bert = y_bert.last_hidden_state[:,0,:]
        y_lstm, _ = self.lstm(x)
        y_lstm = y_lstm[:,-1,:]
        y_multi = torch.cat((y_lstm, y_bert), dim=1)
        y_multi = F.dropout(y_multi, p=0.2)
        y_pred = self.proj(y_multi)
        return y_pred