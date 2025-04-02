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


class BERTAttn(nn.Module):
    def __init__(self, in_size, hid_size):
        super(BERTAttn, self).__init__()
        # Bert with last custom layer trainable
        self.bert = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-uncased')
        for param in self.bert.parameters():
            param.requires_grad = False
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hid_size,
            batch_first=True
        )
        # ATENTION
        self.W = nn.Parameter(
            torch.randn(size = (768, hid_size)), requires_grad=True
        )
        self.ln = nn.LayerNorm(normalized_shape=hid_size)
        self.proj = nn.Linear(in_features=2*hid_size, out_features=1)

    def forward(self, x, tokens, mask):
        x = x.unsqueeze(-1)
        y_bert = self.bert(tokens, attention_mask=mask)
        y_bert = y_bert.last_hidden_state
        y_bert = torch.matmul(y_bert, self.W)
        # V = torch.matmul(y_bert, self.WV) 
        y_lstm, _ = self.lstm(x)
        y_lstm = y_lstm[:,-1,:].unsqueeze(1)
        att = torch.matmul(y_bert, torch.permute(y_lstm, (0, 2, 1)))
        mask = mask.unsqueeze(-1)
        att = att*mask
        att[att == 0] = float('-inf')
        att = torch.nn.functional.softmax(att, dim=1)
        context = torch.matmul(torch.permute(att, (0, 2, 1)), y_bert)
        context = self.ln(context)
        y_lstm = self.ln(y_lstm)
        out = torch.cat((y_lstm, context), dim=2).squeeze(1)
        out = self.proj(out)
        return out

class BERTAttn1(nn.Module):
    def __init__(self, in_size, hid_size):
        super(BERTAttn1, self).__init__()
        # Bert with last custom layer trainable
        self.bert = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-uncased')
        for param in self.bert.parameters():
            param.requires_grad = False
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hid_size,
            batch_first=True
        )
        # ATENTION
        self.W = nn.Parameter(
            torch.randn(size = (768, hid_size)), requires_grad=True
        )
        self.ln = nn.LayerNorm(normalized_shape=hid_size)
        self.proj = nn.Linear(in_features=hid_size, out_features=1)

    def forward(self, x, tokens, mask):
        x = x.unsqueeze(-1)
        y_bert = self.bert(tokens, attention_mask=mask)
        y_bert = y_bert.last_hidden_state
        y_bert = torch.matmul(y_bert, self.W)
        # V = torch.matmul(y_bert, self.WV) 
        y_lstm, _ = self.lstm(x)
        y_lstm = y_lstm[:,-1,:].unsqueeze(1)
        att = torch.matmul(y_bert, torch.permute(y_lstm, (0, 2, 1)))
        mask = mask.unsqueeze(-1)
        att = att*mask
        att[att == 0] = float('-inf')
        att = torch.nn.functional.softmax(att, dim=1)
        context = torch.matmul(torch.permute(att, (0, 2, 1)), y_bert)
        context = self.ln(context)
        y_lstm = self.ln(y_lstm)
        out = y_lstm + context
        # out = torch.cat((y_lstm, context), dim=2).squeeze(1)
        out = self.proj(out.squeeze(1))
        return out