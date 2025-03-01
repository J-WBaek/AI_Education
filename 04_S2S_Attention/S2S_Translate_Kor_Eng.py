import torch 
import torch.nn as nn
import numpy as np
import random
from  sentance_preprocess import read_language

SOS_token = 0
EOS_token = 1
MAX_LENGTH = 3000
device = "cuda" if torch.cuda.is_available() else "cpu"

lang_input, lang_output, pairs = read_language('ENG', 'KOR', reverse=False, verbose=False)
for idx in range(10):
    print(random.choice(pairs))

class Encoder(nn.Module):
    def __init__(self,
                moduleSelect = int,
                input_dims : int,
                dmodel : int,
                hidden : float,
                dropout = 0.1,
                device = 'cuda'
                ):
        super().__init__()

        self.hidden_size = hidden
        self.dmodel = dmodel
        # input_dims : ascii
        # 
        self.embedding = nn.Embedding(input_dims, dmodel).to(device)

        if moduleSelect == 1:
            self.module = nn.RNN(dmodel, hidden_size, )
        elif moduleSelect == 2:
            self.module = nn.LSTM(dmodel, hidden_size, )
        elif moduleSelect == 3:
            self.module = nn.GRU(dmodel, hidden_size, )
    
    def forward(self, input):
        embedded  = self.embedding(input)
        hidden_ = []
        if moduleSelect == 2:
            output, hidden, _ = self.module(embedded, hidden_)
        else
            output, hidden,   = self.module(embedded, hidden_)
        return output, hidden

class DecoderAttention(nn.Module):
    def __init__(self,
                moduleSelect = int,
#                input_dims : int,
                output_dims : int,
                sentance_len : int,
                hidden : float,
                dropout = 0.1,
                device = 'cuda'
                ):
        super().__init__()

        self.hidden_size = hidden
        self.embedding = nn.Embedding(output_dims, hidden_size).to(device)
        self.out = nn.Embedding(hidden_size, output_dims).to(device)


        if moduleSelect == 1:
            self.module = nn.RNN(input_dims, hidden_size, )
        elif moduleSelect == 2:
            self.module = nn.LSTM(input_dims, hidden_size, )
        elif moduleSelect == 3:
            self.module = nn.GRU(input_dims, hidden_size, )

        self.outputs = nn.Linear(hidden_size, output_size)
    

    
    def forward(self, input_context, input_hidden = None, target_tensor = None): 
        # input_context: encoder_output, input_hidden: encoder_hidden, 

        batch_size = input_context.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS_token)
        
        decoder_hidden = input_hidden
        decoder_outputs = []
        decoder_output, decoder_hidden = self.module(decoder_input, )
        decoder_outputs.append(decoder_output)
        
        return output, hidden
        