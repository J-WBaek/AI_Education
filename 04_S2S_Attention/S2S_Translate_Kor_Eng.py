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


class Attention(nn.Module):
    def __init__(self, hidden_size)
        super().__init()
        Wa = nn.Linear(hidden_size, hidden_size)
        Ua = nn.Linear(hidden_size, hidden_size)
        Va = nn.Linear(hidden_size, 1)

    def forward():
        score = Wa()
        

class Encoder(nn.Module):
    def __init__(self,
#                moduleSelect = int,
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
        self.module = nn.RNN(dmodel, hidden_size, )
        
    
    def forward(self, input):
        embedded  = self.embedding(input)
        hidden_ = []
        output, hidden, _ = self.module(embedded, hidden_)
        return output, hidden


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.Query = nn.Linear() # Decoder
        self.Key = nn.Linear()   # Encoder
        self.Value = Key

class DecoderAttention(nn.Module):
    def __init__(self,
                output_dims : int,
                hidden_size : int,
                dropout = 0.1,
                max_length = 100,
                device
                ):
        super().__init__()

        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_dims, hidden_size).to(device)

        # self.        

        self.module  = nn.RNN(input_dims, hidden_size)
        self.outputs = nn.Linear(hidden_size, num_vocabs)
        
        
    def forward(self, input_context, target_tensor = None): 
        # input_context: encoder_output, input_hidden: encoder_hidden, 
        # context : (b, input_length, dims)
        batch_size, context_length, context_dims = input_context.size
        decoder_input0 = torch.empty( (batch_size, 1), dtype=torch.long, device=device).fill_(SOS_token)
        decoder_hidden = input_context

        decoder_outputs = []
        attention = []

        for idx in range(context_length):
            
    def forward_1_step(self, input_vec, hidden_vec, encoder_output):
        embedded = self.embedding(input_vec)
        query = hidden.permute(1, 0, 2)

        torch.cat((embedded, context), dim=2)
    
        decoder_output, decoder_hidden = self.module(decoder_input, )


        decoder_outputs.append(decoder_output)
        
        return output, hidden
        


