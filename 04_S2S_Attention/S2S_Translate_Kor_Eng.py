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
#                moduleSelect = int,
                input_dims : int,
                dmodel : int,
                hidden : float,
                dropout = 0.1,
                device = 'cuda'
                ):
        super().__init__()

        self.hidden_size = hidden
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
        self.hidden_size = hidden_size
        self.softmax = nn.Softmax(2)
        # decoder input으로는 문자가 들어오니, hidden size (wordvec 을 알고있어야 함)

    def forward(self, input, hidden): 
        # input: decoder 
        # hidden: encoder
        key = hidden
        query = input
        value = hidden

        # query          (batch, 1, wodvec)
        # key            (batch, sentance(E), wordvec)
        # key.permute    (batch, wodvec, sentance(E))
        # Attscore       (batch, 1, sentance(E))
        Attscore =  torch.bmm(query, key.permute(0, 2, 1))

        # softmax,
        Attscore = softmax(Attscore)

        # Attscore (batch, 1, sentance(E))
        # value    (batch, sentance(E), wordvec)
        # cross_att (batch, 1, wordvec)
        cross_atention = torch.bmm(Attscore, value)

        return cross_atention


class DecoderAttention(nn.Module):
    def __init__(self,
#                input_dims : int,    # hidden
#                output_dims : int,   # 
                hidden_size : int,   #
                dropout = 0.1,
                max_length = 100,
                deviceencoder_output
                ):
        super().__init__()

        self.hidden_size = hidden_size
        self.embedding_0  = nn.Embedding(input_dims, hidden_size).to(device)
        self.embedding_else = nn.Embedding(2*hidden_size, hidden_size).to(device)
        
        self.attention = Attention()

        self.RNN  = nn.RNN(input_dims, hidden_size)
     
        
    def forward(self, input_context, target_tensor = None): 
        # input_context: encoder_output, input_hidden: encoder_hidden, 
        # context : (b, sentance_length, word_vec)
        batch_size, context_length, context_dims = input_context.size
        decoder_input_current = torch.empty( (batch_size, 1), dtype=torch.long, device=device).fill_(SOS_token)
        decoder_hidden = input_context

        decoder_outputs = []
        attention = []

        for idx in range(context_length):
  
    def forward_1_step(self, input_word, hidden_from_encoder):
        embedded_input_word = self.embedding_in(input_word)
        output, hidden = self.module(embedded_input_word, hidden_from_encoder)
        # Query (batch, sentance_len, word_vec) ====> (batch, word_vec, setance_len) 
        # output (batch, sentance_len, word_vec)
        Query = hidden_from_encoder.perjmute(0, 2, 1)
        nn.matmul(Query, output)
        
        key = context_h
        weight = nn.matmul(query, context_h)

        torch.cat((embedded, context), dim=2) # 
        decoder_output, decoder_hidden = self.module(decoder_input, )

        decoder_outputs.append(decoder_output)
        return output, hidden
