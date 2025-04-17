import torch 
import torch.nn as nn
import numpy as np
import random
from  sentance_preprocess import read_language
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModel

SOS_token = 0
EOS_token = 1
# MAX_LENGTH = 3000
device = "cuda" if torch.cuda.is_available() else "cpu"

lang_input, lang_output, pairs = read_language('ENG', 'KOR', reverse=False, verbose=False)
for idx in range(10):
    print(random.choice(pairs))

tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ko-en")
encoded_input = tokenizer(lang_input, padding=True, truncation=True, return_tensors="pt")
decoded_input = tokenizer(lang_output, padding=True, truncation=True, return_tensors="pt")

input_ids     = encoded_input["input_ids"].to(device)      # (N, src_len)
target_ids    = decoded_input["input_ids"].to(device)      # (N, tgt_len)
dataset = TensorDataset(input_ids, target_ids)
loader  = DataLoader(dataset, batch_size=32, shuffle=True)

class Encoder(nn.Module):
    def __init__(self,
#                moduleSelect = int,
                vocab_size : int,
                embed_size : int,
                hidden_size ,
                dropout = 0.1,
                ):
        super().__init__()

        self.hidden_size = hidden
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=tokenizer.pad_token_id)
        self.GRU = nn.GRU(dmodel, hidden_size, batch_first=True)

    def forward(self, input):
        embedded  = self.embedding(input)
        hidden_ = []
        output, hidden = self.GRU(embedded)
        return output, hidden

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.softmax = nn.Softmax(2)
        # decoder input으로는 문자가 들어오니, hidden size (wordvec 을 알고있어야 함)

    def forward(self, query, key): 
        # input: decoder 
        # hidden: encoder
        key = key
        query = query
        value = key

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
        context = torch.bmm(Attscore, value)

        return context, Attscore


class DecoderAttention(nn.Module):
    def __init__(self,
                vocab_size : int,
                embed_size : int,
                hidden_size : int,   #
                max_length = 100,
                ):
        super().__init__()

        self.hidden_size    = hidden_size
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=tokenizer.pad_token_id)
        self.attention = Attention(hidden_size, max_length)
        
        self.attention = Attention()

        self.RNN  = nn.RNN(input_dims, hidden_size)
        self.out     = nn.Linear(hidden_size, vocab_size)
        self.max_len = max_len
        
    def forward(self, input_context, target_tensor = None): 
        # input_context: encoder_output, input_hidden: encoder_hidden, 
        # context : (b, sentance_length, word_vec)
        batch_size, context_length, context_dims = input_context.size
        decoder_input_current = torch.empty( (batch_size, 1), dtype=torch.long, device=device).fill_(SOS_token)
        decoder_hidden = input_context

        decoder_outputs = []
        attention = []

        #for idx in range(context_length):
  
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
