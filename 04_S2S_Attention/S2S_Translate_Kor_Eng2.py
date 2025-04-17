import torch 
import torch.nn as nn
import numpy as np
import random
from  sentance_preprocess import read_language
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import TensorDataset, DataLoader

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

        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=tokenizer.pad_token_id)
        self.GRU = nn.GRU(embed_size, hidden_size, batch_first=True)

    def forward(self, input):
        embedded  = self.embedding(input)
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
        Attscore = self.softmax(Attscore)

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

        self.hidden_size  = hidden_size
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=tokenizer.pad_token_id)
        self.attention = Attention(hidden_size)

        self.GRU  = nn.GRU(embed_size, hidden_size)
        self.out     = nn.Linear(hidden_size, vocab_size)
        self.max_len = max_length
        
    # def forward(self, encoder_outputs, encoder_hidden, target_tensor = None): 
    #     # input_context: encoder_output, input_hidden: encoder_hidden, 
    #     # context : (b, sentance_length, word_vec)
        
    #     batch_size, context_length, context_dims = input_context.size
    #     decoder_input_current = torch.empty( (batch_size, 1), dtype=torch.long, device=device).fill_(SOS_token)
    #     decoder_hidden = input_context

    #     decoder_outputs = []
    #     attention = []

    #     #for idx in range(context_length):
    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None, teacher_forcing_ratio=0.5):
        """
        encoder_outputs: (batch, src_len, H)
        encoder_hidden:  (1, batch, H)
        target_tensor:   (batch, tgt_len)
        """
        batch_size  = encoder_outputs.size(0)
        tgt_len     = target_tensor.size(1) if target_tensor is not None else self.max_len
        vocab_size  = self.out.out_features

        # 첫 입력은 <sos>
        decoder_input  = torch.full((batch_size,1), SOS_token, dtype=torch.long, device=device)
        decoder_hidden = encoder_hidden
        outputs = torch.zeros(batch_size, tgt_len, vocab_size, device=device)

        for t in range(tgt_len):
            step_logits, decoder_hidden, attn_w = self.forward_1_step(
                decoder_input, decoder_hidden, encoder_outputs
            )
            outputs[:, t, :] = step_logits.squeeze(1)

            # teacher forcing
            if target_tensor is not None and random.random() < teacher_forcing_ratio:
                decoder_input = target_tensor[:, t].unsqueeze(1)    # (batch,1)
            else:
                top1 = step_logits.argmax(2)                        # (batch,1)
                decoder_input = top1

        return outputs

    def forward_1_step(self, input_word, hidden, hidden_from_encoder):
        
        embedded_input_word = self.embedding(input_word) 
        # Query (batch, sentance_len, word_vec) ====> (batch, word_vec, setance_len) 
        # output (batch, sentance_len, word_vec)
        context, attn_weight = self.attention(embedded_input_word, hidden_from_encoder)
        GRU_input = torch.cat((embedded_input_word, context), dim=2) # (batch, 1, en + hid)

        output, hidden1 = self.GRU(GRU_input, hidden)

        logits = self.out(output.squeeze(1)) # (batch, vocab_size)

        decoder_outputs.append(decoder_output)
        return output, hidden


# 5) 모델, 옵티마이저, 손실함수
embed_size  = 256
hidden_size = 256
src_vocab   = tokenizer.vocab_size
tgt_vocab   = tokenizer.vocab_size

encoder = Encoder(src_vocab, embed_size, hidden_size).to(device)
decoder = DecoderAttention(tgt_vocab, embed_size, hidden_size, max_length=target_ids.size(1)).to(device)

optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3)
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

# 6) 학습 루프
num_epochs = 10
for epoch in range(1, num_epochs+1):
    encoder.train(); decoder.train()
    total_loss = 0

    for src_ids, tgt_ids in loader:
        src_ids, tgt_ids = src_ids.to(device), tgt_ids.to(device)

        optimizer.zero_grad()
        enc_outs, enc_hidden = encoder(src_ids)

        # 디코더가 (batch, tgt_len, vocab) 리턴
        dec_outputs = decoder(enc_outs, enc_hidden, target_tensor=tgt_ids)

        # loss: 시퀀스 길이별 합으로 계산
        batch_loss = 0
        for t in range(tgt_ids.size(1)):
            batch_loss += criterion(dec_outputs[:, t, :], tgt_ids[:, t])
        batch_loss = batch_loss / tgt_ids.size(1)

        batch_loss.backward()
        optimizer.step()

        total_loss += batch_loss.item()

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch:2d}  Loss: {avg_loss:.4f}")