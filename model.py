import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertConfig, BertTokenizer

import math
from typing import Dict, Any

class Embedding(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.embed_dim = embed_dim
    
    def forward(self, x):
        embed = self.embedding(x)
        out = embed * math.sqrt(self.embed_dim) # scale embedding matrix
        # 由于embedding matrix的初始化方式是xavier init，这种方式使得方差是1/embed_dim
        # 因此乘以math.sqrt(self.embed_dim)让输出的方差是1，更利于embedding matrix收敛

        return out
    
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, max_len: int, dropout:float, device: torch.device):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # dropout是为了防止对位置编码太敏感

        pe = torch.zeros(max_len, embed_dim, device=device)    # 初始化位置嵌入矩阵
        position = torch.arange(0, max_len, device=device).unsqueeze(1)    # 生成位置下标矩阵
        div_term = torch.exp(torch.arange(0., embed_dim, 2, device=device) * -(math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)   # [1, max_len, embed_dim]

    def forward(self, x):
        x = x + self.pe[:, :x.shape[1], :]
        out = self.dropout(x)
        return out
    
class Generator(nn.Module):
    def __init__(self, embed_dim: int, vocab_size: int):
        super(Generator, self).__init__()
        self.linear = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        linear_out = self.linear(x)
        out = F.log_softmax(linear_out, dim=-1)
        return out
    
class EncoderLayer(nn.Module):
    def __init__(
            self, 
            embed_dim: int, 
            num_heads: int,
            dropout: float,
        ):
        super(EncoderLayer, self).__init__()
        self.self_multihead_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(embed_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(normalized_shape=embed_dim)
        self.norm2 = nn.LayerNorm(normalized_shape=embed_dim)

    def forward(self, x, pad_mask):
        attn_out, _ = self.self_multihead_attn(x, x, x, key_padding_mask=pad_mask)

        out = self.norm1(x + attn_out)
        fc_out = self.fc(out)
        out = self.norm2(out + fc_out)
        return out
    
class Encoder(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            num_layers: int,
            num_heads: int,
            dropout: float,
        ):
        super(Encoder, self).__init__()

        def _get_layer():
            return EncoderLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
        self.layers = nn.ModuleList([_get_layer() for _ in range(num_layers)])
    
    def forward(self, x, pad_mask):
        for layer in self.layers:
            x = layer(x, pad_mask)
        return x
    
class DecoderLayer(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            dropout: float,    
        ):
        super(DecoderLayer, self).__init__()
        self.self_multihead_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_multihead_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(embed_dim, embed_dim),
        )
        self.norm1 = nn.LayerNorm(normalized_shape=embed_dim)
        self.norm2 = nn.LayerNorm(normalized_shape=embed_dim)
        self.norm3 = nn.LayerNorm(normalized_shape=embed_dim)

    def forward(self, x, enc_out, tgt_pad_mask, src_pad_mask, attn_mask):
        attn_out, _ = self.self_multihead_attn(x, x, x, key_padding_mask=tgt_pad_mask, attn_mask=attn_mask)
        out = self.norm1(x + attn_out)
        attn_out, _ = self.cross_multihead_attn(out, enc_out, enc_out, key_padding_mask=src_pad_mask)
        out = self.norm2(out + attn_out)
        fc_out = self.fc(out)
        out = self.norm3(out + fc_out)
        return out
    
class Decoder(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            num_layers: int,
            num_heads: int,
            dropout: float,    
        ):
        super(Decoder, self).__init__()
    
        def _get_layer():
            return DecoderLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
        self.layers = nn.ModuleList([_get_layer() for _ in range(num_layers)])

    def forward(self, x, enc_out, tgt_pad_mask, src_pad_mask, attn_mask):
        for layer in self.layers:
            x = layer(x, enc_out, tgt_pad_mask, src_pad_mask, attn_mask)
        return x

class Transformer(nn.Module):
    def __init__(
            self,
            embedding_enc: nn.Module,
            encoder: nn.Module,
            embedding_dec: nn.Module,
            decoder: nn.Module,
            generator: nn.Module,
            device: torch.device,
            max_length: int,
            # tokenizer_decoder: BertTokenizer,
            # tokenizer_encoder: BertTokenizer,
            bos_id: int,
            eos_id: int,
            beam_size: int,
            top_k: int,
        ):
        super(Transformer, self).__init__()
        self.embedding_enc = embedding_enc
        self.encoder = encoder
        self.embedding_dec = embedding_dec
        self.decoder = decoder
        self.criterion = nn.CrossEntropyLoss()
        self.generator = generator
        self.device = device
        self.max_length = max_length
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.beam_size = beam_size
        self.top_k = top_k

    def teacher_forcing(self, batch_src, batch_tgt):
        src_token_ids = batch_src['input_ids']
        tgt_token_ids = batch_tgt['input_ids']
        src_pad_mask = (1 - batch_src['attention_mask']).bool()
        tgt_pad_mask = (1 - batch_tgt['attention_mask']).bool()
        dec_len = self.max_length - 1
        attn_mask = (1 - torch.ones(dec_len, dec_len, device=self.device).tril()).bool()

        src_embed = self.embedding_enc(src_token_ids)
        tgt_embed = self.embedding_dec(tgt_token_ids)

        enc_out = self.encoder(x=src_embed, pad_mask=src_pad_mask)

        dec_in = tgt_embed[:, :-1, :]
        dec_out = self.decoder(x=dec_in, enc_out=enc_out, tgt_pad_mask=tgt_pad_mask[:, :-1], src_pad_mask=src_pad_mask, attn_mask=attn_mask)
        
        log_prob = self.generator(dec_out)
        label = tgt_token_ids[:, 1:]
        loss = self.criterion(log_prob.permute(0, 2, 1), label)
        
        return loss

    def greedy_decode(self, batch_src):
        src_token_ids = batch_src['input_ids']
        batch_size = src_token_ids.shape[0]
        src_pad_mask = (1 - batch_src['attention_mask']).bool()

        src_embed = self.embedding_enc(src_token_ids)
        enc_out = self.encoder(x=src_embed, pad_mask=src_pad_mask)

        
        results = [[] for _ in range(batch_size)]
        stop_flag = [False for _ in range(batch_size)]
        count = 0

        tgt_token_ids = torch.full((batch_size, 1), self.bos_id, device=self.device)    # [batch_size, 1]
        for i in range(1, self.max_length):
            tgt_pad_mask = torch.full((batch_size, i), 0, device=self.device).bool()
            attn_mask = torch.zeros(i, i, device=self.device).bool()
            dec_in = self.embedding_dec(tgt_token_ids)
            dec_out = self.decoder(x=dec_in, enc_out=enc_out, tgt_pad_mask=tgt_pad_mask, src_pad_mask=src_pad_mask, attn_mask=attn_mask)
            prob = self.generator(dec_out[:, -1, :])    # [batch_size, vocab_size]

            prob = _top_k_logits(logits=prob, k=self.top_k)

            _, next_token_id = torch.max(prob, dim=-1)  # [batch_size]
            next_token_id_list = next_token_id.tolist()
            tgt_token_ids = torch.cat([tgt_token_ids, next_token_id.unsqueeze(1)], dim=1)
            for j in range(batch_size):
                if stop_flag[j] is False:
                    if next_token_id_list[j] == self.eos_id:
                        count += 1
                        stop_flag[j] = True
                    else:
                        results[j].append(next_token_id_list[j])
                if count == batch_size:
                    return results
        # result = self.tokenizer_zh.decode(results[0], skip_special_tokens=False)
        # print(f'greedy decode:\n{result}')
        return results

    def beam_search_decode(self, batch_src):
        src_token_ids = batch_src['input_ids']
        batch_size = src_token_ids.shape[0]  
        src_pad_mask = (1 - batch_src['attention_mask']).bool()

        src_embed = self.embedding_enc(src_token_ids)
        enc_out = self.encoder(x=src_embed, pad_mask=src_pad_mask)

        enc_out = enc_out.repeat(1, self.beam_size, 1).reshape(batch_size * self.beam_size, self.max_length, -1)
        src_pad_mask = src_pad_mask.repeat(1, self.beam_size).reshape(batch_size * self.beam_size, self.max_length)

        results = [[] for _ in range(batch_size * self.beam_size)]
        stop_flag = [-1 for _ in range(batch_size)]
        count = 0

        tgt_token_ids = torch.full((batch_size * self.beam_size, 1), self.bos_id, device=self.device)    # [batch_size, 1]
        # inputs_zh = torch.full((batch_size * self.beam_size, 1), self.bos_zh, device=self.device)
        scores = torch.full((batch_size * self.beam_size, 1), -1e12, device=self.device)
        scores[::self.beam_size, 0] = 0.0
        for i in range(1, self.max_length):
            tgt_pad_mask = torch.full((batch_size * self.beam_size, i), 0, device=self.device).bool()
            attn_mask = torch.zeros(i, i, device=self.device).bool()
            dec_in = self.embedding_dec(tgt_token_ids)
            # embed_zh = self.embedding_zh(inputs_zh)
            dec_out = self.decoder(x=dec_in, enc_out=enc_out, tgt_pad_mask=tgt_pad_mask, src_pad_mask=src_pad_mask, attn_mask=attn_mask)
            prob = self.generator(dec_out[:, -1, :])

            prob = _top_k_logits(logits=prob, k=self.top_k)

            vocab_size = prob.shape[-1]
            beam_prob = prob + scores
            best_scores, best_scores_id = beam_prob.reshape(batch_size, -1).topk(self.beam_size, 1, True, True)
            scores = best_scores.reshape(-1, 1)
            # scores_id = best_scores_id

            inputs_zh_idx = best_scores_id // vocab_size
            next_token_id = best_scores_id % vocab_size
            next_token_id_list = next_token_id.reshape(-1).tolist()

            term = torch.tensor(range(batch_size), device=self.device).unsqueeze(1) * self.beam_size
            old_idx = (inputs_zh_idx + term).reshape(-1).tolist()
            new_token_id = next_token_id.reshape(-1, 1)
            tgt_token_ids = torch.cat([tgt_token_ids[old_idx], new_token_id], dim=1)

            # ids = []
            for j in range(batch_size):
                if stop_flag[j] == -1:
                    k = j * self.beam_size
                    flag = False
                    for p in range(k, k + self.beam_size):
                        if next_token_id_list[p] == self.eos_id:
                            count += 1
                            stop_flag[j] = p
                            flag = True
                    if flag is False:
                        for p in range(k, k + self.beam_size):
                            results[p].append(next_token_id_list[p])
                if count == batch_size:
                    break
                
        new_results = []
        for i in range(batch_size):
            if stop_flag[i] == -1:
                new_results.append(results[i * self.beam_size])
            else:
                new_results.append(results[stop_flag[i]])

        return new_results

def model_init(config_init: Dict[str, Any]):
    config = config_init
    config_en = BertConfig.from_pretrained(config['checkpoint_en'])
    config_zh = BertConfig.from_pretrained(config['checkpoint_zh'])
    vocab_en = config_en.vocab_size
    vocab_zh = config_zh.vocab_size
    embedd_dim_en = config_en.hidden_size
    embedd_dim_zh = config_zh.hidden_size
    tokenizer = BertTokenizer.from_pretrained(config['checkpoint_zh'])

    embedding_enc = nn.Sequential(
        Embedding(
            vocab_size=vocab_en,
            embed_dim=embedd_dim_en,
        ),
        PositionalEncoding(
            embed_dim=embedd_dim_en,
            max_len=config['max_length'],
            dropout=config['dropout'],
            device=config['device'],
        )
    )
    embedding_dec = nn.Sequential(
        Embedding(
            vocab_size=vocab_zh,
            embed_dim=embedd_dim_zh,
        ),
        PositionalEncoding(
            embed_dim=embedd_dim_zh,
            max_len=config['max_length'],
            dropout=config['dropout'],
            device=config['device'],
        )
    )
    generator = Generator(
        embed_dim=embedd_dim_zh,
        vocab_size=vocab_zh,
    )

    encoder = Encoder(
        embed_dim=embedd_dim_en,
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
    )
    decoder = Decoder(
        embed_dim=embedd_dim_zh,
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
    )

    model = Transformer(
        embedding_enc=embedding_enc,
        encoder=encoder,
        embedding_dec=embedding_dec,
        decoder=decoder,
        generator=generator,
        device=config['device'],  
        max_length=config['max_length'],
        bos_id=tokenizer.cls_token_id,
        eos_id=tokenizer.sep_token_id,
        beam_size=config['beam_size'],
        top_k=config['top_k'],
    )

    return model


def _top_k_logits(logits: torch.Tensor, k: int) -> torch.Tensor:
    if k == 0:
        return logits
    values, _ = torch.topk(logits, dim=1, k=k, sorted=True, largest=True)
    # values.shape: [batch_size, k]
    min_values = values[:, -1].unsqueeze(-1)
    return torch.where(
        logits < min_values,
        torch.full_like(logits, float('-inf')),
        logits
    )
