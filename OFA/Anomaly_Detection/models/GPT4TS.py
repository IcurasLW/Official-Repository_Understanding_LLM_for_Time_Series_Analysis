from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from transformers import GPT2ForSequenceClassification
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers import BertTokenizer, BertModel
from einops import rearrange
from layers.Embed import DataEmbedding, DataEmbedding_wo_time
from transformers import RobertaModel, RobertaConfig, RobertaTokenizer, LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, BertModel, BertTokenizer


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class Results_Obj:
    def __init__(self):
        self.last_hidden_state = None

    def set_results(self, results):
        self.last_hidden_state = results


class LinearEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.enccoder = nn.Sequential(nn.LayerNorm(config.d_model), nn.Linear(in_features=config.d_model, out_features=config.d_model))
        self.resutls_obj = Results_Obj()
        
    def forward(self, inputs_embeds):
        out = self.enccoder(inputs_embeds)
        self.resutls_obj.last_hidden_state = out
        return self.resutls_obj


class MHAtt_Backbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.enccoder = nn.Linear(in_features=config.d_model, out_features=config.d_model)
        self.mh_Att = nn.MultiheadAttention(config.d_model, num_heads=4, batch_first=True)
        self.resutls_obj = Results_Obj()
        
    def forward(self, inputs_embeds):
        out = self.enccoder(inputs_embeds)
        out, _ = self.mh_Att(out, out, out)
        self.resutls_obj.last_hidden_state = out
        return self.resutls_obj


class Trans_Backbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.enccoder = nn.Linear(in_features=config.d_model, out_features=config.d_model)
        trans_layer = nn.TransformerEncoderLayer(config.d_model, nhead=4, batch_first=True)
        self.trans_encoder_block = nn.TransformerEncoder(trans_layer, num_layers=1)
        self.resutls_obj = Results_Obj()
        
    def forward(self, inputs_embeds):
        out = self.enccoder(inputs_embeds)
        out = self.trans_encoder_block(out)
        self.resutls_obj.last_hidden_state = out
        return self.resutls_obj


class Model(nn.Module):
    
    def __init__(self, configs):
        super(Model, self).__init__()
        self.is_ln = configs.ln
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.patch_size = configs.patch_size
        self.stride = configs.stride
        self.seq_len = configs.seq_len
        self.d_ff = configs.d_ff
        self.patch_num = (configs.seq_len + self.pred_len - self.patch_size) // self.stride + 1

        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride)) 
        self.patch_num += 1
        self.enc_embedding = DataEmbedding(configs.enc_in * self.patch_size, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.llm = self.get_llm(configs)


        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.predict_linear_pre = nn.Linear(self.seq_len, self.pred_len + self.seq_len)
            self.predict_linear = nn.Linear(self.patch_size, configs.enc_in)
            self.ln = nn.LayerNorm(configs.d_ff)
            self.out_layer = nn.Linear(configs.d_ff, configs.c_out)
        if self.task_name == 'imputation':
            self.ln_proj = nn.LayerNorm(configs.d_model)
            self.out_layer = nn.Linear(
                configs.d_model, 
                configs.c_out, 
                bias=True)
        if self.task_name == 'anomaly_detection':
            self.ln_proj = nn.LayerNorm(configs.d_ff)
            self.out_layer = nn.Linear(
                configs.d_ff, 
                configs.c_out, 
                bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(0.1)
            self.ln_proj = nn.LayerNorm(configs.d_model * self.patch_num)
            self.out_layer = nn.Linear(configs.d_model * self.patch_num, configs.num_class)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(
                x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        B, L, M = x_enc.shape
        # Normalization from Non-stationary Transformer
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc - means
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) /
                           torch.sum(mask == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()
        x_enc /= stdev

        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]

        outputs = self.gpt2(inputs_embeds=enc_out).last_hidden_state
        
        outputs = self.ln_proj(outputs)
        dec_out = self.out_layer(outputs)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        return dec_out

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        B, L, M = x_enc.shape
        
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        enc_out = self.predict_linear_pre(enc_out.permute(0, 2, 1)).permute(
            0, 2, 1)  # align temporal dimension
        enc_out = torch.nn.functional.pad(enc_out, (0, 768-enc_out.shape[-1]))

        # enc_out = rearrange(enc_out, 'b l m -> b m l')
        # enc_out = self.padding_patch_layer(enc_out)
        # enc_out = enc_out.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        # enc_out = self.predict_linear(enc_out)
        # enc_out = rearrange(enc_out, 'b m n p -> b n (m p)')

        dec_out = self.llm(inputs_embeds=enc_out).last_hidden_state
        dec_out = dec_out[:, :, :self.d_ff]
        # dec_out = dec_out.reshape(B, -1)
        
        # dec_out = self.ln(dec_out)
        dec_out = self.out_layer(dec_out)
        # print(dec_out.shape)
        # dec_out = dec_out.reshape(B, self.pred_len + self.seq_len, -1)
        
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        
        return dec_out

    def anomaly_detection(self, x_enc):
        B, L, M = x_enc.shape
        
        # Normalization from Non-stationary Transformer

        seg_num = 25
        x_enc = rearrange(x_enc, 'b (n s) m -> b n s m', s=seg_num)
        means = x_enc.mean(2, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=2, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        x_enc = rearrange(x_enc, 'b n s m -> b (n s) m')

        # means = x_enc.mean(1, keepdim=True).detach()
        # x_enc = x_enc - means
        # stdev = torch.sqrt(
        #     torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        # x_enc /= stdev

        # enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
        outputs = torch.nn.functional.pad(x_enc, (0, 768-x_enc.shape[-1]))
        
        outputs = self.llm(inputs_embeds=outputs).last_hidden_state
        
        outputs = outputs[:, :, :self.d_ff]
        # outputs = self.ln_proj(outputs)
        dec_out = self.out_layer(outputs)

        # De-Normalization from Non-stationary Transformer

        dec_out = rearrange(dec_out, 'b (n s) m -> b n s m', s=seg_num)
        dec_out = dec_out * \
                  (stdev[:, :, 0, :].unsqueeze(2).repeat(
                      1, 1, seg_num, 1))
        dec_out = dec_out + \
                  (means[:, :, 0, :].unsqueeze(2).repeat(
                      1, 1, seg_num, 1))
        dec_out = rearrange(dec_out, 'b n s m -> b (n s) m')

        # dec_out = dec_out * \
        #           (stdev[:, 0, :].unsqueeze(1).repeat(
        #               1, self.pred_len + self.seq_len, 1))
        # dec_out = dec_out + \
        #           (means[:, 0, :].unsqueeze(1).repeat(
        #               1, self.pred_len + self.seq_len, 1))
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # print(x_enc.shape)
        B, L, M = x_enc.shape
        input_x = rearrange(x_enc, 'b l m -> b m l')
        input_x = self.padding_patch_layer(input_x)
        input_x = input_x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        input_x = rearrange(input_x, 'b m n p -> b n (p m)')
        
        outputs = self.enc_embedding(input_x, None)
        
        outputs = self.gpt2(inputs_embeds=outputs).last_hidden_state

        outputs = self.act(outputs).reshape(B, -1)
        outputs = self.ln_proj(outputs)
        # outputs = self.dropout(outputs)
        outputs = self.out_layer(outputs)
        
        return outputs

    def get_llm(self, config):
        if config.LLM == 'LLAMA':
            self.llama_config = LlamaConfig.from_pretrained('huggyllama/llama-7b')
            self.llama_config.num_hidden_layers = 3
            self.llama_config.output_attentions = True
            self.llama_config.output_hidden_states = True
            try:
                self.llm = LlamaModel.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.llama_config,
                    # load_in_4bit=True
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm = LlamaModel.from_pretrained(
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.llama_config,
                    # load_in_4bit=True
                )
                
                
        elif config.LLM == 'GPT2':
            self.gpt2_config = GPT2Config.from_pretrained('gpt2')
            self.gpt2_config.num_hidden_layers = 6
            self.gpt2_config.output_attentions = True
            self.gpt2_config.output_hidden_states = True
            try:
                self.llm = GPT2Model.from_pretrained(
                    'gpt2',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.gpt2_config,
                )
                
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm = GPT2Model.from_pretrained(
                    'gpt2',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.gpt2_config,
                )
                
        
        elif config.LLM == 'Random':
            self.gpt2_config = GPT2Config.from_pretrained('gpt2')
            self.gpt2_config.num_hidden_layers = 6
            self.gpt2_config.output_attentions = True
            self.gpt2_config.output_hidden_states = True
            try:
                self.llm = GPT2Model.from_pretrained(
                    'gpt2',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.gpt2_config,
                )
                
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm = GPT2Model.from_pretrained(
                    'gpt2',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.gpt2_config,
                )
                
            self.llm.init_weights()
            
        
        elif config.LLM == 'BERT':
            self.bert_config = BertConfig.from_pretrained('google-bert/bert-base-uncased')
            self.bert_config.num_hidden_layers = 6
            self.bert_config.output_attentions = True
            self.bert_config.output_hidden_states = True
            try:
                self.llm = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.bert_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.bert_config,
                )
                
                
        elif config.LLM == 'RoBERTa':
            self.bert_config = RobertaConfig.from_pretrained('FacebookAI/roberta-large')
            self.bert_config.num_hidden_layers = 6
            self.bert_config.output_attentions = True
            self.bert_config.output_hidden_states = True
            try:
                self.llm = RobertaModel.from_pretrained(
                    'google-bert/bert-large-uncased',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.bert_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm = RobertaModel.from_pretrained(
                    'FacebookAI/roberta-large',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.bert_config,
                )


        if config.LLM not in ['Linear', 'Att', 'Trans']:
            for name, param in self.llm.named_parameters():
                if 'ln_' in name or 'layernorm' in name or 'layer_norm' in name or 'ln' in name or 'wpe' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        else:
            if config.LLM == 'Linear':
                self.llm = LinearEncoder(config).to(DEVICE)
            elif config.LLM == 'Att':
                self.llm = MHAtt_Backbone(config)
            elif config.LLM == 'Trans':
                self.llm = Trans_Backbone(config)
        
        return self.llm
    