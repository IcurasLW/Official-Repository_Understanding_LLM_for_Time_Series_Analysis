#!pip install transformers

import math
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import pandas as pd

from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from einops import rearrange
from transformers import GPT2Tokenizer
from utils.tokenization import SerializerSettings, serialize_arr,serialize_arr 
from .prompt import Prompt 


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
        self.results_obj = Results_Obj()
        
    def forward(self, inputs_embeds):
        out = self.enccoder(inputs_embeds)
        self.results_obj.last_hidden_state = out
        return self.results_obj


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


class NoLLM_Backbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.resutls_obj = Results_Obj()
        
    def forward(self, inputs_embeds):
        self.resutls_obj.last_hidden_state = inputs_embeds
        return self.resutls_obj



class Model(nn.Module):
    def __init__(self, configs, device):
        super(Model, self).__init__()
        self.device = device
        self.configs = configs
        self.is_ln = configs.ln
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.patch_size = configs.patch_size
        self.stride = configs.stride
        self.d_ff = 768
        self.patch_num = (configs.seq_len - self.patch_size) // self.stride + 1
        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride)) 
        self.patch_num += 1
        
        self.backbone, self.tokenizer = self.get_llm(configs)
        
        if self.task_name == 'long_term_forecast':
            self.in_layer = nn.Linear(configs.patch_size*3, configs.d_model)
            self.out_layer = nn.Linear(int(configs.d_model / 3 * 64) , configs.pred_len)
            # self.out_layer = nn.Linear(int(configs.d_model / 3 * (self.patch_num+configs.prompt_length)) , configs.pred_len)
            self.prompt_pool = Prompt(configs, patch_num = self.patch_num, length=1, embed_dim=768, embedding_key='mean', prompt_init='uniform', prompt_pool=False, 
                    prompt_key=True, pool_size=self.configs.pool_size, top_k=self.configs.prompt_length, batchwise_prompt=False, prompt_key_init=self.configs.prompt_init,wte = self.gpt2.wte.weight).to(DEVICE)
            del self.gpt2
            for layer in (self.backbone, self.in_layer, self.out_layer):       
                layer.cuda()
                layer.train()

        elif self.task_name == 'classification':
            self.in_layer = nn.Linear(configs.patch_size*3*self.configs.enc_in, configs.d_model)
            self.out_layer = nn.Linear(int(configs.d_model * (self.patch_num+configs.prompt_length)) , configs.num_class)
            self.prompt_pool = Prompt(configs, patch_num = self.patch_num, length=1, embed_dim=768, embedding_key='mean', prompt_init='uniform', prompt_pool=False, 
                    prompt_key=True, pool_size=self.configs.pool_size, top_k=self.configs.prompt_length, batchwise_prompt=False, prompt_key_init=self.configs.prompt_init,wte = self.gpt2.wte.weight).to(DEVICE)
            del self.gpt2
            for layer in (self.backbone, self.in_layer, self.out_layer):       
                layer.cuda()
                layer.train()

        elif self.task_name == 'anomaly_detection':
            self.in_layer = nn.Linear(configs.patch_size*3, configs.d_model)
            self.out_layer = nn.Linear(int(configs.d_model / 3 * (self.patch_num+configs.prompt_length)) , configs.seq_len)
            self.prompt_pool = Prompt(configs, patch_num = self.patch_num, length=1, embed_dim=768, embedding_key='mean', prompt_init='uniform', prompt_pool=False, 
                    prompt_key=True, pool_size=self.configs.pool_size, top_k=self.configs.prompt_length, batchwise_prompt=False, prompt_key_init=self.configs.prompt_init,wte = self.gpt2.wte.weight).to(DEVICE)
            del self.gpt2
            for layer in (self.backbone, self.in_layer, self.out_layer):       
                layer.cuda()
                layer.train()
                
        elif self.task_name == 'imputation':
            self.in_layer = nn.Linear(configs.patch_size*3, configs.d_model)
            self.out_layer = nn.Linear(int(configs.d_model / 3 * (self.patch_num+configs.prompt_length)) , configs.seq_len)
            self.prompt_pool = Prompt(configs, patch_num = self.patch_num, length=1, embed_dim=768, embedding_key='mean', prompt_init='uniform', prompt_pool=False, 
                    prompt_key=True, pool_size=self.configs.pool_size, top_k=self.configs.prompt_length, batchwise_prompt=False, prompt_key_init=self.configs.prompt_init,wte = self.gpt2.wte.weight).to(DEVICE)
            del self.gpt2
            for layer in (self.backbone, self.in_layer, self.out_layer):       
                layer.cuda()
                layer.train()


    def forward(self, x_enc, mean_x, std_x, mask=None):
        if self.task_name == 'long_term_forecast':
            # dec_out, res, hid_states = self.forecast(x_enc, mean_x, std_x)
            # return dec_out[:, -self.pred_len:, :], res, hid_states  # [B, L, D]
            dec_out, res = self.forecast(x_enc, mean_x, std_x)
            return dec_out[:, -self.pred_len:, :], res  # [B, L, D]
        elif self.task_name == 'classification':
            dec_out,res = self.classification(x_enc, mean_x, std_x)
            return dec_out, res  # [B, L, D]
        elif self.task_name == 'anomaly_detection':
            dec_out, res = self.anomaly_detection(x_enc, mean_x, std_x)
            return dec_out, res 
        elif self.task_name == 'imputation':
            dec_out, res = self.imputation(x_enc, mean_x, std_x, mask)
            return dec_out, res 
        
        return None


    def imputation(self, x, means, stdev, mask):
        B, L, C, M = x.shape
        means = means.to(DEVICE).to(torch.float32)
        stdev = stdev.to(DEVICE).to(torch.float32)
        x = rearrange(x, 'b l c m -> (b m) c l') 
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        x = rearrange(x, 'b c n p -> b n (c p)', c = 3)  
        pre_prompted_embedding = self.in_layer(x.float())
        outs = self.prompt_pool(pre_prompted_embedding)
        prompted_embedding = outs['prompted_embedding']
        sim = outs['similarity']
        prompt_key = outs['prompt_key']
        simlarity_loss = outs['reduce_sim']
        
        last_embedding = self.backbone(inputs_embeds=prompted_embedding).last_hidden_state
        outputs = self.out_layer(last_embedding.reshape(B*3*M, -1))
        outputs = rearrange(outputs, '(b m c) h -> b m c h', b=B,m=M,c=3)
        outputs = outputs.sum(dim=2)
        outputs = rearrange(outputs, 'b m l -> b l m')
        # print(means.shape)
        # print(outputs.shape)
        res = dict()
        res['simlarity_loss'] = simlarity_loss
        outputs = outputs * stdev[:, :, :M]
        outputs = outputs + means[:, :, :M]
        
        return outputs, res


    def forecast(self, x, means, stdev):
        B, L, C, M = x.shape
        means = means.to(DEVICE).unsqueeze(1).to(torch.float32)
        stdev = stdev.to(DEVICE).unsqueeze(1).to(torch.float32)
        x = rearrange(x, 'b l c m -> (b m) c l') 
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        x = rearrange(x, 'b c n p -> b n (c p)', c = 3)  
        pre_prompted_embedding = self.in_layer(x.float()) # Time series
        outs = self.prompt_pool(pre_prompted_embedding)
        prompted_embedding = outs['prompted_embedding']
        sim = outs['similarity']
        prompt_key = outs['prompt_key']
        simlarity_loss = outs['reduce_sim']
        
        hidden = self.backbone(inputs_embeds=prompted_embedding)
        # hid_states = hidden.hidden_states
        last_embedding = hidden.last_hidden_state
        outputs = self.out_layer(last_embedding.reshape(B*3*M, -1))
        outputs = rearrange(outputs, '(b m c) h -> b m c h', b=B,m=M,c=3)
        outputs = outputs.sum(dim=2)
        outputs = rearrange(outputs, 'b m l -> b l m')

        res = dict()
        res['simlarity_loss'] = simlarity_loss
        outputs = outputs * stdev[:, :, :M]
        outputs = outputs + means[:, :, :M]

        # return outputs, res, hid_states
        return outputs, res


    def classification(self, x_enc, means, stdev):
        x = x_enc
        B, L, C, M = x.shape
        means = means.to(DEVICE).unsqueeze(1).to(torch.float32)
        stdev = stdev.to(DEVICE).unsqueeze(1).to(torch.float32)
        
        x = torch.tensor(x).to(DEVICE)
        
        x = rearrange(x, 'b l c d -> b l (c d)', c = 3)
        x = rearrange(x, 'b l e -> b e l')
        x = self.padding_patch_layer(x)
        x = rearrange(x, 'b e l -> b l e')
        x = x.unfold(dimension=1, size=self.patch_size, step=self.stride)
        x = rearrange(x, 'b n c p -> b n (c p)')
        pre_prompted_embedding = self.in_layer(x.float())
        
        outs = self.prompt_pool(pre_prompted_embedding)
        prompted_embedding = outs['prompted_embedding']
        sim = outs['similarity']
        prompt_key = outs['prompt_key']
        simlarity_loss = outs['reduce_sim']
        
        last_embedding = self.backbone(inputs_embeds=prompted_embedding).last_hidden_state
        last_embedding = self.out_layer(last_embedding.reshape(B, -1))
        res = dict()
        res['simlarity_loss'] = simlarity_loss
        return last_embedding, res
        
    
    def anomaly_detection(self, x_enc, means, stdev):
        x = x_enc
        B, L, C, M = x.shape
        means = means.to(DEVICE).unsqueeze(1).to(torch.float32)
        stdev = stdev.to(DEVICE).unsqueeze(1).to(torch.float32)
        x = rearrange(x, 'b l c m -> (b m) c l') 
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        x = rearrange(x, 'b c n p -> b n (c p)', c = 3)  
        pre_prompted_embedding = self.in_layer(x.float())
        outs = self.prompt_pool(pre_prompted_embedding)
        prompted_embedding = outs['prompted_embedding']
        sim = outs['similarity']
        prompt_key = outs['prompt_key']
        simlarity_loss = outs['reduce_sim']
        
        last_embedding = self.backbone(inputs_embeds=prompted_embedding).last_hidden_state
        outputs = self.out_layer(last_embedding.reshape(B*3*M, -1))
        outputs = rearrange(outputs, '(b m c) h -> b m c h', b=B,m=M,c=3)
        outputs = outputs.sum(dim=2)
        outputs = rearrange(outputs, 'b m l -> b l m')

        res = dict()
        res['simlarity_loss'] = simlarity_loss
        outputs = outputs * stdev[:, :, :M]
        outputs = outputs + means[:, :, :M]

        return outputs, res
    
    def get_llm(self, config):
        if config.LLM == 'GPT2':
            self.backbone = GPT2Model.from_pretrained('gpt2', output_attentions=True, output_hidden_states=True)  # loads a pretrained GPT-2 base model
            self.backbone.h = self.backbone.h[:config.gpt_layers]
            self.backbone = self.backbone.to(DEVICE)
            
        elif config.LLM == 'Random':
            self.gpt2_config = GPT2Config()
            self.gpt2_config.num_hidden_layers = config.gpt_layers
            self.gpt2_config.output_attentions = True
            self.gpt2_config.output_hidden_states = True
            self.backbone = GPT2Model(self.gpt2_config) # loads a pretrained GPT-2 base model
            self.backbone = self.backbone.to(DEVICE)

        if config.LLM not in ['Linear', 'Att', 'Trans', 'NoLLM']:
            for i, (name, param) in enumerate(self.backbone.named_parameters()):
                if 'ln' in name or 'wpe' in name or 'lora' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        
        else:
            if config.LLM == 'Linear':
                self.backbone = LinearEncoder(config).to(DEVICE)
                
            elif config.LLM == 'Att':
                self.backbone = MHAtt_Backbone(config).to(DEVICE)
                
            elif config.LLM == 'Trans':
                self.backbone = Trans_Backbone(config).to(DEVICE)
                
            elif config.LLM == 'NoLLM':
                self.backbone = NoLLM_Backbone(config).to(DEVICE)
                
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2',
                                                        trust_remote_code=True,
                                                        local_files_only=False
                                                        )
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        self.gpt2 = GPT2Model.from_pretrained('gpt2', output_attentions=True, output_hidden_states=True)
        self.gpt2.h = self.gpt2.h[:2]
        
        return self.backbone, self.tokenizer












