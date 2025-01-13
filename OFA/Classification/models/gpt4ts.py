from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from transformers import RobertaModel, RobertaConfig, RobertaTokenizer, LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, BertModel, BertTokenizer
from transformers import GPT2ForSequenceClassification
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from einops import rearrange
from models.embed import DataEmbedding, DataEmbedding_wo_time

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class Results_Obj:
    def __init__(self):
        self.last_hidden_state = None

    def set_results(self, results):
        self.last_hidden_state = results


class LinearEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.enccoder = nn.Sequential(nn.LayerNorm(config['d_model']), nn.Linear(in_features=config['d_model'], out_features=config['d_model']))
        self.resutls_obj = Results_Obj()
        
    def forward(self, inputs_embeds):
        out = self.enccoder(inputs_embeds)
        self.resutls_obj.last_hidden_state = out
        return self.resutls_obj


class MHAtt_Backbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.enccoder = nn.Linear(in_features=config['d_model'], out_features=config['d_model'])
        self.mh_Att = nn.MultiheadAttention(config['d_model'], num_heads=4, batch_first=True)
        self.resutls_obj = Results_Obj()
        
    def forward(self, inputs_embeds):
        out = self.enccoder(inputs_embeds)
        out, _ = self.mh_Att(out, out, out)
        self.resutls_obj.last_hidden_state = out
        return self.resutls_obj


class Trans_Backbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.enccoder = nn.Linear(in_features=config['d_model'], out_features=config['d_model'])
        trans_layer = nn.TransformerEncoderLayer(config['d_model'], nhead=4, batch_first=True)
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


class gpt4ts(nn.Module):
    def __init__(self, config, data):
        super(gpt4ts, self).__init__()
        self.pred_len = 0
        self.seq_len = data.max_seq_len
        self.max_len = data.max_seq_len
        self.patch_size = config['patch_size']
        self.stride = config['stride']
        self.gpt_layers = 6
        self.feat_dim = data.feature_df.shape[1]
        self.num_classes = len(data.class_names)
        self.d_model = config['d_model']
        self.patch_num = (self.seq_len - self.patch_size) // self.stride + 1

        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride)) 
        self.patch_num += 1
        self.enc_embedding = DataEmbedding(self.feat_dim * self.patch_size, config['d_model'], config['dropout'])
        self.llm = self.get_llm(config)
        
        # self.gpt2 = GPT2Model.from_pretrained('gpt2', output_attentions=True, output_hidden_states=True)
        # self.gpt2.h = self.gpt2.h[:self.gpt_layers]
        
        for name, param in self.llm.named_parameters():
            if 'ln_' in name or 'layernorm' in name or 'layer_norm' in name or 'ln' in name or 'wpe' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        device = torch.device('cuda:{}'.format(0))
        self.llm.to(device=device)

        self.act = F.gelu
        self.dropout = nn.Dropout(0.1)
        self.ln_proj = nn.LayerNorm(config['d_model'] * self.patch_num)
        self.out_layer = nn.Linear(config['d_model'] * self.patch_num, self.num_classes)
        
        
    def forward(self, x_enc, x_mark_enc=None):
        B, L, M = x_enc.shape
        input_x = rearrange(x_enc, 'b l m -> b m l')
        input_x = self.padding_patch_layer(input_x)
        input_x = input_x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        input_x = rearrange(input_x, 'b m n p -> b n (p m)')
        
        enc_embed = self.enc_embedding(input_x, None)
        outputs = self.llm(inputs_embeds=enc_embed)
        # hidden_state = outputs.hidden_states
        # hidden_state = torch.stack(hidden_state, dim=0)
        last_hidden = outputs.last_hidden_state
        last_hidden = self.act(last_hidden)
        outputs = last_hidden.reshape(B, -1)
        outputs = self.ln_proj(outputs)
        outputs = self.out_layer(outputs)
        
        return outputs
    
    
    def get_llm(self, config):
        
        if config['LLM']  == 'GPT2':
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
                
        
        elif config['LLM'] == 'Random':
            self.gpt2_config = GPT2Config()
            self.gpt2_config.num_hidden_layers = config.gpt_layers
            self.llm = GPT2Model(self.gpt2_config)

        
        
        if config['LLM'] not in ['Linear', 'Att', 'Trans', 'NoLLM']:
            for name, param in self.llm.named_parameters():
                if 'ln_' in name or 'layernorm' in name or 'layer_norm' in name or 'ln' in name or 'wpe' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        else:
            if config['LLM'] == 'Linear':
                self.llm = LinearEncoder(config).to(DEVICE)
            elif config['LLM'] == 'Att':
                self.llm = MHAtt_Backbone(config)
            elif config['LLM'] == 'Trans':
                self.llm = Trans_Backbone(config)
            elif config['LLM'] == 'NoLLM':
                self.llm = NoLLM_Backbone(config)
                
        return self.llm