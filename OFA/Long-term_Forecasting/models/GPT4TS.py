import numpy as np
import torch
import torch.nn as nn
from torch import optim

from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers import BertTokenizer, BertModel
from einops import rearrange
from embed import DataEmbedding, DataEmbedding_wo_time
from transformers.models.gpt2.configuration_gpt2 import GPT2Config


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

class NoLLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.resutls_obj = Results_Obj()

    def forward(self, inputs_embeds):
        self.resutls_obj.last_hidden_state = inputs_embeds
        return self.resutls_obj

class GPT4TS(nn.Module):
    
    def __init__(self, configs, device):
        super(GPT4TS, self).__init__()
        self.is_gpt = configs.is_gpt
        self.patch_size = configs.patch_size
        self.pretrain = configs.pretrain
        self.stride = configs.stride
        self.patch_num = (configs.seq_len - self.patch_size) // self.stride + 1

        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride)) 
        self.patch_num += 1
        
        self.llm = self.get_llm(configs)
        
        self.in_layer = nn.Linear(configs.patch_size, configs.d_model)
        self.out_layer = nn.Linear(configs.d_model * self.patch_num, configs.pred_len)
        
        for layer in (self.llm, self.in_layer, self.out_layer):
            layer.to(device=device)
            layer.train()
        
        self.cnt = 0


    def forward(self, x, itr):
        B, L, M = x.shape

        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False)+ 1e-5).detach() 
        x /= stdev

        x = rearrange(x, 'b l m -> b m l')

        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        x = rearrange(x, 'b m n p -> (b m) n p')
        outputs = self.in_layer(x)
        mid_state = self.llm(inputs_embeds=outputs)
        outputs = mid_state.last_hidden_state
        hid_state = mid_state.hidden_states
        outputs = self.out_layer(outputs.reshape(B*M, -1))
        outputs = rearrange(outputs, '(b m) l -> b l m', b=B)
        outputs = outputs * stdev
        outputs = outputs + means
        return outputs, hid_state



    def get_llm(self, config):
        if config.LLM == 'GPT2':
            self.gpt2_config = GPT2Config.from_pretrained('gpt2')
            self.gpt2_config.num_hidden_layers = config.gpt_layers
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
            self.gpt2_config = GPT2Config()
            self.gpt2_config.output_attentions = True
            self.gpt2_config.output_hidden_states = True
            self.gpt2_config.num_hidden_layers = config.gpt_layers
            self.llm = GPT2Model(self.gpt2_config)


        if config.LLM in ['GPT2', 'Random']:
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
            elif config.LLM == 'NoLLM':
                self.llm = NoLLM()
        return self.llm
    