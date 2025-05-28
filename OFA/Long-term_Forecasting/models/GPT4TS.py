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
        self.text_topk = 5
        
        
        self.words_embedding = self.llm.get_input_embeddings().weight.data.to(DEVICE)
        self.att = nn.MultiheadAttention(embed_dim=self.words_embedding.shape[-1], num_heads=4, batch_first=True)
        
        
        self.mlp_1 = nn.Sequential(nn.Linear(self.patch_num + self.text_topk * self.patch_num, 8),
                                   nn.ReLU(),
                                   nn.LayerNorm(8),
                                   nn.Linear(8, 8),
                                   nn.ReLU(),
                                   nn.LayerNorm(8))
        
        self.mlp_2 = nn.Sequential(nn.Linear(self.words_embedding.shape[-1], self.words_embedding.shape[-1]),
                                    nn.ReLU(),
                                    nn.LayerNorm(self.words_embedding.shape[-1]),
                                    nn.Linear(self.words_embedding.shape[-1], self.words_embedding.shape[-1]),
                                    nn.ReLU(),
                                    nn.LayerNorm(self.words_embedding.shape[-1]))
        self.words2proto = nn.Linear(self.words_embedding.shape[0], 1000)
        
        self.in_layer = nn.Linear(configs.patch_size, configs.d_model)        
        self.dropout = nn.Dropout(0.5)
        self.out_layer = nn.Linear(configs.d_model * 8, configs.pred_len)
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
        
        
        ## Comment the below sesion if you don't need it
        words = self.words2proto(self.words_embedding.permute(1, 0))
        words = words.permute(1, 0).unsqueeze(0)
        _, attn_weights = self.att(outputs, words.expand(outputs.shape[0], -1, -1), words.expand(outputs.shape[0], -1, -1))
        topk_weight, topk_indices = torch.topk(attn_weights, k=self.text_topk, dim=-1) 
        words_expand = words.unsqueeze(0).expand(outputs.shape[0], outputs.shape[1], -1, -1) #[32, 1000, 768]

        topk_v = torch.gather(words_expand, dim=2, index=topk_indices.unsqueeze(-1).expand(-1, -1, -1, 768))  # (32, 4, 64, topk, 768)
        topk_v = topk_v.view(B, -1, 768)
        outputs = torch.concat([topk_v, outputs], dim=1)
        outputs = outputs.permute(0, 2, 1)
        outputs = self.mlp_1(outputs)
        
        outputs = outputs.permute(0, 2, 1)
        outputs = self.mlp_2(outputs)
        ## Comment the above sesion if you don't need it
        
        
        mid_state = self.llm(inputs_embeds=outputs)
        outputs = mid_state.last_hidden_state
        last_state = None
        first_state = None
        last_state = mid_state.hidden_states[-1]
        first_state = mid_state.hidden_states[0]
        
        outputs = self.out_layer(outputs.reshape(B*M, -1))
        outputs = rearrange(outputs, '(b m) l -> b l m', b=B)
        outputs = outputs * stdev
        outputs = outputs + means
        return outputs, first_state, last_state




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
    