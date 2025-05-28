import torch
import torch.nn as nn
from einops import rearrange
from peft import LoraConfig, TaskType
from models.GPT2_arch import AccustumGPT2Model
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
        return out, torch.tensor(0)


class MHAtt_Backbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.enccoder = nn.Linear(in_features=config.d_model, out_features=config.d_model)
        self.mh_Att = nn.MultiheadAttention(config.d_model, num_heads=4, batch_first=True)
        self.resutls_obj = Results_Obj()
        
    def forward(self, inputs_embeds):
        out = self.enccoder(inputs_embeds)
        out, _ = self.mh_Att(out, out, out)
        return out, torch.tensor(0)



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

        return out, torch.tensor(0)


class NoLLM_Backbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.resutls_obj = Results_Obj()
        
    def forward(self, inputs_embeds):
        return inputs_embeds, torch.tensor(0)


class Encoder_PCA(nn.Module):
    def __init__(self, input_dim, word_embedding, hidden_dim=768, num_heads=12, num_encoder_layers=1):
        super(Encoder_PCA, self).__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads)
        
        self.word_embedding = word_embedding.T

    def forward(self, x):
        B = x.shape[0]
        if self.word_embedding.ndim == 2:
            self.word_embedding = self.word_embedding.repeat(B, 1, 1)
        elif self.word_embedding.shape[0] != B:
            self.word_embedding = self.word_embedding[0].repeat(B, 1, 1)

        x = self.linear(x)

        x = self.transformer_encoder(x.transpose(0, 1)).transpose(0, 1)

        x_time = x

        q = x.transpose(0, 1)
        k = v = self.word_embedding.transpose(0, 1)
        x, _ = self.cross_attention(q, k, v)

        x = x.transpose(0, 1)

        return x_time, x




class TimeTextMixer(nn.Module):
    def __init__(self, seq_len, text_topk, compress_token, d_model=None):
        super().__init__()
        self.text_topk = text_topk 
        self.seq_len = seq_len
        self.d_model = d_model
        self.compress_token = compress_token
        
        self.mlp_1 = nn.Sequential(nn.Linear(self.seq_len, self.compress_token),
                                   nn.ReLU(),
                                   nn.LayerNorm(self.compress_token),
                                   nn.Linear(self.compress_token, self.compress_token),
                                   nn.ReLU(),
                                   nn.LayerNorm(self.compress_token))
        
        self.mlp_2 = nn.Sequential(nn.Linear(self.d_model, self.d_model),
                                    nn.ReLU(),
                                    nn.LayerNorm(self.d_model),
                                    nn.Linear(self.d_model, self.d_model),
                                    nn.ReLU(),
                                    nn.LayerNorm(self.d_model))
        
    def forward(self, x_time, x_text):
        x = torch.concat([x_time, x_text], dim=1)
        x = x.permute(0, 2, 1)
        x = self.mlp_1(x)
        
        x = x.permute(0, 2, 1)
        x = self.mlp_2(x)
        return x 





class Model(nn.Module):
    def __init__(self, configs, device):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.configs = configs
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            inference_mode=False, 
            r=configs.r,
            lora_alpha=configs.lora_alpha,
            lora_dropout=configs.lora_dropout,
            target_modules=["c_attn"]
        )
    
        self.task_name = configs.task_name
        self.gpt2, self.gpt2_text = self.get_llm(configs)

        word_embedding = torch.tensor(torch.load(configs.word_embedding_path)).to(device=device)
        
        self.time_proj = nn.ModuleList([nn.Linear(configs.d_model, configs.d_model, bias=False) for _ in range(configs.gpt_layers+1)])
        self.text_proj = nn.ModuleList([nn.Linear(configs.d_model, configs.d_model, bias=False) for _ in range(configs.gpt_layers+1)])
        self.in_layer = Encoder_PCA(configs.seq_len, word_embedding, hidden_dim=configs.d_model)
        
        self.timetext_mixer = TimeTextMixer(configs.enc_in*2, 10, configs.enc_in, 768) # comment out if you don't need 
        
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.out_layer = nn.Linear(configs.d_model, configs.pred_len)
        elif self.task_name == 'classification':
            self.out_layer = nn.Linear(configs.d_model * configs.enc_in, configs.num_class)
        elif self.task_name == 'imputation':
            self.out_layer = nn.Linear(configs.d_model, configs.seq_len)
        elif self.task_name == 'anomaly_detection':
            self.out_layer = nn.Linear(configs.d_model, configs.seq_len)

        for layer in (self.gpt2_text, self.gpt2, self.in_layer, self.out_layer, self.time_proj, self.text_proj):
            layer.to(device=device)
            layer.train()
        
        self.cnt = 0



    def forecast(self, x):
        B, L, M = x.shape

        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach() 
        x /= stdev

        x = rearrange(x, 'b l m -> b m l')

        outputs_time1, outputs_text1 = self.in_layer(x)

        outputs_time, intermidiate_feat_time = self.gpt2(inputs_embeds=outputs_time1)
        outputs_text, intermidiate_feat_text = self.gpt2_text(inputs_embeds=outputs_text1)
        # residue connection
        outputs_time = outputs_time1 + outputs_time
        outputs_text = outputs_text1 + outputs_text
        
        # outputs_time = self.timetext_mixer(outputs_time, outputs_text)
        
        if self.configs.LLM in ['GPT2', 'Random']:
            intermidiate_feat_time = tuple([self.time_proj[idx](feat) for idx, feat in enumerate(list(intermidiate_feat_time))])
            intermidiate_feat_text = tuple([self.text_proj[idx](feat) for idx, feat in enumerate(list(intermidiate_feat_text))])
        elif self.configs.LLM in ['Trans', 'Att', 'Linear', 'NoLLM']:
            intermidiate_feat_time = self.time_proj[0](outputs_time)
            intermidiate_feat_text = self.text_proj[0](outputs_text)
            
        outputs_time = self.out_layer(outputs_time[:, -M:, :])
        outputs_text = self.out_layer(outputs_text[:, -M:, :])

        outputs_time = rearrange(outputs_time, 'b m l -> b l m')
        outputs_text = rearrange(outputs_text, 'b m l -> b l m')

        outputs_text = outputs_text * stdev + means
        outputs_time = outputs_time * stdev + means

        return {
            'outputs_text': outputs_text,
            'outputs_time':outputs_time,
            'intermidiate_time':intermidiate_feat_time,
            'intermidiate_text':intermidiate_feat_text,
            'in_time':outputs_time1,
            'in_txt':outputs_text1
        }


    def classification(self, x):
        B, L, M = x.shape

        x = rearrange(x, 'b l m -> b m l')

        outputs_time1, outputs_text1 = self.in_layer(x)
        
        outputs_time, intermidiate_feat_time = self.gpt2(inputs_embeds=outputs_time1)
        outputs_text, intermidiate_feat_text = self.gpt2_text(inputs_embeds=outputs_text1)
        
        outputs_time += outputs_time1
        outputs_text += outputs_text1
        
        if self.configs.LLM in ['GPT2', 'Random']:
            intermidiate_feat_time = tuple([self.time_proj[idx](feat) for idx, feat in enumerate(list(intermidiate_feat_time))])
            intermidiate_feat_text = tuple([self.text_proj[idx](feat) for idx, feat in enumerate(list(intermidiate_feat_text))])
        elif self.configs.LLM in ['Trans', 'Att', 'Linear', 'NoLLM']:
            intermidiate_feat_time = self.time_proj[0](outputs_time)
            intermidiate_feat_text = self.text_proj[0](outputs_text)
            
            
        outputs_time = outputs_time.reshape(B, -1)
        outputs_text = outputs_text.reshape(B, -1)
        
        outputs_time = self.out_layer(outputs_time)
        outputs_text = self.out_layer(outputs_text)
        
        return {
            'outputs_text': outputs_text,
            'outputs_time':outputs_time,
            'intermidiate_time':intermidiate_feat_time,
            'intermidiate_text':intermidiate_feat_text,
        }
    
    
    
    def imputation(self, x, mask):
        B, L, M = x.shape

        means = torch.sum(x, dim=1) / torch.sum(mask == 1, dim=1)
        x = x - means
        x = x.masked_fill(mask == 0, 0)

        stdev = torch.sqrt(torch.sum(x**2, dim=1) / torch.sum(mask == 1, dim=1) + 1e-5).unsqueeze(1).detach()
        x /= stdev

        x = rearrange(x, 'b l m -> b m l')

        outputs_time1, outputs_text1 = self.in_layer(x)
        outputs_time, intermidiate_feat_time = self.gpt2(inputs_embeds=outputs_time1)
        outputs_text, intermidiate_feat_text = self.gpt2_text(inputs_embeds=outputs_text1)
        # residue connection
        outputs_time += outputs_time1
        outputs_text += outputs_text1
        
        if self.configs.LLM in ['GPT2', 'Random']:
            intermidiate_feat_time = tuple([self.time_proj[idx](feat) for idx, feat in enumerate(list(intermidiate_feat_time))])
            intermidiate_feat_text = tuple([self.text_proj[idx](feat) for idx, feat in enumerate(list(intermidiate_feat_text))])
        elif self.configs.LLM in ['Trans', 'Att', 'Linear', 'NoLLM']:
            intermidiate_feat_time = self.time_proj[0](outputs_time)
            intermidiate_feat_text = self.text_proj[0](outputs_text)
            
        outputs_time = self.out_layer(outputs_time)
        outputs_text = self.out_layer(outputs_text)

        outputs_time = rearrange(outputs_time, 'b m l -> b l m')
        outputs_text = rearrange(outputs_text, 'b m l -> b l m')

        outputs_text = outputs_text * stdev + means
        outputs_time = outputs_time * stdev + means

        return {
            'outputs_text': outputs_text,
            'outputs_time':outputs_time,
            'intermidiate_time':intermidiate_feat_time,
            'intermidiate_text':intermidiate_feat_text,
        }

    def anomaly_detection(self, x):
        B, L, M = x.shape

        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach() 
        x /= stdev
        x = rearrange(x, 'b l m -> b m l')
        outputs_time1, outputs_text1 = self.in_layer(x)

        outputs_time, intermidiate_feat_time = self.gpt2(inputs_embeds=outputs_time1)
        outputs_text, intermidiate_feat_text = self.gpt2_text(inputs_embeds=outputs_text1)

        # residue connection
        outputs_time += outputs_time1
        outputs_text += outputs_text1
        if self.configs.LLM in ['GPT2', 'Random']:
            intermidiate_feat_time = tuple([self.time_proj[idx](feat) for idx, feat in enumerate(list(intermidiate_feat_time))])
            intermidiate_feat_text = tuple([self.text_proj[idx](feat) for idx, feat in enumerate(list(intermidiate_feat_text))])
        elif self.configs.LLM in ['Trans', 'Att', 'Linear', 'NoLLM']:
            intermidiate_feat_time = self.time_proj[0](outputs_time)
            intermidiate_feat_text = self.text_proj[0](outputs_text)

        outputs_time = self.out_layer(outputs_time)
        outputs_text = self.out_layer(outputs_text)

        outputs_time = rearrange(outputs_time, 'b m l -> b l m')
        outputs_text = rearrange(outputs_text, 'b m l -> b l m')

        outputs_text = outputs_text * stdev + means
        outputs_time = outputs_time * stdev + means

        return {
            'outputs_text': outputs_text,
            'outputs_time':outputs_time,
            'intermidiate_time':intermidiate_feat_time,
            'intermidiate_text':intermidiate_feat_text,
        }



    def forward(self, x, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            output = self.forecast(x)
        if self.task_name == 'classification':
            output = self.classification(x)
        if self.task_name == "imputation":
            output = self.imputation(x, mask)
        if self.task_name == "anomaly_detection":
            output = self.anomaly_detection(x)
        return output


    def get_llm(self, config):
        if config.LLM == 'GPT2':
            self.gpt2 = AccustumGPT2Model.from_pretrained('gpt2', output_attentions=True, output_hidden_states=True)  # loads a pretrained GPT-2 base model
            self.gpt2_text = AccustumGPT2Model.from_pretrained('gpt2', output_attentions=True, output_hidden_states=True)  # loads a pretrained GPT-2 base model
            self.gpt2.h = self.gpt2.h[:config.gpt_layers]
            self.gpt2_text.h = self.gpt2_text.h[:config.gpt_layers]
    
        elif config.LLM == 'Random':
            self.gpt2_config = GPT2Config()
            self.gpt2_config.num_hidden_layers = config.gpt_layers
            self.gpt2_config.output_attentions = True
            self.gpt2_config.output_hidden_states = True
            self.gpt2 = AccustumGPT2Model(self.gpt2_config) # loads a pretrained GPT-2 base model
            self.gpt2_text = AccustumGPT2Model(self.gpt2_config)  # loads a pretrained GPT-2 base model
            

        if config.LLM not in ['Linear', 'Att', 'Trans', 'NoLLM']:
            for i, (name, param) in enumerate(self.gpt2.named_parameters()):
                if 'ln' in name or 'wpe' in name or 'lora' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            
            for i, (name, param) in enumerate(self.gpt2_text.named_parameters()):
                if 'wpe' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        else:
            if config.LLM == 'Linear':
                self.gpt2 = LinearEncoder(config).to(DEVICE)
                self.gpt2_text = LinearEncoder(config).to(DEVICE)
                
            elif config.LLM == 'Att':
                self.gpt2 = MHAtt_Backbone(config)
                self.gpt2_text = MHAtt_Backbone(config)
                
            elif config.LLM == 'Trans':
                self.gpt2 = Trans_Backbone(config)
                self.gpt2_text = Trans_Backbone(config)
                
            elif config.LLM == 'NoLLM':
                self.gpt2 = NoLLM_Backbone(config)
                self.gpt2_text = NoLLM_Backbone(config)
                
        return self.gpt2, self.gpt2_text