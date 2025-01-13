from math import sqrt

import torch
import torch.nn as nn

from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer
from layers.Embed import PatchEmbedding
import transformers
from layers.StandardNorm import Normalize
transformers.logging.set_verbosity_error()

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x



class Results_Obj:
    def __init__(self):
        self.last_hidden_state = None

    def set_results(self, results):
        self.last_hidden_state = results


class LinearEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.enccoder = nn.Sequential(nn.LayerNorm(config.llm_dim), nn.Linear(in_features=config.llm_dim, out_features=config.llm_dim))
        self.results_obj = Results_Obj()
        
    def forward(self, inputs_embeds):
        out = self.enccoder(inputs_embeds)
        self.results_obj.last_hidden_state = out
        return self.results_obj


class MHAtt_Backbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.enccoder = nn.Linear(in_features=config.llm_dim, out_features=config.llm_dim)
        self.mh_Att = nn.MultiheadAttention(config.llm_dim, num_heads=4, batch_first=True)
        self.resutls_obj = Results_Obj()
        
    def forward(self, inputs_embeds):
        out = self.enccoder(inputs_embeds)
        out, _ = self.mh_Att(out, out, out)
        self.resutls_obj.last_hidden_state = out
        return self.resutls_obj


class Trans_Backbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.enccoder = nn.Linear(in_features=config.llm_dim, out_features=config.llm_dim)
        trans_layer = nn.TransformerEncoderLayer(config.llm_dim, nhead=4, batch_first=True)
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

    def __init__(self, configs, patch_len=16, stride=8):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.d_ff = configs.d_ff
        self.top_k = 5
        self.d_llm = configs.llm_dim
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.llm, self.tokenizer = self.get_llm(configs)
        
        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token
            
            
        if configs.prompt_domain:
            self.description = configs.content
        else:
            self.description = 'The Electricity Transformer Temperature (ETT) is a crucial indicator in the electric power long-term deployment.'

        self.dropout = nn.Dropout(configs.dropout)
        self.patch_embedding = PatchEmbedding(
            configs.d_model, self.patch_len, self.stride, configs.dropout)


        if configs.llm_model in ['Linear', 'Att', 'Trans', 'NoLLM']:
            self.llama = LlamaModel.from_pretrained('huggyllama/llama-7b', output_attentions=True, output_hidden_states=True)
            self.word_embeddings = self.llama.get_input_embeddings().weight
            self.word_encoder = self.llama.get_input_embeddings()
            del self.llama
        else:
            self.word_embeddings = self.llm.get_input_embeddings().weight
            
            
        self.vocab_size = self.word_embeddings.shape[0]
        self.num_tokens = 1000
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)
        self.reprogramming_layer = ReprogrammingLayer(configs.d_model, configs.n_heads, self.d_ff, self.d_llm)
        self.patch_nums = int((configs.seq_len - self.patch_len) / self.stride + 2)
        self.head_nf = self.d_ff * self.patch_nums

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.output_projection = FlattenHead(configs.enc_in, self.head_nf, self.pred_len, head_dropout=configs.dropout)
            
        elif configs.task_name == 'classification':
            self.output_projection = nn.Linear(in_features=self.d_llm, out_features=self.configs.n_classes)
            
        elif configs.task_name == 'imputation':
            self.output_projection = FlattenHead(configs.enc_in, self.head_nf, configs.seq_len, head_dropout=configs.dropout)
            
        elif configs.task_name == 'anomaly_detection':
            self.output_projection = FlattenHead(configs.enc_in, self.head_nf, configs.seq_len, head_dropout=configs.dropout)
            
        self.normalize_layers = Normalize(configs.enc_in, affine=False)
        
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, prompt, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, None, None, None, prompt=prompt)
            return dec_out[:, -self.pred_len:, :]
        
        elif self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, None, None, None, prompt)
            return dec_out
        
        elif self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc, prompt)
            return dec_out
        
        elif self.task_name == 'classification':
            dec_out = self.classification(x_enc, prompt)
            return dec_out


    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, prompt):
        
        x_enc = self.normalize_layers(x_enc, 'norm')
        B, T, N = x_enc.size()
        x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
        x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous()

        prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids.to(DEVICE)
        if self.configs.llm_model in ['Linear', 'Att', 'Trans', 'NoLLM']:
            prompt_embeddings = self.word_encoder(prompt.to(DEVICE))  # (batch, prompt_token, dim)
        else:
            prompt_embeddings = self.llm.get_input_embeddings()(prompt.to(DEVICE))
        
        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)
        x_enc = x_enc.permute(0, 2, 1).contiguous()
        enc_out, n_vars = self.patch_embedding(x_enc)
        enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)
        llama_enc_out = torch.cat([prompt_embeddings, enc_out], dim=1)
        dec_out = self.llm(inputs_embeds=llama_enc_out).last_hidden_state
        
        dec_out = dec_out[:, :, :self.d_ff]
        dec_out = torch.reshape(dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]))
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous()
        dec_out = self.output_projection(dec_out[:, :, :, -self.patch_nums:])
        dec_out = dec_out.permute(0, 2, 1).contiguous()
        dec_out = self.normalize_layers(dec_out, 'denorm')
        
        return dec_out


    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark, prompt):
        
        x_enc = self.normalize_layers(x_enc, 'norm')
        B, T, N = x_enc.size()
        x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
        x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous()

        prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
        if self.configs.llm_model in ['Linear', 'Att', 'Trans', 'NoLLM']:
            prompt_embeddings = self.word_encoder(prompt.to(DEVICE))  # (batch, prompt_token, dim)
        else:
            prompt_embeddings = self.llm.get_input_embeddings()(prompt.to(DEVICE))
        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0).to(torch.bfloat16)).permute(1, 0)

        x_enc = x_enc.permute(0, 2, 1).contiguous()
        enc_out, n_vars = self.patch_embedding(x_enc.to(torch.bfloat16))
        enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)
        llama_enc_out = torch.cat([prompt_embeddings, enc_out], dim=1)
        dec_out = self.llm(inputs_embeds=llama_enc_out).last_hidden_state
        dec_out = dec_out[:, :, :self.d_ff]

        dec_out = torch.reshape(dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]))
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous()
        dec_out = self.output_projection(dec_out[:, :, :, -self.patch_nums:])
        dec_out = dec_out.permute(0, 2, 1).contiguous()
        dec_out = self.normalize_layers(dec_out, 'denorm')
        return dec_out


    def anomaly_detection(self, x_enc, prompt):
        x_enc = self.normalize_layers(x_enc, 'norm')
        B, T, N = x_enc.size()
        x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
        x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous()
        
        prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
        if self.configs.llm_model in ['Linear', 'Att', 'Trans', 'NoLLM']:
            prompt_embeddings = self.word_encoder(prompt.to(DEVICE))  # (batch, prompt_token, dim)
        else:
            prompt_embeddings = self.llm.get_input_embeddings()(prompt.to(DEVICE))
        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)

        x_enc = x_enc.permute(0, 2, 1).contiguous()
        enc_out, n_vars = self.patch_embedding(x_enc)
        enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)
        llama_enc_out = torch.cat([prompt_embeddings, enc_out], dim=1)
        dec_out = self.llm(inputs_embeds=llama_enc_out).last_hidden_state
        dec_out = dec_out[:, :, :self.d_ff]

        dec_out = torch.reshape(dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]))
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous()
        dec_out = self.output_projection(dec_out[:, :, :, -self.patch_nums:])
        dec_out = dec_out.permute(0, 2, 1).contiguous()
        dec_out = self.normalize_layers(dec_out, 'denorm')
        return dec_out
    

    def classification(self, x_enc, prompt):
        x_enc = self.normalize_layers(x_enc, 'norm')
        B, T, N = x_enc.size()
        x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous()
        x_enc = x_enc.permute(0, 2, 1).contiguous()
        
        enc_out, n_vars = self.patch_embedding(x_enc.to(torch.bfloat16))
        prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048 - x_enc.shape[1]).input_ids
        
        if self.configs.llm_model in ['Linear', 'Att', 'Trans', 'NoLLM']:
            prompt_embeddings = self.word_encoder(prompt.to(DEVICE))  # (batch, prompt_token, dim)
        else:
            prompt_embeddings = self.llm.get_input_embeddings()(prompt.to(DEVICE))
            
        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)
        enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)
        llama_enc_out = torch.cat([enc_out, prompt_embeddings], dim=1)
        dec_out = self.llm(inputs_embeds=llama_enc_out).last_hidden_state
        
        dec_out = torch.mean(dec_out, dim=1)
        dec_out = self.output_projection(dec_out)
        return dec_out
    
    
    
    def get_llm(self, config):
        if config.llm_model == 'LLaMa':
            self.llama_config = LlamaConfig.from_pretrained('huggyllama/llama-7b')
            self.llama_config.num_hidden_layers = config.llm_layers
            self.llama_config.output_attentions = True
            self.llama_config.output_hidden_states = True
            self.llm = LlamaModel.from_pretrained(
                # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
                'huggyllama/llama-7b',
                trust_remote_code=True,
                local_files_only=True,
                config=self.llama_config,
                # load_in_4bit=True
            )
            self.tokenizer = LlamaTokenizer.from_pretrained(
                # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
                'huggyllama/llama-7b',
                trust_remote_code=True,
                local_files_only=True
            )
            
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            
        elif config.llm_model == 'Random':
            self.llama_config = LlamaConfig()
            self.llama_config.num_hidden_layers = config.llm_layers
            self.llama_config.output_attentions = True
            self.llama_config.output_hidden_states = True
            self.llm = LlamaModel(config=self.llama_config)

            self.tokenizer = LlamaTokenizer.from_pretrained(
                # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
                'huggyllama/llama-7b',
                trust_remote_code=True,
                local_files_only=True
            )
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            
            
        elif config.llm_model == 'Linear':
            self.llm = LinearEncoder(config)
        elif config.llm_model == 'Att':
            self.llm = MHAtt_Backbone(config)
        elif config.llm_model == 'Trans':
            self.llm = Trans_Backbone(config)
        elif config.llm_model == 'NoLLM':
            self.llm = NoLLM_Backbone(config)
        
        if config.llm_model not in ['Linear', 'Att', 'Trans', 'NoLLM']:
            for i, (name, param) in enumerate(self.llm.named_parameters()):
                param.requires_grad = False
        else:
            self.tokenizer = LlamaTokenizer.from_pretrained(
                'huggyllama/llama-7b',
                trust_remote_code=True,
                local_files_only=True
            )
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                
        return self.llm, self.tokenizer



class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        super(ReprogrammingLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)
        out = self.reprogramming(target_embedding, source_embedding, value_embedding)
        out = out.reshape(B, L, -1)

        return self.out_projection(out)

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape
        scale = 1. / sqrt(E)
        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)
        return reprogramming_embedding

