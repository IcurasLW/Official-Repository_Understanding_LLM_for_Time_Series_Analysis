from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual,adjust_model
from utils.metrics import metric
import torch
import torch.nn as nn
from models import  S2IPLLM
from torch.nn.utils import clip_grad_norm_
from utils.losses import mape_loss, mase_loss, smape_loss
from transformers import AdamW
from torch.utils.data import Dataset, DataLoader
from torch import optim
import os
import time
import warnings
import numpy as np

from tqdm import tqdm

warnings.filterwarnings('ignore')
torch.set_default_dtype(torch.float32)

class Exp_Long_Term_Forecast(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'S2IPLLM': S2IPLLM,
            
        }

        self.device = torch.device('cuda:0')
        self.model = self._build_model()
        
        self.train_data, self.train_loader = self._get_data(flag='train')
        self.vali_data, self.vali_loader = self._get_data(flag='val')
        # self.test_data, self.test_loader = self._get_data(flag='test')

        self.optimizer = self._select_optimizer()
        self.criterion = self._select_criterion()

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args, self.device).to(self.device)
        
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        if self.args.loss=='MSE':
            criterion = nn.MSELoss()
    
        elif self.args.loss=='SMAPE':
            criterion = smape_loss()
            
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        print('=============================================')
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, means, stdev) in tqdm(enumerate(vali_loader)):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                if i % 20 == 0:
                    torch.cuda.empty_cache()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).to(torch.bfloat16).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, means, stdev, mask=batch_x_mark)[0]
                        else:
                            outputs = self.model(batch_x, means, stdev, mask=batch_x_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, means, stdev, mask=batch_x_mark)[0]
                    else:
                        # outputs, res, _ = self.model(batch_x, means, stdev, mask=batch_x_mark)
                        outputs, res = self.model(batch_x, means, stdev, mask=batch_x_mark)
                f_dim = -1 if self.args.features == 'MS' else 0

                outputs = outputs[:, -self.args.pred_len:, f_dim:self.args.number_variable].float().to(self.device)
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:self.args.number_variable].float().to(self.device)
                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                loss = criterion(pred, true)
                total_loss.append(loss)
                
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss


    def train(self, setting):
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(self.train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        # total_params = sum(p.numel() for p in self.model.parameters())
        # print(total_params)
        # print('Number of Prameter ======================================================================================================================')
        print('Training Steps ======================================================================================================================')
        print(train_steps)
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            simlarity_losses = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, means, stdev) in tqdm(enumerate(self.train_loader)):
                iter_count += 1
                self.optimizer.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float().to(self.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, means, stdev, mask=batch_x_mark)[0]
                        else:
                            outputs = self.model(batch_x, means, stdev, mask=batch_x_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:self.args.number_variable]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:self.args.number_variable].float().to(self.device)
                        loss = self.criterion(outputs, batch_y)
                        
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, means, stdev, mask=batch_x_mark)[0]
                    else:
                        # outputs, res, _ = self.model(batch_x, means, stdev, mask=batch_x_mark)
                        outputs, res = self.model(batch_x, means, stdev, mask=batch_x_mark)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:self.args.number_variable]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:self.args.number_variable].float().to(self.device)
                    loss = self.criterion(outputs, batch_y)
                    
                    train_loss.append(loss.item())
                    simlarity_losses.append(res['simlarity_loss'].item())
                    loss += self.args.sim_coef*res['simlarity_loss']
                    
                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    loss.backward()
                    self.optimizer.step()
                else:
                    loss.backward()
                    self.optimizer.step()
                
            torch.cuda.empty_cache()
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            sim_loss = np.average(simlarity_losses)
            vali_loss = self.vali(self.vali_data, self.vali_loader, self.criterion)
            

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Sim Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss,sim_loss))
            
            
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            
            adjust_learning_rate(self.optimizer, epoch + 1, self.args)
            adjust_model(self.model, epoch + 1,self.args)
            


    def test(self, setting, test=1):
        test_data, test_loader = self._get_data(flag='test')
        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        sim_matrix = []
        input_embedding = []
        prompted_embedding = []
        last_embedding = []
        # save_hid = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, means, stdev) in tqdm(enumerate(test_loader)):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                if i % 20 == 0:
                    torch.cuda.empty_cache()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :])
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, means, stdev, mask=batch_x_mark)[0]
                        else:
                            
                            outputs = self.model(batch_x, means, stdev, mask=batch_x_mark)
                else:
                    if self.args.output_attention:
                        outputs =  self.model(batch_x, means, stdev, mask=batch_x_mark)[0]
                    else:
                        # outputs, res, hid_states =  self.model(batch_x, means, stdev, mask=batch_x_mark)
                        outputs, res =  self.model(batch_x, means, stdev, mask=batch_x_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:self.args.number_variable].float().detach().cpu().numpy()
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:self.args.number_variable].float().detach().cpu().numpy()
                pred = outputs
                true = batch_y
                preds.append(pred)
                trues.append(true)
        preds = np.array(preds)
        trues = np.array(trues)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open(f"result_long_term_forecast_{self.args.LLM}_{self.args.model_id}.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        # np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)
        # np.save(f'{self.args.LLM}_{self.args.model_id}_time_token.npy', save_hid)
        return mse, mae
    
