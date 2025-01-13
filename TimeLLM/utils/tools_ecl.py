import numpy as np
import torch
import matplotlib.pyplot as plt
import shutil
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score


plt.switch_backend('agg')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def adjust_learning_rate(optimizer, scheduler, epoch, args, printout=True):
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == 'PEMS':
        lr_adjust = {epoch: args.learning_rate * (0.95 ** (epoch // 1))}
    elif args.lradj == 'TST':
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}
    elif args.lradj == 'constant':
        lr_adjust = {epoch: args.learning_rate}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if printout:
            print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, accelerator=None, patience=7, verbose=False, delta=0, save_mode=True):
        self.accelerator = accelerator
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.save_mode = save_mode

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            if self.save_mode:
                self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.accelerator is None:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            else:
                self.accelerator.print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if self.save_mode:
                self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            if self.accelerator is not None:
                self.accelerator.print(
                    f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            else:
                print(
                    f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
                
                
        if not os.path.exists(path):
            os.makedirs(path)
            
        if self.accelerator is not None:
            model = self.accelerator.unwrap_model(model)
            torch.save(model.state_dict(), path + '/' + 'checkpoint')
        else:
            torch.save(model.state_dict(), path + '/' + 'checkpoint')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)


def del_files(dir_path):
    shutil.rmtree(dir_path)


def vali(args, model, vali_data, vali_loader, criterion, mae_metric):
    total_loss = []
    total_mae_loss = []
    model.eval()
    num_batch = len(vali_loader)
    print(f'======================================== Number Batch of Validation or Tesing ------> {num_batch} ========================================')
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, prompts) in enumerate(vali_loader):
            if i % 100 == 0:
                torch.cuda.empty_cache()
            batch_x = batch_x.float().to(DEVICE)
            batch_y = batch_y.float()

            batch_x_mark = batch_x_mark.float().to(DEVICE)
            batch_y_mark = batch_y_mark.float().to(DEVICE)

            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(DEVICE)
            # encoder - decoder
            if args.use_amp:
                with torch.cuda.amp.autocast():
                    if args.output_attention:
                        outputs, ts_token = model(batch_x, batch_x_mark, dec_inp, batch_y_mark, prompt=prompts)[0]
                    else:
                        outputs, ts_token = model(batch_x, batch_x_mark, dec_inp, batch_y_mark, prompt=prompts)
            else:
                if args.output_attention:
                    outputs, ts_token = model(batch_x, batch_x_mark, dec_inp, batch_y_mark, prompt=prompts)[0]
                else:
                    outputs, ts_token = model(batch_x, batch_x_mark, dec_inp, batch_y_mark, prompt=prompts)
            
            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:].to(DEVICE)
            pred = outputs.detach()
            true = batch_y.detach()
            loss = criterion(pred, true)
            mae_loss = mae_metric(pred, true)
            total_loss.append(loss.item())
            total_mae_loss.append(mae_loss.item())
            
            del ts_token
            
    total_loss = np.average(total_loss)
    total_mae_loss = np.average(total_mae_loss)
    model.train()
    return total_loss, total_mae_loss



def vali_classification(args, model, vali_data, vali_loader, criterion):
    total_loss = []
    model.eval()
    num_batch = len(vali_loader)
    val_scores = []
    val_true = []
    y_pred = []
    print(f'======================================== Number Batch of Validation or Tesing ------> {num_batch} ========================================')
    with torch.no_grad():
        for i, (batch_x, batch_y) in enumerate(vali_loader):
            batch_x = batch_x.float().to(DEVICE)
            batch_y = batch_y.squeeze(-1).long().to(DEVICE)

            batch_x_mark = None
            batch_y_mark = None

            # decoder input
            dec_inp = None
            # encoder - decoder
            if args.use_amp:
                with torch.cuda.amp.autocast():
                    if args.output_attention:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            else:
                if args.output_attention:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)


            loss = criterion(outputs, batch_y)
            outputs = torch.nn.functional.softmax(outputs, dim=1)
            val_scores.extend(outputs.detach().cpu().numpy())
            _, y_pred_batch = torch.max(outputs, dim=1)
            y_pred.extend(y_pred_batch.detach().cpu().numpy())
            val_true.extend(batch_y.detach().cpu().numpy())
            total_loss.append(loss.item())
            
    total_loss = np.average(total_loss)
    y_pred = np.stack(y_pred)
    val_true = np.stack(val_true)
    acc = accuracy_score(val_true, y_pred)
    
    
    model.train()
    return total_loss, acc






def test(args, model, test_data, test_loader, criterion, mae_metric):
    total_loss = []
    total_mae_loss = []
    model.eval()
    num_batch = len(test_loader)
    print(f'======================================== Number Batch of Tesing ------> {num_batch} ========================================')
    ts_tokens = []
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, prompts) in tqdm(enumerate(test_loader)):
            if i % 100 == 0:
                torch.cuda.empty_cache()
            batch_x = batch_x.float().to(DEVICE)
            batch_y = batch_y.float()

            batch_x_mark = batch_x_mark.float().to(DEVICE)
            batch_y_mark = batch_y_mark.float().to(DEVICE)

            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(DEVICE)
            # encoder - decoder
            if args.use_amp:
                with torch.cuda.amp.autocast():
                    if args.output_attention:
                        outputs, ts_token = model(batch_x, batch_x_mark, dec_inp, batch_y_mark, prompt=prompts)[0]
                    else:
                        outputs, ts_token = model(batch_x, batch_x_mark, dec_inp, batch_y_mark, prompt=prompts)
            else:
                if args.output_attention:
                    outputs, ts_token = model(batch_x, batch_x_mark, dec_inp, batch_y_mark, prompt=prompts)[0]
                else:
                    outputs, ts_token = model(batch_x, batch_x_mark, dec_inp, batch_y_mark, prompt=prompts)

            ts_tokens.append(ts_token.detach().cpu().numpy())
            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:].to(DEVICE)
            pred = outputs.detach()
            true = batch_y.detach()
            loss = criterion(pred, true)
            mae_loss = mae_metric(pred, true)
            total_loss.append(loss.item())
            total_mae_loss.append(mae_loss.item())
            
            
    ts_tokens = np.concatenate(ts_tokens, axis=0)
    total_loss = np.average(total_loss)
    total_mae_loss = np.average(total_mae_loss)

    model.train()
    return total_loss, total_mae_loss, ts_tokens


def load_content(args):
    if 'ETT' in args.data:
        file = 'ETT'
    else:
        file = args.data
    with open('/home/nathan/LLM4TS/datasets/forecasting/prompt_bank/{0}.txt'.format(file), 'r') as f:
        content = f.read()
    return content