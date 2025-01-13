import numpy as np
import torch
import matplotlib.pyplot as plt
import shutil
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch.nn as nn
from data_provider.data_factory import data_provider

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
            try:
                os.makedirs(path)
            except FileExistsError:
                pass
                
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
    '''
    gt : True label
    pred: Pred label
    '''
    
    anomaly_state = False
    for i in range(len(gt)): # When sensor is 
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
            if i % 10 == 0:
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
                        outputs, _, _  = model(batch_x, batch_x_mark, dec_inp, batch_y_mark, prompt=prompts)[0]
                    else:
                        outputs, _, _  = model(batch_x, batch_x_mark, dec_inp, batch_y_mark, prompt=prompts)
            else:
                if args.output_attention:
                    outputs, _, _  = model(batch_x, batch_x_mark, dec_inp, batch_y_mark, prompt=prompts)[0]
                else:
                    outputs, _, _  = model(batch_x, batch_x_mark, dec_inp, batch_y_mark, prompt=prompts)
            
            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:].to(DEVICE)
            pred = outputs.detach()
            true = batch_y.detach()
            loss = criterion(pred, true)
            mae_loss = mae_metric(pred, true)
            total_loss.append(loss.item())
            total_mae_loss.append(mae_loss.item())
            
            # del ts_token
            
    total_loss = np.average(total_loss)
    total_mae_loss = np.average(total_mae_loss)
    model.train()
    return total_loss, total_mae_loss


def vali_forecasting_multi(args, model, vali_data, vali_loader, criterion, mae_metric, accelerator):
    total_loss = []
    total_mae_loss = []
    model.eval()
    num_batch = len(vali_loader)
    accelerator.print(f'======================================== Number Batch of Validation ------> {num_batch} ========================================')
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, prompts) in tqdm(enumerate(vali_loader), disable= not accelerator.is_main_process):
            if i % 20 == 0 and i != 0:
                torch.cuda.empty_cache()
            batch_x = batch_x.bfloat16().to(accelerator.device)
            batch_y = batch_y.bfloat16().to(accelerator.device)
            
            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(DEVICE)
            # encoder - decoder
            if args.use_amp:
                with torch.cuda.amp.autocast():
                    if args.output_attention:
                        outputs = model(batch_x, None, None, None, prompt=prompts)[0]
                    else:
                        outputs = model(batch_x, None, None, None, prompt=prompts)
            else:
                if args.output_attention:
                    outputs = model(batch_x, None, None, None, prompt=prompts)[0]
                else:
                    outputs = model(batch_x, None, None, None, prompt=prompts)
            
            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:].to(DEVICE)
            
            pred = outputs.detach()
            true = batch_y.detach()
            loss = criterion(pred, true)
            mae_loss = mae_metric(pred, true)
            loss = accelerator.gather(loss)
            mae_loss = accelerator.gather(mae_loss)
            total_loss.append(loss)
            total_mae_loss.append(mae_loss)

    total_loss = torch.cat(total_loss, dim=0).mean().item()
    total_mae_loss = torch.cat(total_mae_loss, dim=0).mean().item()
    model.train()
    return total_loss, total_mae_loss




def vali_classification(args, model, vali_data, vali_loader, criterion, accelerator):
    total_loss = []
    model.eval()
    num_batch = len(vali_loader)
    val_scores = []
    val_true = []
    y_pred = []
    accelerator.print(f'======================================== Number Batch of Validation or Tesing ------> {num_batch} ========================================')
    with torch.no_grad():
        for i, (batch_x, batch_y) in enumerate(vali_loader):
            if i % 100 == 0:
                torch.cuda.empty_cache()
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
    
    
    total_loss = accelerator.gather(total_loss)
    y_pred = accelerator.gather(y_pred)
    val_true = accelerator.gather(val_true)
    
    
    total_loss = np.average(total_loss)
    y_pred = np.stack(y_pred)
    val_true = np.stack(val_true)
    acc = accuracy_score(val_true, y_pred)
    model.train()
    return total_loss, acc


def test_forecasting_multi(args, model, test_data, test_loader, criterion, mae_metric, accelerator):
    total_loss = []
    total_mae_loss = []
    model.eval()
    num_batch = len(test_loader)
    accelerator.print(f'======================================== Number Batch of Tesing ------> {num_batch} ========================================')
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, prompts) in tqdm(enumerate(test_loader), disable=not accelerator.is_main_process):
            if i % 20 == 0 and i != 0:
                torch.cuda.empty_cache()
            batch_x = batch_x.bfloat16().to(accelerator.device)
            batch_y = batch_y.bfloat16().to(accelerator.device)
            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(DEVICE)
            # encoder - decoder
            if args.use_amp:
                with torch.cuda.amp.autocast():
                    if args.output_attention:
                        outputs = model(batch_x, None, None, None, prompt=prompts)[0]
                    else:
                        outputs = model(batch_x, None, None, None, prompt=prompts)
            else:
                if args.output_attention:
                    outputs = model(batch_x, None, None, None, prompt=prompts)[0]
                else:
                    outputs = model(batch_x, None, None, None, prompt=prompts)

            
            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:].to(DEVICE)
            pred = outputs.detach()
            true = batch_y.detach()
            loss = criterion(pred, true)
            mae_loss = mae_metric(pred, true)
            
            loss = accelerator.gather(loss)
            mae_loss = accelerator.gather(mae_loss)
            pred = accelerator.gather(pred)
            true = accelerator.gather(true)
            pred = pred.float().detach().cpu().numpy()
            true = true.float().detach().cpu().numpy()
            preds.append(pred)
            trues.append(true)
            total_loss.append(loss)
            total_mae_loss.append(mae_loss)
            
    trues = np.concatenate(trues, axis=0)
    preds = np.concatenate(preds, axis=0)
    total_loss = torch.cat(total_loss, dim=0).mean().item()
    total_mae_loss = torch.cat(total_mae_loss, dim=0).mean().item()
    
    model.train()
    return total_loss, total_mae_loss, trues, preds



def test(args, model, test_data, test_loader, criterion, mae_metric):
    total_loss = []
    total_mae_loss = []
    model.eval()
    num_batch = len(test_loader)
    pred_out = []
    true_out = []
    print(f'======================================== Number Batch of Tesing ------> {num_batch} ========================================')
    ts_tokens = []
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, prompts) in tqdm(enumerate(test_loader)):
            if i % 10 == 0:
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
                        outputs, _, _ = model(batch_x, batch_x_mark, dec_inp, batch_y_mark, prompt=prompts)[0]
                    else:
                        outputs, _, _ = model(batch_x, batch_x_mark, dec_inp, batch_y_mark, prompt=prompts)
            else:
                if args.output_attention:
                    outputs, _, _ = model(batch_x, batch_x_mark, dec_inp, batch_y_mark, prompt=prompts)[0]
                else:
                    outputs, _, _ = model(batch_x, batch_x_mark, dec_inp, batch_y_mark, prompt=prompts)


            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:].to(DEVICE)
            pred = outputs.detach()
            true = batch_y.detach()
            loss = criterion(pred, true)
            mae_loss = mae_metric(pred, true)
            
            pred_out.append(pred.detach().cpu().numpy())
            true_out.append(true.detach().cpu().numpy())
            
            total_loss.append(loss.item())
            total_mae_loss.append(mae_loss.item())
            
    pred_out = np.concatenate(pred_out, axis=0)
    true_out = np.concatenate(true_out, axis=0)
    total_loss = np.average(total_loss)
    total_mae_loss = np.average(total_mae_loss)

    
    np.save(f'./{args.llm_model}_{args.model_id}_pred.npy', pred_out)
    np.save(f'./{args.llm_model}_{args.model_id}_true.npy', true_out)
    model.train()
    return total_loss, total_mae_loss



def vali_imputation(args, model, vali_data, vali_loader, criterion, mae_metric):
    total_loss = []
    total_mae_loss = []
    model.eval()
    num_batch = len(vali_loader)
    print(f'======================================== Number Batch of Validation or Tesing ------> {num_batch} ========================================')
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, prompts) in tqdm(enumerate(vali_loader)):
            if i % 20 == 0:
                torch.cuda.empty_cache()
            batch_x = batch_x.float().to(DEVICE)
            batch_y = batch_y.float()
            batch_x_mark = batch_x_mark.float().to(DEVICE)
            batch_y_mark = batch_y_mark.float().to(DEVICE)

            B, T, N = batch_x.shape
            mask = torch.rand((B, T, N)).to(DEVICE)
            mask[mask <= args.mask_rate] = 0  # masked
            mask[mask > args.mask_rate] = 1  # remained
            inp = batch_x.masked_fill(mask == 0, 0)
            
            if args.use_amp:
                with torch.cuda.amp.autocast():
                    if args.output_attention:
                        outputs = model(batch_x, batch_x_mark, None, mask, prompt=prompts)[0]
                    else:
                        outputs = model(batch_x, batch_x_mark, None, mask, prompt=prompts)
            else:
                if args.output_attention:
                    outputs = model(batch_x, batch_x_mark, None, mask, prompt=prompts)[0]
                else:
                    outputs = model(batch_x, batch_x_mark, None, mask, prompt=prompts)

            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, :, f_dim:]
            pred = outputs[mask == 0].detach()
            true = batch_x[mask == 0].detach()
            loss = criterion(outputs[mask == 0], batch_x[mask == 0])
            mae_loss = mae_metric(pred, true)
            total_loss.append(loss.item())
            total_mae_loss.append(mae_loss.item())
            # break
    total_loss = np.average(total_loss)
    total_mae_loss = np.average(total_mae_loss)
    model.train()

    return total_loss, total_mae_loss




def vali_imputation_multigpu(args, model, vali_data, vali_loader, criterion, mae_metric, accelerator):
    total_loss = []
    total_mae_loss = []
    model.eval()
    num_batch = len(vali_loader)
    accelerator.print(f'======================================== Number Batch of Validation or Tesing ------> {num_batch} ========================================')
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, prompts) in tqdm(enumerate(vali_loader), disable=not accelerator.is_main_process):
            if i % 20 == 0:
                torch.cuda.empty_cache()
            batch_x = batch_x.float().to(DEVICE)
            batch_y = batch_y.float()
            batch_x_mark = batch_x_mark.float().to(DEVICE)
            batch_y_mark = batch_y_mark.float().to(DEVICE)

            B, T, N = batch_x.shape
            mask = torch.rand((B, T, N)).to(DEVICE)
            mask[mask <= args.mask_rate] = 0  # masked
            mask[mask > args.mask_rate] = 1  # remained
            inp = batch_x.masked_fill(mask == 0, 0)
            
            if args.use_amp:
                with torch.cuda.amp.autocast():
                    if args.output_attention:
                        outputs = model(batch_x, batch_x_mark, None, mask, prompt=prompts)[0]
                    else:
                        outputs = model(batch_x, batch_x_mark, None, mask, prompt=prompts)
            else:
                if args.output_attention:
                    outputs = model(batch_x, batch_x_mark, None, mask, prompt=prompts)[0]
                else:
                    outputs = model(batch_x, batch_x_mark, None, mask, prompt=prompts)
                    
            
            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, :, f_dim:]
            pred = outputs[mask == 0].detach()
            true = batch_x[mask == 0].detach()
            loss = criterion(outputs[mask == 0], batch_x[mask == 0])
            mae_loss = mae_metric(pred, true)
            
            loss = accelerator.gather(loss)
            mae_loss = accelerator.gather(mae_loss)
            
            total_loss.append(loss)
            total_mae_loss.append(mae_loss)
            # break
            
    total_loss = torch.cat(total_loss, dim=0).mean()
    total_mae_loss = torch.cat(total_mae_loss, dim=0).mean()
    model.train()

    return total_loss, total_mae_loss


def vali_anomaly(args, model, vali_data, vali_loader, criterion, mae_metric):
    total_loss = []
    total_mae_loss = []
    model.eval()
    num_batch = len(vali_loader)
    print(f'======================================== Number Batch of Validation ------> {num_batch} ========================================')
    with torch.no_grad():
        for i, (batch_x, batch_y, prompts) in tqdm(enumerate(vali_loader)):
            if i % 20 == 0:
                torch.cuda.empty_cache()
            batch_x = batch_x.float().to(DEVICE)
            batch_y = batch_y.float()
            B, T, N = batch_x.shape
            if args.use_amp:
                with torch.cuda.amp.autocast():
                    if args.output_attention:
                        outputs = model(batch_x, None, None, None, prompt=prompts)[0]
                    else:
                        outputs = model(batch_x, None, None, None, prompt=prompts)
            else:
                if args.output_attention:
                    outputs = model(batch_x, None, None, None, prompt=prompts)[0]
                else:
                    outputs = model(batch_x, None, None, None, prompt=prompts)
            
            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, :, f_dim:]
            loss = criterion(outputs, batch_x)
            total_loss.append(loss.item())
            
    total_loss = np.average(total_loss)
    model.train()

    return total_loss



def vali_anomaly_multi_gpu(args, model, vali_data, vali_loader, criterion, mae_metric, accelerator):
    total_loss = []
    total_mae_loss = []
    model.eval()
    num_batch = len(vali_loader)
    accelerator.print(f'======================================== Number Batch of Validation ------> {num_batch} ========================================')
    with torch.no_grad():
        for i, (batch_x, batch_y, prompts) in tqdm(enumerate(vali_loader)):
            if i % 20 == 0:
                torch.cuda.empty_cache()
                
            batch_x = batch_x.bfloat16().to(DEVICE)
            batch_y = batch_y.bfloat16()
            
            if args.use_amp:
                with torch.cuda.amp.autocast():
                    if args.output_attention:
                        outputs = model(batch_x, None, None, None, prompt=prompts)[0]
                    else:
                        outputs = model(batch_x, None, None, None, prompt=prompts)
            else:
                if args.output_attention:
                    outputs = model(batch_x, None, None, None, prompt=prompts)[0]
                else:
                    outputs = model(batch_x, None, None, None, prompt=prompts)

            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, :, f_dim:]
            loss = criterion(outputs, batch_x)
            
            loss = accelerator.gather(loss)
            total_loss.append(loss)
            
    total_loss = torch.cat(total_loss, dim=0).mean()
    model.train()
    return total_loss




def test_anomaly(args, model, path):
    total_loss = []
    total_mae_loss = []
    _, train_loader = data_provider(args, 'train')
    _, test_loader = data_provider(args, 'test')
    
    attens_energy = []
    
    # folder_path = f'./{args.task_name}_results/'
    # if not os.path.exists(folder_path):
    #     os.makedirs(folder_path)
    model.eval()
    num_batch = len(test_loader)
    anomaly_criterion = nn.MSELoss(reduce=False)
    print(f'======================================== Number Batch of Tesing ------> {num_batch} ========================================')
    with torch.no_grad():
        print('Determine training energy')
        for i, (batch_x, batch_y, prompts) in tqdm(enumerate(train_loader)):
            if i % 20 == 0:
                torch.cuda.empty_cache()

            batch_x = batch_x.float().to(DEVICE)
            batch_y = batch_y.float()

            if args.use_amp:
                with torch.cuda.amp.autocast():
                    if args.output_attention:
                        outputs = model(batch_x, None, None, None,prompts)[0]
                    else:
                        outputs = model(batch_x, None, None, None, prompts)
            else:
                if args.output_attention:
                    outputs = model(batch_x, None, None, None, prompts)[0]
                else:
                    outputs = model(batch_x, None, None, None, prompts)

            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, :, f_dim:]
            pred = outputs.detach()
            true = batch_x.detach()
            score = torch.mean(anomaly_criterion(pred, true), dim=-1)
            score = score.detach().cpu().numpy()
            attens_energy.append(score)
            
        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)
        
        
        print('Determine testing energy')
        attens_energy = []
        test_labels = []
        for i, (batch_x, batch_y, prompts) in tqdm(enumerate(test_loader)):
            if i % 20 == 0 and i != 0:
                torch.cuda.empty_cache()
                
            batch_x = batch_x.float().to(DEVICE)
            test_labels.append(batch_y)
            prompts = list(map(lambda x: x[0], prompts))
            if args.use_amp:
                with torch.cuda.amp.autocast():
                    if args.output_attention:
                        outputs = model(batch_x, None, None, None, prompts)[0]
                    else:
                        outputs = model(batch_x, None, None, None, prompts)
            else:
                if args.output_attention:
                    outputs = model(batch_x, None, None, None, prompts)[0]
                else:
                    outputs = model(batch_x, None, None, None, prompts)

            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, :, f_dim:]
            pred = outputs.detach()
            true = batch_x.detach()
            score = torch.mean(anomaly_criterion(pred, true), dim=-1)
            score = score.detach().cpu().numpy()
            attens_energy.append(score)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        threshold = np.percentile(combined_energy, 100 - args.anomaly_ratio)
        print("Threshold:", threshold)
        
        # (3) evaluation on the test set
        pred = (test_energy > threshold).astype(int)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_labels = np.array(test_labels)
        gt = test_labels.astype(int)
        
        # (4) detection adjustment
        gt, pred = adjustment(gt, pred)
        pred = np.array(pred)
        gt = np.array(gt)
        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')
        
    return accuracy, precision, recall, f_score



def test_anomaly_multi_gpu(args, model, path, accelerator):
    _, train_loader = data_provider(args, 'train')
    _, test_loader = data_provider(args, 'test')
    train_loader, test_loader = accelerator.prepare(train_loader, test_loader)
    attens_energy = []
    model.eval()

    anomaly_criterion = nn.MSELoss(reduce=False)
    with torch.no_grad():
        accelerator.print('Determine training energy')
        num_batch = len(train_loader)
        accelerator.print(f'======================================== Number Batch of Training Energy ------> {num_batch} ========================================')
        for i, (batch_x, batch_y, prompts) in tqdm(enumerate(train_loader), disable=not accelerator.is_main_process):
            if i % 20 == 0:
                torch.cuda.empty_cache()
                
            batch_x = batch_x.bfloat16().to(DEVICE)
            batch_y = batch_y.bfloat16()

            if args.use_amp:
                with torch.cuda.amp.autocast():
                    if args.output_attention:
                        outputs = model(batch_x, None, None, None,prompts)[0]
                    else:
                        outputs = model(batch_x, None, None, None, prompts)
            else:
                if args.output_attention:
                    outputs = model(batch_x, None, None, None, prompts)[0]
                else:
                    outputs = model(batch_x, None, None, None, prompts)

            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, :, f_dim:]
            pred = outputs.detach()
            true = batch_x.detach()
            score = torch.mean(anomaly_criterion(pred, true), dim=-1)
            score = accelerator.gather(score)
            attens_energy.append(score)
            
        attens_energy = torch.cat(attens_energy, dim=0).contiguous().view(-1)
        train_energy = attens_energy.float().detach().cpu().numpy()
        
        
        accelerator.print('Determine testing energy')
        attens_energy = []
        test_labels = []
        num_batch = len(test_loader)
        accelerator.print(f'======================================== Number Batch of Testing Energy ------> {num_batch} ========================================')
        for i, (batch_x, batch_y, prompts) in tqdm(enumerate(test_loader), disable = not accelerator.is_main_process):
            if i % 20 == 0 and i != 0:
                torch.cuda.empty_cache()
                
            batch_x = batch_x.bfloat16().to(DEVICE)
            
            prompts = list(map(lambda x: x[0], prompts))
            if args.use_amp:
                with torch.cuda.amp.autocast():
                    if args.output_attention:
                        outputs = model(batch_x, None, None, None, prompts)[0]
                    else:
                        outputs = model(batch_x, None, None, None, prompts)
            else:
                if args.output_attention:
                    outputs = model(batch_x, None, None, None, prompts)[0]
                else:
                    outputs = model(batch_x, None, None, None, prompts)

            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, :, f_dim:]
            pred = outputs.detach()
            true = batch_x.detach()
            score = torch.mean(anomaly_criterion(pred, true), dim=-1)
            score, batch_y = accelerator.gather((score, batch_y))
            attens_energy.append(score.float().detach().cpu())
            test_labels.append(batch_y.float().detach().cpu())
            
        attens_energy = torch.cat(attens_energy, dim=0).contiguous().view(-1)
        test_energy = attens_energy.float().detach().cpu().numpy()
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        combined_energy = torch.tensor(combined_energy)
        threshold = np.percentile(combined_energy, 100 - args.anomaly_ratio)
        print()
        accelerator.print("Threshold:", threshold)
        
        # (3) evaluation on the test set
        pred = (test_energy > threshold).astype(int)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_labels = np.array(test_labels)
        gt = test_labels.astype(int)
        
        gt, pred = adjustment(gt, pred)
        pred = np.array(pred)
        gt = np.array(gt)
        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')
        
    return accuracy, precision, recall, f_score


def load_content(args):
    if 'ETT' in args.data:
        file = 'ETT'
    else:
        file = args.data
    with open('./prompt_bank/{0}.txt'.format(file), 'r') as f:
        content = f.read()
    return content