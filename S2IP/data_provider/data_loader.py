import os
import numpy as np
import pandas as pd
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
from statsmodels.tsa.seasonal import STL
import warnings
from tqdm import tqdm
from joblib import Parallel, delayed
from data_provider.uea import subsample, interpolate_missing, Normalizer
from sktime.datasets import load_from_tsfile_to_dataframe
from einops import rearrange


warnings.filterwarnings('ignore')
torch.set_default_dtype(torch.float32)

class Dataset_ETT_hour(Dataset):
    def __init__(self, configs, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', 
                 seasonal_patterns=None, percent=10):
        # size [seq_len, label_len, pred_len]
        # info
        self.configs = configs
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.percent = percent
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()
        
    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        if self.configs.task_name != 'imputation':
            s_begin = index
            s_end = s_begin + self.seq_len
            r_begin = s_end - self.label_len
            r_end = r_begin + self.label_len + self.pred_len
            seq_x = self.data_x[s_begin:s_end]
            seq_y = self.data_y[r_begin:r_end]
            seq_x_mark = self.data_stamp[s_begin:s_end]
            seq_y_mark = self.data_stamp[r_begin:r_end]
            
            means = np.mean(seq_x, axis=0)
            seq_x = seq_x - means
            stdev = np.std(seq_x, axis=0) + 1e-5
            seq_x /= stdev
            
            decompose = self.decompose(seq_x)
            seq_x = decompose
            return seq_x, seq_y, seq_x_mark, seq_y_mark, means, stdev
        
        elif self.configs.task_name == 'imputation':
            s_begin = index
            s_end = s_begin + self.seq_len
            r_begin = s_end - self.label_len
            r_end = r_begin + self.label_len + self.pred_len
            seq_x = self.data_x[s_begin:s_end]
            seq_y = self.data_y[r_begin:r_end]
            seq_x_mark = self.data_stamp[s_begin:s_end]
            seq_y_mark = self.data_stamp[r_begin:r_end]
            seq_x = torch.tensor(seq_x)
            T, N = seq_x.shape
            imput_mask = torch.rand((T, N))
            imput_mask[imput_mask <= self.configs.mask_rate] = 0  # masked
            imput_mask[imput_mask > self.configs.mask_rate] = 1  # remained
            
            x_enc = seq_x.masked_fill(imput_mask == 0, 0)

            means = torch.sum(x_enc, dim=0) / torch.sum(imput_mask == 1, dim=0)
            means = means.unsqueeze(0).detach()
            x_enc = x_enc - means
            x_enc = x_enc.masked_fill(imput_mask == 0, 0)
            stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=0) /
                                torch.sum(imput_mask == 1, dim=0) + 1e-5)
            stdev = stdev.unsqueeze(0).detach()
            x_enc /= stdev
            decompose = self.decompose(seq_x)
            seq_x = decompose
            return seq_x, seq_y, seq_x_mark, seq_y_mark, means, stdev, imput_mask


    def decompose(self, x):
        df = pd.DataFrame(x)
        trend = df.rolling(window=self.configs.trend_length, center=True).mean().fillna(method='bfill').fillna(method='ffill')
        detrended = df - trend
        seasonal = detrended.groupby(detrended.index % self.configs.seasonal_length).transform('mean').fillna(method='bfill').fillna(method='ffill') 
        residuals = df - trend - seasonal
        combined = np.stack([trend, seasonal, residuals], axis=1)
        return combined

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_hour_decomposed(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', 
                 seasonal_patterns=None, percent=10):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.percent = percent
        self.root_path = root_path
        self.data_path = data_path
        self.decomposed_cache = {}
        self.__read_data__()
        
        
    def stl_decomposition(self, series, period,variate_name, start_timestamp, end_timestamp):

        cache_key = (start_timestamp, end_timestamp, period, variate_name)
        if cache_key in self.decomposed_cache:
            return self.decomposed_cache[cache_key]

        
        stl = STL(series, period=period)
        result = stl.fit()
        trend = result.trend
        seasonal = result.seasonal
        resid = result.resid
        self.decomposed_cache[cache_key] = (trend, seasonal, resid)
        return trend, seasonal, resid

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len
        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data.iloc[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
        
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.time_mark = df_raw['date'].values[border1:border2]
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        concatenated_result = np.empty((seq_x.shape[0], 0))

        start_timestamp = self.time_mark[s_begin]
        end_timestamp = self.time_mark[s_end]

        for variate in range(seq_x.shape[1]):
            series = pd.Series(seq_x[:, variate])
            trend, seasonal, resid = self.stl_decomposition(series, 24, variate, start_timestamp , end_timestamp)
            trend_array = trend.to_numpy().reshape(-1, 1)
            seasonal_array = seasonal.to_numpy().reshape(-1, 1)
            resid_array = resid.to_numpy().reshape(-1, 1)

            variate_components = np.concatenate([trend_array, seasonal_array, resid_array], axis=1)
            concatenated_result = np.concatenate([concatenated_result, variate_components], axis=1)
        
        seq_x_expanded = seq_x if len(seq_x.shape) > 1 else seq_x.reshape(-1, 1)
        final_result = np.concatenate([seq_x_expanded, concatenated_result], axis=1)


        return final_result, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, configs, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t', 
                 seasonal_patterns=None, percent=10):
        # size [seq_len, label_len, pred_len]
        # info
        self.configs = configs
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.percent = percent
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        if self.configs.task_name != 'imputation':
                s_begin = index
                s_end = s_begin + self.seq_len
                r_begin = s_end - self.label_len
                r_end = r_begin + self.label_len + self.pred_len

                seq_x = self.data_x[s_begin:s_end]
                seq_y = self.data_y[r_begin:r_end]
                seq_x_mark = self.data_stamp[s_begin:s_end]
                seq_y_mark = self.data_stamp[r_begin:r_end]
                
                means = np.mean(seq_x, axis=0)
                seq_x = seq_x - means
                stdev = np.std(seq_x, axis=0) + 1e-5
                seq_x /= stdev
                
                decompose = self.decompose(seq_x)
                seq_x = decompose
                return seq_x, seq_y, seq_x_mark, seq_y_mark, means, stdev
            
        elif self.configs.task_name == 'imputation':
            s_begin = index
            s_end = s_begin + self.seq_len
            r_begin = s_end - self.label_len
            r_end = r_begin + self.label_len + self.pred_len
            seq_x = self.data_x[s_begin:s_end]
            seq_y = self.data_y[r_begin:r_end]
            seq_x_mark = self.data_stamp[s_begin:s_end]
            seq_y_mark = self.data_stamp[r_begin:r_end]
            seq_x = torch.tensor(seq_x)
            T, N = seq_x.shape
            imput_mask = torch.rand((T, N))
            imput_mask[imput_mask <= self.configs.mask_rate] = 0  # masked
            imput_mask[imput_mask > self.configs.mask_rate] = 1  # remained
            
            x_enc = seq_x.masked_fill(imput_mask == 0, 0)

            means = torch.sum(x_enc, dim=0) / torch.sum(imput_mask == 1, dim=0)
            means = means.unsqueeze(0).detach()
            x_enc = x_enc - means
            x_enc = x_enc.masked_fill(imput_mask == 0, 0)
            stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=0) /
                                torch.sum(imput_mask == 1, dim=0) + 1e-5)
            stdev = stdev.unsqueeze(0).detach()
            x_enc /= stdev
            decompose = self.decompose(seq_x)
            seq_x = decompose
            return seq_x, seq_y, seq_x_mark, seq_y_mark, means, stdev, imput_mask


    def decompose(self, x):
        df = pd.DataFrame(x)
        trend = df.rolling(window=self.configs.trend_length, center=True).mean().fillna(method='bfill').fillna(method='ffill')
        detrended = df - trend
        seasonal = detrended.groupby(detrended.index % self.configs.seasonal_length).transform('mean').fillna(method='bfill').fillna(method='ffill') 
        residuals = df - trend - seasonal
        combined = np.stack([trend, seasonal, residuals], axis=1)
        return combined


    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    




class Dataset_ETT_minute_decomposed(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t', 
                 seasonal_patterns=None, percent=10):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.percent = percent
        self.root_path = root_path
        self.data_path = data_path
        self.decomposed_cache = {}


        self.__read_data__()
    
    
    def stl_decomposition(self, series, period,variate_name, start_timestamp, end_timestamp):

        cache_key = (start_timestamp, end_timestamp, period, variate_name)
        if cache_key in self.decomposed_cache:
            return self.decomposed_cache[cache_key]

        

        stl = STL(series, period=period)
        result = stl.fit()
        trend = result.trend
        seasonal = result.seasonal
        resid = result.resid
        self.decomposed_cache[cache_key] = (trend, seasonal, resid)

        return trend, seasonal, resid
     

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        
        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        

      
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len
        
        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data.iloc[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.time_mark = df_raw['date'].values[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        concatenated_result = np.empty((seq_x.shape[0], 0))

        start_timestamp = self.time_mark[s_begin]
        end_timestamp = self.time_mark[s_end]

        for variate in range(seq_x.shape[1]):
            series = pd.Series(seq_x[:, variate])
            trend, seasonal, resid = self.stl_decomposition(series, 96, variate, start_timestamp , end_timestamp)
            trend_array = trend.to_numpy().reshape(-1, 1)
            seasonal_array = seasonal.to_numpy().reshape(-1, 1)
            resid_array = resid.to_numpy().reshape(-1, 1)

            variate_components = np.concatenate([trend_array, seasonal_array, resid_array], axis=1)
            concatenated_result = np.concatenate([concatenated_result, variate_components], axis=1)
        
        seq_x_expanded = seq_x if len(seq_x.shape) > 1 else seq_x.reshape(-1, 1)
        final_result = np.concatenate([seq_x_expanded, concatenated_result], axis=1)


        return final_result, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)



class Dataset_Custom(Dataset):
    def __init__(self, configs, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h',
                 percent=10, max_len=-1, train_all=False,seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        # info
        self.configs = configs
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.percent = percent

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()
        
        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        # print(cols)
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        if self.configs.task_name != 'imputation':
            feat_id = index // self.tot_len
            s_begin = index % self.tot_len
            
            s_end = s_begin + self.seq_len
            r_begin = s_end - self.label_len
            r_end = r_begin + self.label_len + self.pred_len
            seq_x = self.data_x[s_begin:s_end, feat_id:feat_id+1]
            seq_y = self.data_y[r_begin:r_end, feat_id:feat_id+1]
            seq_x_mark = self.data_stamp[s_begin:s_end]
            seq_y_mark = self.data_stamp[r_begin:r_end]
            means = np.mean(seq_x, axis=0)
            seq_x = seq_x - means
            stdev = np.std(seq_x, axis=0) + 1e-5
            seq_x /= stdev
            
            decompose = self.decompose(seq_x)
            seq_x = decompose
            return seq_x, seq_y, seq_x_mark, seq_y_mark, means, stdev
        
        elif self.configs.task_name == 'imputation':
            feat_id = index // self.tot_len
            s_begin = index % self.tot_len
            
            s_end = s_begin + self.seq_len
            r_begin = s_end - self.label_len
            r_end = r_begin + self.label_len + self.pred_len
            seq_x = self.data_x[s_begin:s_end, feat_id:feat_id+1]
            seq_y = self.data_y[r_begin:r_end, feat_id:feat_id+1]
            seq_x_mark = self.data_stamp[s_begin:s_end]
            seq_y_mark = self.data_stamp[r_begin:r_end]
            seq_x = torch.tensor(seq_x)
            T, N = seq_x.shape
            
            imput_mask = torch.rand((T, N))
            imput_mask[imput_mask <= self.configs.mask_rate] = 0  # masked
            imput_mask[imput_mask > self.configs.mask_rate] = 1  # remained
            
            x_enc = seq_x.masked_fill(imput_mask == 0, 0)

            means = torch.sum(x_enc, dim=0) / torch.sum(imput_mask == 1, dim=0)
            means = means.unsqueeze(0).detach()
            x_enc = x_enc - means
            x_enc = x_enc.masked_fill(imput_mask == 0, 0)
            stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=0) /
                                torch.sum(imput_mask == 1, dim=0) + 1e-5)
            stdev = stdev.unsqueeze(0).detach()
            x_enc /= stdev
            decompose = self.decompose(seq_x)
            seq_x = decompose
            return seq_x, seq_y, seq_x_mark, seq_y_mark, means, stdev, imput_mask


    def decompose(self, x):
        df = pd.DataFrame(x)
        trend = df.rolling(window=self.configs.trend_length, center=True).mean().fillna(method='bfill').fillna(method='ffill')
        detrended = df - trend
        seasonal = detrended.groupby(detrended.index % self.configs.seasonal_length).transform('mean').fillna(method='bfill').fillna(method='ffill') 
        residuals = df - trend - seasonal
        combined = np.stack([trend, seasonal, residuals], axis=1)
        return combined
    
    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)




class PSMSegLoader(Dataset):
    def __init__(self, configs, root_path, win_size, step=1, flag="train", percent=100):
        self.configs = configs
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = pd.read_csv(os.path.join(root_path, 'train.csv'))
        data = data.values[:, 1:]
        data = np.nan_to_num(data)
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = pd.read_csv(os.path.join(root_path, 'test.csv'))
        test_data = test_data.values[:, 1:]
        test_data = np.nan_to_num(test_data)
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = pd.read_csv(os.path.join(root_path, 'test_label.csv')).values[:, 1:]
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            seq_x, label = np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            seq_x, label = np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            seq_x, label = np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            seq_x, label = np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])
        mean = np.mean(seq_x, axis=0)
        seq_x = seq_x - mean
        stdev = np.std(seq_x, axis=0) + 1e-5
        seq_x /= stdev
        seq_x = self.decompose(seq_x)
        
        return seq_x, label, mean, stdev
    
    def decompose(self, x):
        df = pd.DataFrame(x)
        trend = df.rolling(window=self.configs.trend_length, center=True).mean().fillna(method='bfill').fillna(method='ffill')
        detrended = df - trend
        seasonal = detrended.groupby(detrended.index % self.configs.seasonal_length).transform('mean').fillna(method='bfill').fillna(method='ffill') 
        residuals = df - trend - seasonal
        combined = np.stack([trend, seasonal, residuals], axis=1)
        return combined


class MSLSegLoader(Dataset):
    def __init__(self, configs, root_path, win_size, step=1, flag="train", percent=100):
        self.configs = configs
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "MSL_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "MSL_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(os.path.join(root_path, "MSL_test_label.npy"))
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            seq_x, label = np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            seq_x, label = np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            seq_x, label = np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            seq_x, label = np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])

        mean = np.mean(seq_x, axis=0)
        seq_x = seq_x - mean
        stdev = np.std(seq_x, axis=0) + 1e-5
        seq_x /= stdev
        seq_x = self.decompose(seq_x)
        
        return seq_x, label, mean, stdev
    
    def decompose(self, x):
        df = pd.DataFrame(x)
        trend = df.rolling(window=self.configs.trend_length, center=True).mean().fillna(method='bfill').fillna(method='ffill')
        detrended = df - trend
        seasonal = detrended.groupby(detrended.index % self.configs.seasonal_length).transform('mean').fillna(method='bfill').fillna(method='ffill') 
        residuals = df - trend - seasonal
        combined = np.stack([trend, seasonal, residuals], axis=1)
        return combined
    

class SMAPSegLoader(Dataset):
    def __init__(self, configs, root_path, win_size, step=1, flag="train", percent=100):
        self.configs = configs
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "SMAP_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "SMAP_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(os.path.join(root_path, "SMAP_test_label.npy"))
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            seq_x, label = np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            seq_x, label = np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            seq_x, label = np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            seq_x, label = np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])

        mean = np.mean(seq_x, axis=0)
        seq_x = seq_x - mean
        stdev = np.std(seq_x, axis=0) + 1e-5
        seq_x /= stdev
        seq_x = self.decompose(seq_x)
        
        return seq_x, label, mean, stdev
    
    def decompose(self, x):
        df = pd.DataFrame(x)
        trend = df.rolling(window=self.configs.trend_length, center=True).mean().fillna(method='bfill').fillna(method='ffill')
        detrended = df - trend
        seasonal = detrended.groupby(detrended.index % self.configs.seasonal_length).transform('mean').fillna(method='bfill').fillna(method='ffill') 
        residuals = df - trend - seasonal
        combined = np.stack([trend, seasonal, residuals], axis=1)
        return combined
    

class SMDSegLoader(Dataset):
    def __init__(self, configs, root_path, win_size, step=100, flag="train", percent=100):
        self.configs = configs
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "SMD_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "SMD_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(os.path.join(root_path, "SMD_test_label.npy"))

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):

            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            seq_x, label = np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            seq_x, label = np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            seq_x, label = np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            seq_x, label = np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])

        mean = np.mean(seq_x, axis=0)
        seq_x = seq_x - mean
        stdev = np.std(seq_x, axis=0) + 1e-5
        seq_x /= stdev
        seq_x = self.decompose(seq_x)
        
        return seq_x, label, mean, stdev
    
    def decompose(self, x):
        df = pd.DataFrame(x)
        trend = df.rolling(window=self.configs.trend_length, center=True).mean().fillna(method='bfill').fillna(method='ffill')
        detrended = df - trend
        seasonal = detrended.groupby(detrended.index % self.configs.seasonal_length).transform('mean').fillna(method='bfill').fillna(method='ffill') 
        residuals = df - trend - seasonal
        combined = np.stack([trend, seasonal, residuals], axis=1)
        return combined

class SWATSegLoader(Dataset):
    def __init__(self, configs, root_path, win_size, step=1, flag="train", percent=100):
        self.configs = configs
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        train_data = pd.read_csv(os.path.join(root_path, 'swat_train2.csv'))
        test_data = pd.read_csv(os.path.join(root_path, 'swat2.csv'))
        labels = test_data.values[:, -1:]
        train_data = train_data.values[:, :-1]
        test_data = test_data.values[:, :-1]

        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        test_data = self.scaler.transform(test_data)
        self.train = train_data
        self.test = test_data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = labels
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            seq_x, label = np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            seq_x, label = np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            seq_x, label = np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            seq_x, label = np.float32(self.test[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])
        
        mean = np.mean(seq_x, axis=0)
        seq_x = seq_x - mean
        stdev = np.std(seq_x, axis=0) + 1e-5
        seq_x /= stdev
        seq_x = self.decompose(seq_x)
        
        return seq_x, label, mean, stdev
    
    def decompose(self, x):
        df = pd.DataFrame(x)
        trend = df.rolling(window=self.configs.trend_length, center=True).mean().fillna(method='bfill').fillna(method='ffill')
        detrended = df - trend
        seasonal = detrended.groupby(detrended.index % self.configs.seasonal_length).transform('mean').fillna(method='bfill').fillna(method='ffill') 
        residuals = df - trend - seasonal
        combined = np.stack([trend, seasonal, residuals], axis=1)
        return combined



class UEAloader(Dataset):
    """
    Dataset class for datasets included in:
        Time Series Classification Archive (www.timeseriesclassification.com)
    Argument:
        limit_size: float in (0, 1) for debug
    Attributes:
        all_df: (num_samples * seq_len, num_columns) dataframe indexed by integer indices, with multiple rows corresponding to the same index (sample).
            Each row is a time step; Each column contains either metadata (e.g. timestamp) or a feature.
        feature_df: (num_samples * seq_len, feat_dim) dataframe; contains the subset of columns of `all_df` which correspond to selected features
        feature_names: names of columns contained in `feature_df` (same as feature_df.columns)
        all_IDs: (num_samples,) series of IDs contained in `all_df`/`feature_df` (same as all_df.index.unique() )
        labels_df: (num_samples, num_labels) pd.DataFrame of label(s) for each sample
        max_seq_len: maximum sequence (time series) length. If None, script argument `max_seq_len` will be used.
            (Moreover, script argument overrides this attribute)
    """

    def __init__(self, configs, root_path, file_list=None, limit_size=None, flag=None, percent=100):
        self.root_path = root_path
        self.all_df, self.labels_df = self.load_all(root_path, file_list=file_list, flag=flag)
        self.all_IDs = self.all_df.index.unique()  # all sample IDs (integer indices 0 ... num_samples-1)
        self.configs = configs
        if limit_size is not None:
            if limit_size > 1:
                limit_size = int(limit_size)
            else:  # interpret as proportion if in (0, 1]
                limit_size = int(limit_size * len(self.all_IDs))
            self.all_IDs = self.all_IDs[:limit_size]
            self.all_df = self.all_df.loc[self.all_IDs]

        # use all features
        self.feature_names = self.all_df.columns
        self.feature_df = self.all_df

        # pre_process
        normalizer = Normalizer()
        self.feature_df = normalizer.normalize(self.feature_df)
        print(len(self.all_IDs))

    def load_all(self, root_path, file_list=None, flag=None):
        """
        Loads datasets from csv files contained in `root_path` into a dataframe, optionally choosing from `pattern`
        Args:
            root_path: directory containing all individual .csv files
            file_list: optionally, provide a list of file paths within `root_path` to consider.
                Otherwise, entire `root_path` contents will be used.
        Returns:
            all_df: a single (possibly concatenated) dataframe with all data corresponding to specified files
            labels_df: dataframe containing label(s) for each sample
        """
        # Select paths for training and evaluation
        if file_list is None:
            data_paths = glob.glob(os.path.join(root_path, '*'))  # list of all paths
        else:
            data_paths = [os.path.join(root_path, p) for p in file_list]
        if len(data_paths) == 0:
            raise Exception('No files found using: {}'.format(os.path.join(root_path, '*')))
        if flag is not None:
            data_paths = list(filter(lambda x: re.search(flag, x), data_paths))
        input_paths = [p for p in data_paths if os.path.isfile(p) and p.endswith('.ts')]
        if len(input_paths) == 0:
            pattern='*.ts'
            raise Exception("No .ts files found using pattern: '{}'".format(pattern))

        all_df, labels_df = self.load_single(input_paths[0])  # a single file contains dataset

        return all_df, labels_df

    def load_single(self, filepath):
        df, labels = load_from_tsfile_to_dataframe(filepath, return_separate_X_and_y=True,
                                                             replace_missing_vals_with='NaN')
        labels = pd.Series(labels, dtype="category")
        self.class_names = labels.cat.categories
        labels_df = pd.DataFrame(labels.cat.codes,
                                 dtype=np.int8)  # int8-32 gives an error when using nn.CrossEntropyLoss

        lengths = df.applymap(
            lambda x: len(x)).values  # (num_samples, num_dimensions) array containing the length of each series

        horiz_diffs = np.abs(lengths - np.expand_dims(lengths[:, 0], -1))

        if np.sum(horiz_diffs) > 0:  # if any row (sample) has varying length across dimensions
            df = df.applymap(subsample)

        lengths = df.applymap(lambda x: len(x)).values
        vert_diffs = np.abs(lengths - np.expand_dims(lengths[0, :], 0))
        if np.sum(vert_diffs) > 0:  # if any column (dimension) has varying length across samples
            self.max_seq_len = int(np.max(lengths[:, 0]))
        else:
            self.max_seq_len = lengths[0, 0]

        # First create a (seq_len, feat_dim) dataframe for each sample, indexed by a single integer ("ID" of the sample)
        # Then concatenate into a (num_samples * seq_len, feat_dim) dataframe, with multiple rows corresponding to the
        # sample index (i.e. the same scheme as all datasets in this project)

        df = pd.concat((pd.DataFrame({col: df.loc[row, col] for col in df.columns}).reset_index(drop=True).set_index(
            pd.Series(lengths[row, 0] * [row])) for row in range(df.shape[0])), axis=0)

        # Replace NaN values
        grp = df.groupby(by=df.index)
        df = grp.transform(interpolate_missing)

        return df, labels_df

    def instance_norm(self, case):
        # if self.root_path.count('EthanolConcentration') > 0:  # special process for numerical stability
        mean = torch.mean(case, axis=0)
        case = case - mean
        stdev = torch.std(case, axis=0) + 1e-5
        case /= stdev
        return case, mean, stdev

    def __getitem__(self, ind):
        seq_x = torch.from_numpy(self.feature_df.loc[self.all_IDs[ind]].values)
        label = torch.from_numpy(self.labels_df.loc[self.all_IDs[ind]].values)
        # print(seq_x.shape)
        seq_x, mean, stdev = self.instance_norm(seq_x)
        seq_x = self.decompose(seq_x)
        
        return seq_x, label, mean, stdev
    
    
    def decompose(self, x):
        df = pd.DataFrame(x)
        trend = df.rolling(window=self.configs.trend_length, center=True).mean().fillna(method='bfill').fillna(method='ffill')
        detrended = df - trend
        seasonal = detrended.groupby(detrended.index % self.configs.seasonal_length).transform('mean').fillna(method='bfill').fillna(method='ffill') 
        residuals = df - trend - seasonal
        combined = np.stack([trend, seasonal, residuals], axis=1)
        return combined

    def __len__(self):
        return len(self.all_IDs)







