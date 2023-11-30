import os
import copy
import torch
import torch_geometric.transforms as T

from torch_geometric.transforms import Polar
from torch_geometric.data import Data
from torch_geometric.data import Dataset

import numpy as np
from torch.utils.data.sampler import Sampler
from sklearn.model_selection import train_test_split, StratifiedKFold
from tqdm import tqdm

class Sampler_custom(Sampler):

    def __init__(self, event_list, censor_list, batch_size):
        self.event_list = event_list
        self.censor_list = censor_list
        self.batch_size = batch_size

    def __iter__(self):

        train_batch_sampler = []
        Event_idx = copy.deepcopy(self.event_list)
        Censored_idx = copy.deepcopy(self.censor_list)
        np.random.shuffle(Event_idx)
        np.random.shuffle(Censored_idx)

        Int_event_batch_num = Event_idx.shape[0] // 2
        Int_event_batch_num = Int_event_batch_num * 2
        Event_idx_batch_select = np.random.choice(Event_idx.shape[0], Int_event_batch_num, replace=False)
        Event_idx = Event_idx[Event_idx_batch_select]

        Int_censor_batch_num = Censored_idx.shape[0] // (self.batch_size - 2)
        Int_censor_batch_num = Int_censor_batch_num * (self.batch_size - 2)
        Censored_idx_batch_select = np.random.choice(Censored_idx.shape[0], Int_censor_batch_num, replace=False)
        Censored_idx = Censored_idx[Censored_idx_batch_select]

        Event_idx_selected = np.random.choice(Event_idx, size=(len(Event_idx) // 2, 2), replace=False)
        Censored_idx_selected = np.random.choice(Censored_idx, size=(
            (Censored_idx.shape[0] // (self.batch_size - 2)), (self.batch_size - 2)), replace=False)

        if Event_idx_selected.shape[0] > Censored_idx_selected.shape[0]:
            Event_idx_selected = Event_idx_selected[:Censored_idx_selected.shape[0],:]
        else:
            Censored_idx_selected = Censored_idx_selected[:Event_idx_selected.shape[0],:]

        for c in range(Event_idx_selected.shape[0]):
            train_batch_sampler.append(
                Event_idx_selected[c, :].flatten().tolist() + Censored_idx_selected[c, :].flatten().tolist())

        return iter(train_batch_sampler)

    def __len__(self):
        return len(self.event_list) // 2

class CoxGraphDataset(Dataset):

    def __init__(self, df, root_dir, feature_size):
        super(CoxGraphDataset, self).__init__()
        self.df = df
        self.root_dir = root_dir
        self.polar_transform = Polar()
        self.feature_size = feature_size

    def processed_file_names(self):
        return self.filelist

    def len(self):
        return len(self.df)

    def get(self, idx):
        file_name = os.path.join(self.root_dir, self.df.iloc[idx].File)
        data_origin = torch.load(file_name)
        transfer = T.ToSparseTensor()
        item = file_name.split('/')[-1].split('.pt')[0].split('_')[0]

        survival = self.df.iloc[idx].Survival
        phase = self.df.iloc[idx].Censor

        data_re = Data(x=data_origin.x[:,:self.feature_size], edge_index=data_origin.edge_index)
        mock_data = Data(x=data_origin.x[:,:self.feature_size], edge_index=data_origin.edge_index, pos=data_origin.pos)

        data_re.pos = data_origin.pos
        data_re_polar = self.polar_transform(mock_data)
        polar_edge_attr = data_re_polar.edge_attr

        if (data_re.edge_index.shape[1] != data_origin.edge_attr.shape[0]):
            print('error!')
            print(self.filelist[idx].split('/')[-1])
        else:
            data = transfer(data_re)
            data.survival = torch.tensor(survival)
            data.phase = torch.tensor(phase)
            data.item = item
            data.edge_attr = polar_edge_attr
            data.pos = data_origin.pos

        return data

def nnet_pred_surv(y_pred, breaks, fu_time):
  y_pred=np.cumprod(y_pred, axis=1)
  pred_surv = []
  for i in range(y_pred.shape[0]):
    pred_surv.append(np.interp(fu_time,breaks[1:],y_pred[i,:]))
  return np.array(pred_surv)

def make_surv_array(t,f,breaks):
  n_samples=t.shape[0]
  n_intervals=len(breaks)-1
  timegap = breaks[1:] - breaks[:-1]
  breaks_midpoint = breaks[:-1] + 0.5*timegap
  y_train = np.zeros((n_samples,n_intervals*2))

  for i in range(n_samples):
    if f[i]:
      y_train[i,0:n_intervals] = 1.0*(t[i]>=breaks[1:])
      if t[i]<breaks[-1]:
        y_train[i,n_intervals+np.where(t[i]<breaks[1:])[0][0]]=1
    else: #if censored
      y_train[i,0:n_intervals] = 1.0*(t[i]>=breaks_midpoint)
  return y_train

def make_label(t):
    n_samples = t.shape[0]
    max_time = max(t)
    time_break = np.arange(0, max_time, (max_time)/20)
    y_labels = np.zeros((n_samples, ))
    for i in range(n_samples):
        y_labels[i] = int(np.where(t[i] > time_break)[0][-1])
    return y_labels.tolist(), time_break

class SlidePatchDataset():
    def __init__(self, image, x, y, transform):
        self.image = image
        self.x = x
        self.y = y
        self.transform = transform

    def __len__(self):
        return len((self.image))

    def __getitem__(self, idx):

        image = self.image[idx]
        x = self.x[idx]
        y = self.y[idx]
        image = image.convert('RGB')
        R = self.transform(image)

        sample = {'image': R, 'X': torch.tensor(x), 'Y': torch.tensor(y)}

        return sample

def add_kfold_to_df(df, n_fold, seed):
    skf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=seed)
    for fold, (_, val_) in enumerate(skf.split(X=df, y=df.Censor)):
        df.loc[df.index[val_], "kfold"] = int(fold)

    df['kfold'] = df['kfold'].astype(int)

def metadata_list_generation(DatasetType, Trainlist, Metadata):
    Train_survivallist = []
    Train_censorlist = []
    Train_stagelist = []
    exclude_list = []
    patient_list = []

    if "TCGA" in DatasetType:
        with tqdm(total=len(Trainlist)) as tbar:
            for idx in range(len(Trainlist)):
                item = '-'.join(Trainlist[idx].split('/')[-1].split('.pt')[0].split('_')[0].split('-')[0:3])
                Match_item = Metadata[Metadata["case_submitter_id"] == item]
                if Match_item.shape[0] != 0:
                    if Match_item['vital_status'].tolist()[0] == "Alive":
                        if '--' not in Match_item['days_to_last_follow_up'].tolist()[0]:
                            if '--' not in Match_item['ajcc_pathologic_stage'].tolist()[0]:
                                Train_censorlist.append(0)
                                Train_survivallist.append(
                                    int(float(Match_item['days_to_last_follow_up'].tolist()[0])))
                                if ('IV' or 'X') in Match_item['ajcc_pathologic_stage'].tolist()[0]:
                                    Train_stagelist.append(4)
                                elif "III" in Match_item['ajcc_pathologic_stage'].tolist()[0]:
                                    Train_stagelist.append(3)
                                elif "II" in Match_item['ajcc_pathologic_stage'].tolist()[0]:
                                    Train_stagelist.append(2)
                                elif "I" in Match_item['ajcc_pathologic_stage'].tolist()[0]:
                                    Train_stagelist.append(1)
                                else:
                                    Train_stagelist.append(0)
                            else:
                                exclude_list.append(idx)
                        else:
                            exclude_list.append(idx)
                    else:
                        if '--' not in Match_item['days_to_death'].tolist()[0]:
                            if '--' not in Match_item['ajcc_pathologic_stage'].tolist()[0]:
                                Train_censorlist.append(1)
                                Train_survivallist.append(int(float(Match_item['days_to_death'].tolist()[0])))

                                if ('IV' or 'X') in Match_item['ajcc_pathologic_stage'].tolist()[0]:
                                    Train_stagelist.append(4)
                                elif "III" in Match_item['ajcc_pathologic_stage'].tolist()[0]:
                                    Train_stagelist.append(3)
                                elif "II" in Match_item['ajcc_pathologic_stage'].tolist()[0]:
                                    Train_stagelist.append(2)
                                elif "I" in Match_item['ajcc_pathologic_stage'].tolist()[0]:
                                    Train_stagelist.append(1)
                                else:
                                    Train_stagelist.append(0)
                            else:
                                exclude_list.append(idx)
                        else:
                            exclude_list.append(idx)
                else:
                    exclude_list.append(idx)
        _ = [Trainlist.pop(idx_item - c) for c, idx_item in enumerate(exclude_list)]

    return Trainlist, Train_survivallist, Train_censorlist, Train_stagelist