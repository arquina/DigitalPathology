#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 09:58:47 2021

@author: taliq
"""
import os
import torch

from torch_geometric.data import DataLoader
from DataLoader import CoxGraphDataset

import pandas as pd
from train_utils import model_selection
import numpy as np
from tqdm import tqdm
from DataLoader import metadata_list_generation
from Interpretation import calculate_feature_IG, Calculate_feature_attention
from Interpretation import Visualize_IG, whole_IG_normalize, ig_normalization


def IG_value_calculation(Argument):

    save_dir = os.path.join('/'.join(Argument.analysis_path.split('/')[:-1]), 'IG_analysis')
    if not os.path.isdir(save_dir):
       os.mkdir(save_dir)
    ig_value_dir = os.path.join(save_dir, 'IG_value')
    whole_feature_dir = os.path.join(save_dir, 'Graph_feature')
    attention_dir = os.path.join(save_dir, 'Attention')
    if os.path.exists(ig_value_dir) is False:
        os.mkdir(ig_value_dir)
        os.mkdir(whole_feature_dir)
        os.mkdir(attention_dir)

    batch_num = Argument.batch_size
    device = torch.device(int(Argument.gpu))
    metadata_root = os.path.join(Argument.meta_root, Argument.DatasetType, Argument.CancerType)
    Metadata = pd.read_csv(os.path.join(metadata_root, Argument.CancerType + '_clinical.tsv'), sep='\t')

    TrainRoot = os.path.join(Argument.graph_root, Argument.DatasetType, Argument.CancerType, Argument.magnification,
                             Argument.patch_size, Argument.pretrain, 'graph', 'pt_files')
    Trainlist = os.listdir(TrainRoot)
    Trainlist, Train_survivallist, Train_censorlist, Train_stagelist = metadata_list_generation(Argument.DatasetType,
                                                                                                Trainlist, Metadata)
    train_df = pd.DataFrame(zip(Trainlist, Train_survivallist, Train_censorlist, Train_stagelist), columns = ['File', 'Survival', 'Censor', 'Stage'])
    TestDataset = CoxGraphDataset(df=train_df, root_dir=TrainRoot, feature_size=Argument.feature_size)
    test_loader = DataLoader(TestDataset, batch_size=batch_num, shuffle=False, num_workers=8, pin_memory=False, drop_last=False)

    if Argument.loss == 'CoxPH':
        Argument.n_intervals = 1

    model = model_selection(Argument)

    Temp_state_dict = torch.load(Argument.analysis_path, map_location="cpu")
    Temp_state_dict_list = list(Temp_state_dict.keys())
    for item in Temp_state_dict_list:
        Temp_state_dict[item.split('module.')[1]] = Temp_state_dict[item]
        del Temp_state_dict[item]

    model.load_state_dict(Temp_state_dict)
    model = model.to(torch.device(Argument.gpu))
    model.eval()

    with tqdm(total=len(test_loader)) as pbar:
        for c,d in enumerate(test_loader,1):
            d = d.to(torch.device(Argument.gpu))
            method = "ig"
            batch_count = np.unique(d.batch.cpu().detach().numpy(), return_counts=True)[1].tolist()
            id_path = d.absolute_path
            IG_Node = calculate_feature_IG(method, d, device, model)
            out, graph_feature, attention_list = Calculate_feature_attention(d, model)

            row, col, _ = d.adj_t.coo()
            adj_batch = d.batch[row]
            adj_batch_count = np.unique(adj_batch.cpu().detach().numpy(), return_counts=True)[1].tolist()

            start_idx_node = 0
            start_idx_adj = 0
            node_num_cumul = 0

            for inside_count, (batch_item_num, adj_batch_item_num, id_item) in enumerate(
                    zip(batch_count, adj_batch_count, id_path)):

                patient_id = id_item.split('/')[-1].split('_')[0]
                end_idx_node = start_idx_node + batch_item_num
                np.save(os.path.join(ig_value_dir, patient_id + '_IG_value.npy'), IG_Node[start_idx_node:end_idx_node, :])
                np.save(os.path.join(whole_feature_dir, patient_id + '_whole_feature.npy'), graph_feature[start_idx_node:end_idx_node, :].cpu().detach().numpy())
                start_idx_node = end_idx_node

                end_idx_adj = start_idx_adj + adj_batch_item_num
                np.save(os.path.join(attention_dir, patient_id + '_attention_value.npy'), attention_list[:, start_idx_adj:end_idx_adj, :].cpu().detach().numpy())
                start_idx_adj = end_idx_adj

                node_num_cumul = node_num_cumul + batch_item_num

            pbar.update()

    ig_value_dir = os.path.join(save_dir, 'IG_value')
    whole_feature_dir = os.path.join(save_dir, 'Graph_feature')
    attention_dir = os.path.join(save_dir, 'Attention')
    vis_dir = os.path.join(save_dir, 'IG_visualization')
    if os.path.exists(vis_dir) is False:
        os.mkdir(vis_dir)

    max_IG, min_IG, top_threshold, low_threshold, mid_threshold_top, mid_threshold_low = whole_IG_normalize(os.path.join(save_dir, 'IG_value'))

    with tqdm(total=len(test_loader)) as tbar:
        with torch.set_grad_enabled(False):
            for c, d in enumerate(test_loader, 1):
                batch_count = np.unique(d.batch.cpu().detach().numpy(), return_counts=True)[1].tolist()
                id_path = d.absolute_path
                row, col, _ = d.adj_t.coo()
                adj_batch = d.batch[row]
                adj_batch_count = np.unique(adj_batch.cpu().detach().numpy(), return_counts=True)[1].tolist()

                start_idx_node = 0
                end_idx_node = 0

                start_idx_adj = 0
                end_idx_adj = 0

                node_num_cumul = 0

                for inside_count, (batch_item_num, adj_batch_item_num, id_item) in enumerate(zip(batch_count, adj_batch_count, id_path)):
                    patient_id = id_item.split('/')[-1].split('_')[0]
                    end_idx_node = start_idx_node + batch_item_num
                    if os.path.isfile(os.path.join(ig_value_dir, patient_id + '_IG_value.npy')):
                        node_ig = np.load(os.path.join(ig_value_dir, patient_id + '_IG_value.npy'))
                        global_ig_norm, local_ig_norm = ig_normalization(node_ig, max_IG, min_IG)

                        end_idx_adj = start_idx_adj + adj_batch_item_num

                        inside_row = row[start_idx_adj:end_idx_adj].cpu().detach().numpy()
                        inside_col = col[start_idx_adj:end_idx_adj].cpu().detach().numpy()

                        inside_row = np.subtract(inside_row, node_num_cumul)
                        inside_col = np.subtract(inside_col, node_num_cumul)

                        start_idx_adj = end_idx_adj

                        node_num_cumul = node_num_cumul + batch_item_num
                        print(id_item)
                        Visualize_IG(d, inside_count, inside_row, inside_col, node_ig, global_ig_norm,
                                     local_ig_norm, patient_id, Argument,
                                     top_threshold, low_threshold,
                                     mid_threshold_top, mid_threshold_low, save_dir, dataset_root=TrainRoot)
                tbar.update()

def IG_analysis(Argument):
    IG_value_calculation(Argument)
