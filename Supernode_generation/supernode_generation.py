#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 21:10:37 2021

@author: kyungsub
"""

import matplotlib as mpl

mpl.use('Agg')
import json
import os
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import torch
import h5py
import torch_geometric.utils as g_util
from torch_geometric.transforms import Polar
import openslide as osd
from torch_geometric.data import Data
import argparse
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

def supernode_generation(h5_file, Argument, save_dir):
    sample = h5_file.split('.h5')[0].split('/')[-1]
    print(sample)
    # Load the h5_data and feature array
    h5_data = h5py.File(h5_file, 'r')
    coord_array = h5_data['coord_array'][()]
    feature_array = np.array(torch.load(os.path.join(('/').join(h5_file.split('/')[0:10]), 'pt_files', Argument.feature_type, sample + '.pt')))[:, 0:Argument.feature_size]

    node_dict = {}

    reference_patch_size = int(Argument.patch_size)
    if Argument.magnification == '20x':
        reference_patch_size = 4 * reference_patch_size

    for i in range(coord_array.shape[0]):
        node_dict.setdefault(i, [])

    width = max(set(coord_array[:, 0])) - min(set(coord_array[:, 0]))
    height = max(set(coord_array[:, 1])) - min(set(coord_array[:, 1]))

    min_x = min(set(coord_array[:, 0]))
    min_y = min(set(coord_array[:, 1]))

    max_x = max(set(coord_array[:, 0]))
    max_y = max(set(coord_array[:, 1]))

    gridNum = 4
    X_grid_size = int(width / gridNum)
    Y_grid_size = int(height / gridNum)

    # Supernode candidate generation: Calculate the cosine similarity with in 5x5 patches (threshold)
    if Argument.supernode:
        with tqdm(total=(gridNum) * (gridNum)) as pbar:
            for p in range(gridNum):
                for q in range(gridNum):
                    if p == 0:
                        x_cond_reference = np.where(
                            (coord_array[:, 0] <= min_x + X_grid_size * (p + 1)) & (coord_array[:, 0] >= min_x))
                        x_cond_compare = np.where(
                            (coord_array[:, 0] <= min_x + (X_grid_size * (p + 1)) + (reference_patch_size * 2)) & (
                                        coord_array[:, 0] >= min_x))
                    elif p == (gridNum - 1):
                        x_cond_reference = np.where(
                            (coord_array[:, 0] <= max_x) & (coord_array[:, 0] >= min_x + X_grid_size * p))
                        x_cond_compare = np.where((coord_array[:, 0] <= max_x) & (
                                    coord_array[:, 0] >= min_x + ((X_grid_size * p) - (reference_patch_size * 2))))
                    else:
                        x_cond_reference = np.where((coord_array[:, 0] <= min_x + X_grid_size * (p + 1)) & (
                                    coord_array[:, 0] >= min_x + X_grid_size * p))
                        x_cond_compare = np.where(
                            (coord_array[:, 0] <= min_x + X_grid_size * (p + 1) + (reference_patch_size * 2)) & (
                                        coord_array[:, 0] >= min_x + X_grid_size * p - (reference_patch_size * 2)))

                    if q == 0:
                        y_cond_reference = np.where(
                            (coord_array[:, 1] <= min_y + Y_grid_size * (q + 1)) & (coord_array[:, 1] >= min_y))
                        y_cond_compare = np.where(
                            (coord_array[:, 1] <= min_y + (Y_grid_size * (q + 1)) + (reference_patch_size * 2)) & (
                                        coord_array[:, 1] >= min_y))
                    elif q == (gridNum - 1):
                        y_cond_reference = np.where(
                            (coord_array[:, 1] <= max_y) & (coord_array[:, 1] >= min_y + (Y_grid_size * q)))
                        y_cond_compare = np.where((coord_array[:, 1] <= max_y) & (
                                    coord_array[:, 1] >= (min_y + (Y_grid_size * q) - (reference_patch_size * 2))))
                    else:
                        y_cond_reference = np.where((coord_array[:, 1] <= min_y + Y_grid_size * (q + 1)) & (
                                    coord_array[:, 1] >= min_y + Y_grid_size * q))
                        y_cond_compare = np.where(
                            (coord_array[:, 1] <= min_y + Y_grid_size * (q + 1) + (reference_patch_size * 2)) & (
                                        coord_array[:, 1] >= min_y + Y_grid_size * q - (reference_patch_size * 2)))

                    common_indices_reference = np.intersect1d(x_cond_reference, y_cond_reference)
                    common_indices_compare = np.intersect1d(x_cond_compare, y_cond_compare)

                    grid_coord_reference = coord_array[common_indices_reference]
                    grid_coord_compare = coord_array[common_indices_compare]

                    if len(grid_coord_reference) == 0:
                        print('No node is selected')
                        pbar.update()
                        continue

                    grid_feature_reference = feature_array[common_indices_reference]
                    grid_feature_compare = feature_array[common_indices_compare]

                    index_list_reference = common_indices_reference.tolist()
                    index_list_compare = common_indices_compare.tolist()

                    coordinate_matrix = euclidean_distances(X=grid_coord_reference, Y=grid_coord_compare)
                    coordinate_mask = np.where(
                        coordinate_matrix > Argument.node_spatial_threshold * reference_patch_size, 0, 1)
                    cosine_matrix = cosine_similarity(grid_feature_reference, grid_feature_compare)

                    Adj_list = (coordinate_mask == 1).astype(int) * (cosine_matrix >= Argument.threshold).astype(int)

                    for i in range(Adj_list.shape[0]):
                        adj_nodes = np.where((Adj_list[i] > 0))[0].tolist()
                        adj_nodes = [index_list_compare[node] for node in adj_nodes if
                                     index_list_compare[node] != index_list_reference[i]]
                        node_dict[index_list_reference[i]] += adj_nodes

                    pbar.update()

        # Select the supernode in the supernode candidate, the most representative supernode(including many nodes)
        for key_value in node_dict.keys():
            node_dict[key_value] = list(set(node_dict[key_value]))
        dict_len_list = [len(node_dict[key]) for key in node_dict.keys()]
        arglist_strict = np.argsort(np.array(dict_len_list))
        arglist_strict = arglist_strict[::-1]

        for arg_value in arglist_strict:
            if arg_value in node_dict.keys():
                for adj_item in node_dict[arg_value]:
                    if adj_item in node_dict.keys():
                        node_dict.pop(adj_item)
                        arglist_strict = np.delete(arglist_strict, np.argwhere(arglist_strict == adj_item))

        supernode_coordinate_x = []
        supernode_coordinate_y = []
        supernode_feature = []
        supernode_index = []

        # Calculate the feature of the supernode: mean pooling the included supernode
        with tqdm(total=len(node_dict.keys())) as pbar_node:
            for key_value in node_dict.keys():
                supernode_index.append(key_value)
                supernode_coordinate_x.append(coord_array[key_value, 0])
                supernode_coordinate_y.append(coord_array[key_value, 1])

                if len(node_dict[key_value]) == 0:
                    select_feature = feature_array[key_value, :]
                else:
                    select_feature = feature_array[node_dict[key_value] + [key_value], :]
                    select_feature = select_feature.mean(axis=0)

                if len(supernode_feature) == 0:
                    supernode_feature = np.reshape(select_feature, (1, Argument.feature_size))
                else:
                    supernode_feature = np.concatenate(
                        (supernode_feature, np.reshape(select_feature, (1, Argument.feature_size))), axis=0)
                pbar_node.update()

        supernode_coord_array = np.column_stack((supernode_coordinate_x, supernode_coordinate_y))
    else:
        supernode_coord_array = coord_array
        supernode_feature = feature_array
        supernode_index = range(coord_array.shape[0])

    # Draw a graph based on the supernode distance threshold (7x7)
    supernode_coordinate_matrix = euclidean_distances(supernode_coord_array, supernode_coord_array)
    supernode_coordinate_mask = np.where(
        supernode_coordinate_matrix > Argument.supernode_spatial_threshold * reference_patch_size, 0, 1)

    Edge_label = np.where(supernode_coordinate_mask == 1)
    fromlist = Edge_label[0].tolist()
    tolist = Edge_label[1].tolist()

    edge_index = torch.tensor([fromlist, tolist], dtype=torch.long)
    x = torch.tensor(supernode_feature, dtype=torch.float)
    data = Data(x=x, edge_index=edge_index)

    # Get the graph only if the node number is more than 100
    connected_graph = g_util.to_networkx(data, to_undirected=True)
    connected_graph = [connected_graph.subgraph(item_graph).copy() for item_graph in
                       nx.connected_components(connected_graph) if len(item_graph) > 100]

    connected_graph_node_list = []

    for graph_item in connected_graph:
        connected_graph_node_list.extend(list(graph_item.nodes))
    connected_graph = connected_graph_node_list
    connected_graph = list(connected_graph)
    new_node_order_dict = dict(zip(connected_graph, range(len(connected_graph))))

    new_feature = data.x[connected_graph]
    new_edge_index = data.edge_index.numpy()
    new_edge_mask_from = np.isin(new_edge_index[0], connected_graph)
    new_edge_mask_to = np.isin(new_edge_index[1], connected_graph)
    new_edge_mask = new_edge_mask_from * new_edge_mask_to
    new_edge_index_from = new_edge_index[0]
    new_edge_index_from = new_edge_index_from[new_edge_mask]
    new_edge_index_from = [new_node_order_dict[item] for item in new_edge_index_from]
    new_edge_index_to = new_edge_index[1]
    new_edge_index_to = new_edge_index_to[new_edge_mask]
    new_edge_index_to = [new_node_order_dict[item] for item in new_edge_index_to]

    new_edge_index = torch.tensor([new_edge_index_from, new_edge_index_to], dtype=torch.long)
    supernode_index = list(np.array(supernode_index)[connected_graph])
    supernode_coord_array = supernode_coord_array[connected_graph]

    # Polar transformation
    actual_pos = torch.tensor(supernode_coord_array)
    actual_pos = actual_pos.float()

    pos_transfrom = Polar()
    new_graph = Data(x=new_feature, edge_index=new_edge_index, pos=actual_pos)
    try:
        new_graph = pos_transfrom(new_graph)
    except:
        print('Polar error')
        return 0

    # Save the data
    h5_output_dir = os.path.join(save_dir, 'h5_files')
    pt_output_dir = os.path.join(save_dir, 'pt_files')
    image_output_dir = os.path.join(save_dir, 'images')
    if os.path.exists(image_output_dir) is False:
        os.mkdir(image_output_dir)
    h5_output_path = os.path.join(h5_output_dir, sample + '.h5')
    pt_output_path = os.path.join(pt_output_dir, sample + '.pt')
    file = h5py.File(h5_output_path, 'a')
    supernode_asset_dict = {
        'coords': supernode_coord_array,
        'supernode_index': np.array(supernode_index)
    }
    for key, val in supernode_asset_dict.items():
        data_shape = val.shape
        data_type = val.dtype
        chunk_shape = (1,) + data_shape[1:]
        maxshape = (None,) + data_shape[1:]
        dset = file.create_dataset(key, shape=data_shape, maxshape=maxshape, chunks=chunk_shape,
                                   dtype=data_type)
        dset[:] = val

    serialized_dict = json.dumps(node_dict)
    if 'supernode_include' in file:
        del file['supernode_include']
    file.create_dataset('supernode_include', data=serialized_dict, dtype=h5py.string_dtype(encoding='utf-8'))
    file.close()
    torch.save(new_graph, pt_output_path)

    try:
        slideimage = osd.OpenSlide(os.path.join(Argument.svs_dir,Argument.database, Argument.cancertype, sample + '.svs'))
    except:
        print('openslide error')
        return 0

    # Visualize the graph
    WSI_graph = nx.Graph()
    WSI_graph.add_nodes_from(list(range(new_graph.x.shape[0])))
    WSI_edge_index = [(row_item, col_item) for row_item, col_item in zip(new_edge_index_from, new_edge_index_to)]
    WSI_graph.add_edges_from(WSI_edge_index)
    WSI_graph.remove_edges_from(nx.selfloop_edges(WSI_graph))

    X_pos = supernode_coord_array[:, 0]
    Y_pos = supernode_coord_array[:, 1]
    X_Y_pos = [(X / 16, Y / 16) for X, Y in zip(X_pos, Y_pos)]
    pos_dict = zip(range(new_graph.x.shape[0]), X_Y_pos)
    pos_dict = dict(pos_dict)

    WSI_level = 2
    WSI_width, WSI_height = slideimage.level_dimensions[WSI_level]
    WSI_image = slideimage.read_region((0, 0), WSI_level, (WSI_width, WSI_height))

    my_dpi = 24
    plt.figure(figsize=(WSI_image.size[0] / my_dpi, WSI_image.size[1] / my_dpi), dpi=24)
    plt.imshow(WSI_image)
    plt.axis('off')
    nx.draw_networkx(WSI_graph, pos=pos_dict, node_size=50, node_color='black', width=2, alpha=0.8, arrows=False,
                     with_labels=False)
    plt.subplots_adjust(left=0., right=1., top=1., bottom=0.)
    plt.gca().invert_yaxis()
    plt.savefig(os.path.join(
        image_output_dir,
        sample + '_WSI_graph_wo_IG_temp.jpeg'), transparent=True)
    plt.clf()
    plt.cla()
    plt.close()


def Parser_main():
    parser = argparse.ArgumentParser(description="TEA-graph superpatch generation")
    parser.add_argument("--database", default='SNUH', help="Use in the savedir", type=str)
    parser.add_argument("--cancertype", default='GBM', help="cancer type", type=str)
    parser.add_argument("--magnification", default='40x', help='magnification', type=str)
    parser.add_argument("--save_dir", default="/mnt/disk2/TEAgraph_preprocessing/")
    parser.add_argument("--graphdir", default="graphs", help="graph save dir", type=str)
    parser.add_argument("--patch_size", default='256', help='patch_size', type=str)
    parser.add_argument("--pretrained_model", default='Efficientnet', type=str)
    parser.add_argument("--threshold", default=0.75, help="cosine similarity threshold", type=float)
    parser.add_argument("--node_spatial_threshold", default=2.9,
                        help="Spatial threshold between nodes for supernode generation", type=float)
    parser.add_argument("--supernode_spatial_threshold", default=4.3,
                        help="Spatial threshold between nodes for graph construction", type=float)
    parser.add_argument("--feature_size", default=1792, help="embedding size of pretrained_model", type=int)
    parser.add_argument("--supernode", action='store_true', default=False)
    parser.add_argument("--group_num", default=0, type=int)
    parser.add_argument("--group", default=None, type=int)
    parser.add_argument("--feature_type", default = 'ki_67', type = str)
    parser.add_argument("--svs_dir", default = "/mnt/disk3/svs_data/", type = str, help = 'svs_data_directory for drawing graph')
    return parser.parse_args()

def main():
    Argument = Parser_main()
    cancer_type = Argument.cancertype
    database = Argument.database
    root_dir = Argument.save_dir
    database_dir = os.path.join(root_dir, database)
    cancer_type_dir = os.path.join(database_dir, cancer_type)
    magnification_dir = os.path.join(cancer_type_dir, Argument.magnification)
    patch_size_dir = os.path.join(magnification_dir, Argument.patch_size)
    pretrained_model_dir = os.path.join(patch_size_dir, Argument.pretrained_model)
    feature_dir = os.path.join(pretrained_model_dir, 'feature')
    h5_dir = os.path.join(feature_dir, 'h5_files')
    save_dir = os.path.join(pretrained_model_dir, Argument.graphdir)
    if os.path.exists(save_dir) is False:
        os.mkdir(save_dir)
    save_dir = os.path.join(save_dir, Argument.feature_type)
    if os.path.exists(save_dir) is False:
        os.mkdir(save_dir)
    save_dir = os.path.join(save_dir, str(Argument.threshold))
    if os.path.exists(save_dir) is False:
        os.mkdir(save_dir)
    if os.path.exists(os.path.join(save_dir, 'h5_files')) is False:
        os.mkdir(os.path.join(save_dir, 'h5_files'))
        os.mkdir(os.path.join(save_dir, 'pt_files'))
        os.mkdir(os.path.join(save_dir, 'images'))

    files = os.listdir(h5_dir)
    processed_sample = os.listdir(os.path.join(save_dir, 'h5_files'))
    processed_sample = [sample.split('.h5')[0] for sample in processed_sample]

    final_files = [f for f in files if f.split('.h5')[0] not in processed_sample]
    final_files = [os.path.join(h5_dir, file) for file in final_files]
    final_files.sort(key=lambda f: os.stat(f).st_size, reverse=True)
    if Argument.group_num > 1:
        group_list = [i % Argument.group_num for i in range(len(final_files))]
        final_process_df = pd.DataFrame({
            'file_location': final_files,
            'group': group_list
        })
        final_files = final_process_df[final_process_df['group'] == int(Argument.group)]['file_location'].tolist()

    with tqdm(total=len(final_files)) as pbar_tot:
        for h5py_file in final_files:
            supernode_generation(h5py_file, Argument, save_dir)
            pbar_tot.update()

if __name__ == "__main__":
    main()