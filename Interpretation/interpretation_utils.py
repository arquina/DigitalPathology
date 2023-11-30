import matplotlib as mpl
mpl.use('Agg')

import os
import numpy as np
from tqdm import tqdm
import openslide as osd
import json
import torch
import h5py
import cv2 as cv
from PIL import Image
from PIL import ImageFilter
import matplotlib.pyplot as plt

from matplotlib import cm
import networkx as nx
from captum.attr import IntegratedGradients

def whole_IG_normalize(rootdir):
    Node_IG_list = os.listdir(rootdir)
    Whole_IG_value = []

    for item in Node_IG_list:
        patient_IG_path = os.path.join(rootdir, item)
        if len(Whole_IG_value) == 0:
            Whole_IG_value = np.load(patient_IG_path)
        else:
            Whole_IG_value = np.concatenate((Whole_IG_value, np.load(patient_IG_path)))

    pos_term = np.where(Whole_IG_value >= 0)[0]
    neg_term = np.where(Whole_IG_value < 0)[0]

    upper_outlier = np.quantile(Whole_IG_value[pos_term], 0.98)
    lower_outlier = np.quantile(Whole_IG_value[neg_term], 0.02)
    top_threshold = np.quantile(Whole_IG_value[pos_term], 0.88)
    low_threshold = np.quantile(Whole_IG_value[neg_term], 0.12)
    mid_threshold_top = np.quantile(Whole_IG_value[pos_term], 0.05)
    mid_threshold_low = np.quantile(Whole_IG_value[neg_term], 0.93)

    outlier_removed_IG_value = Whole_IG_value[
        list(set(np.where(Whole_IG_value < upper_outlier)[0]) & set(np.where(Whole_IG_value > lower_outlier)[0]))]

    max_IG = outlier_removed_IG_value.max()
    min_IG = outlier_removed_IG_value.min()

    return max_IG, min_IG, top_threshold, low_threshold, mid_threshold_top, mid_threshold_low


def whole_attention_normalize(rootdir):
    Patients = os.listdir(rootdir)
    Whole_IG_value = []

    with tqdm(total=len(Patients)) as pbar:
        for item in Patients:
            patient_root = os.path.join(rootdir, item)
            patient_features = os.listdir(patient_root)
            patient_features = [item for item in patient_features if 'attention_value.npy' in item]
            if len(patient_features) > 0:
                if len(Whole_IG_value) == 0:
                    Whole_IG_value = np.mean(np.mean(np.load(os.path.join(patient_root,
                                                                          patient_features[0])), axis=0), axis=1)
                else:
                    Whole_IG_value = np.concatenate((Whole_IG_value,
                                                     np.mean(np.mean(np.load(os.path.join(patient_root,
                                                                                          patient_features[0])),
                                                                     axis=0), axis=1)))
            pbar.update()

    upper_outlier = np.quantile(Whole_IG_value, 0.98)
    lower_outlier = np.quantile(Whole_IG_value, 0.02)

    outlier_removed_IG_value = Whole_IG_value[
        list(set(np.where(Whole_IG_value < upper_outlier)[0]) & set(np.where(Whole_IG_value > lower_outlier)[0]))]

    max_IG = outlier_removed_IG_value.max()
    min_IG = outlier_removed_IG_value.min()

    top_threshold = np.quantile(outlier_removed_IG_value, 0.88)
    low_threshold = np.quantile(outlier_removed_IG_value, 0.12)
    mid_threshold_top = np.quantile(outlier_removed_IG_value, 0.55)
    mid_threshold_low = np.quantile(outlier_removed_IG_value, 0.45)

    return max_IG, min_IG, top_threshold, low_threshold, mid_threshold_top, mid_threshold_low


def ig_normalization(node_ig, max_IG, min_IG):
    node_ig_norm = node_ig.copy()
    upper_threshold_node = np.where(node_ig > max_IG)[0]
    lower_threshold_node = np.where(node_ig < min_IG)[0]

    node_ig_norm[upper_threshold_node] = max_IG
    node_ig_norm[lower_threshold_node] = min_IG
    node_ig_remove_outlier = node_ig_norm.copy()
    global_node_ig_norm = (node_ig_norm - min_IG) / (max_IG - min_IG)
    local_node_ig_norm = node_ig_remove_outlier.copy()

    node_ig_positive = np.where(node_ig_remove_outlier >= 0)[0]
    node_ig_negative = np.where(node_ig_remove_outlier < 0)[0]

    posnorm = node_ig_remove_outlier[node_ig_positive] / max_IG
    local_node_ig_norm[node_ig_positive] = posnorm
    negnorm = -1 * node_ig_remove_outlier[node_ig_negative] / min_IG
    local_node_ig_norm[node_ig_negative] = negnorm

    local_node_ig_norm = local_node_ig_norm / 2.0 + 0.5

    return global_node_ig_norm, local_node_ig_norm


def Whole_supernode_vis(patient, supernode_coords, supernode_index, node_dict, original_node_coords, edge_index,
                        IG_normalized, Argument, save_dir):
    graph_w_IG_dir = os.path.join(save_dir, 'graph_w_IG')
    graph_wo_IG_dir = os.path.join(save_dir, 'graph_wo_IG')
    IG_mask_dir = os.path.join(save_dir, 'IG_mask')
    IG_mask_with_wsi = os.path.join(save_dir, 'IG_mask_with_WSI')

    if os.path.exists(graph_w_IG_dir) is False:
        os.mkdir(graph_wo_IG_dir)
        os.mkdir(graph_w_IG_dir)
        os.mkdir(IG_mask_dir)
        os.mkdir(IG_mask_with_wsi)

    svs_dir = os.path.join("/mnt/disk3/svs_data/", Argument.DatasetType, Argument.CancerType)
    svs_file_list = os.listdir(svs_dir)
    svs_file = [svs for svs in svs_file_list if patient in svs][0]
    svs_file_path = os.path.join(svs_dir, svs_file)

    WSI = osd.open_slide(svs_file_path)
    WSI_level = 2
    Downsample_ratio = int(WSI.level_downsamples[WSI_level])
    WSI_width, WSI_height = WSI.level_dimensions[WSI_level]
    WSI_image = WSI.read_region((0, 0), WSI_level, (WSI_width, WSI_height))
    Image_mask = np.zeros((WSI_height, WSI_width))
    Image_IG_mask = np.zeros((WSI_height, WSI_width))
    Image_IG_mask_count = np.zeros((WSI_height, WSI_width))
    WSI_patch_dimension = int(256 / Downsample_ratio)

    for col_idx in range(supernode_coords.shape[0]):
        Superpatch = int(supernode_index[col_idx])
        Superpatch_pos = (original_node_coords[Superpatch, :] / 256).astype(int)
        Envpatch = node_dict[str(Superpatch)]
        Envpatch_pos = (original_node_coords[Envpatch, :] / 256).astype(int)

        Image_mask[Superpatch_pos[1] * WSI_patch_dimension:(Superpatch_pos[1] + 1) * WSI_patch_dimension:,
        Superpatch_pos[0] * WSI_patch_dimension:(Superpatch_pos[0] + 1) * WSI_patch_dimension] = 2

        Image_IG_mask[Superpatch_pos[1] * WSI_patch_dimension:(Superpatch_pos[1] + 1) * WSI_patch_dimension:,
        Superpatch_pos[0] * WSI_patch_dimension:(Superpatch_pos[0] + 1) * WSI_patch_dimension] = (
        IG_normalized[col_idx])

        Image_IG_mask_count[Superpatch_pos[1] * WSI_patch_dimension:(Superpatch_pos[1] + 1) * WSI_patch_dimension:,
        Superpatch_pos[0] * WSI_patch_dimension:(Superpatch_pos[0] + 1) * WSI_patch_dimension] = 1

        for Env_y, Env_x in zip(Envpatch_pos[:, 1].tolist(), Envpatch_pos[:, 0].tolist()):
            Image_mask[Env_y * WSI_patch_dimension:(Env_y + 1) * WSI_patch_dimension,
            Env_x * WSI_patch_dimension:(Env_x + 1) * WSI_patch_dimension] = 1
            Image_IG_mask[Env_y * WSI_patch_dimension:(Env_y + 1) * WSI_patch_dimension,
            Env_x * WSI_patch_dimension:(Env_x + 1) * WSI_patch_dimension] = Image_IG_mask[Env_y * WSI_patch_dimension:(
                                                                                                                                   Env_y + 1) * WSI_patch_dimension,
                                                                             Env_x * WSI_patch_dimension:(
                                                                                                                     Env_x + 1) * WSI_patch_dimension] + (
                                                                             IG_normalized[col_idx])
            Image_IG_mask_count[Env_y * WSI_patch_dimension:(Env_y + 1) * WSI_patch_dimension,
            Env_x * WSI_patch_dimension:(Env_x + 1) * WSI_patch_dimension] = Image_IG_mask_count[
                                                                             Env_y * WSI_patch_dimension:(
                                                                                                                     Env_y + 1) * WSI_patch_dimension,
                                                                             Env_x * WSI_patch_dimension:(
                                                                                                                     Env_x + 1) * WSI_patch_dimension] + 1

    alpha_channel = np.where(Image_IG_mask_count != 0, 100, 0).astype(np.uint8)
    Image_IG_mask_count[Image_IG_mask_count == 0] = 1
    Image_IG_mask = Image_IG_mask / Image_IG_mask_count
    Image_IG_mask_rgba = np.uint8(255 * cm.coolwarm(Image_IG_mask))
    Image_IG_mask_array = Image.fromarray(Image_IG_mask_rgba)
    Image_IG_mask_array = Image_IG_mask_array.filter(ImageFilter.GaussianBlur)
    Image_IG_mask_array.save(os.path.join(IG_mask_dir, patient + '_WSI_Image_mask_IG_new.png'))

    Image_IG_mask_rgba[..., 3] = alpha_channel
    Image_IG_mask = Image.fromarray(Image_IG_mask_rgba)
    Image_IG_mask = Image_IG_mask.filter(ImageFilter.GaussianBlur)
    combined_image = Image.alpha_composite(WSI_image, Image_IG_mask)
    combined_image.save(os.path.join(IG_mask_with_wsi, patient + '_WSI_Image_mask_IG_with_WSI.png'))

    Image_mask = Image_mask - Image_mask.min()
    Image_mask = Image_mask / 1.0
    Image_mask = cv.applyColorMap(np.uint8(255 * Image_mask), cv.COLORMAP_JET)
    Colover_converted_mask = cv.cvtColor(Image_mask, cv.COLOR_BGR2RGB)
    Image_mask = Image.fromarray(Colover_converted_mask)

    Mask_fig = Image_mask
    Mask_fig = Mask_fig.convert('RGBA')
    Mask_fig_IG = Image_IG_mask.convert('RGBA')

    WSI_node_idx = supernode_index
    WSI_node_idx = [int(item) for item in WSI_node_idx]
    WSI_Location = original_node_coords[WSI_node_idx, :]
    X_pos = WSI_Location[:, 0].tolist()
    Y_pos = WSI_Location[:, 1].tolist()
    X_Y_pos = [(X, Y) for X, Y in zip(X_pos, Y_pos)]
    pos_dict = zip(list(range(len(WSI_node_idx))), X_Y_pos)
    pos_dict = dict(pos_dict)

    WSI_edge_index = [(row, col) for row, col in zip(edge_index[0, :], edge_index[1, :])]
    WSI_graph = nx.Graph()
    WSI_graph.add_nodes_from(list(range(len(WSI_node_idx))))
    WSI_graph.add_edges_from(WSI_edge_index)
    WSI_graph.remove_edges_from(nx.selfloop_edges(WSI_graph))

    my_dpi = 96
    plt.figure(figsize=(WSI_image.size[0] / my_dpi, WSI_image.size[1] / my_dpi), dpi=96)
    plt.axis('off')
    nx.draw_networkx(WSI_graph, pos=pos_dict, node_size=20, node_color='black', width=0.2, alpha=0.8, arrows=False,
                     with_labels=False)
    plt.subplots_adjust(left=0., right=1., top=1., bottom=0.)
    plt.gca().invert_yaxis()
    plt.savefig(os.path.join(graph_wo_IG_dir, patient + '_WSI_graph_wo_IG.png'), transparent=True)
    plt.clf()
    plt.cla()
    plt.close()

    my_dpi = 96
    plt.figure(figsize=(WSI_image.size[0] / my_dpi, WSI_image.size[1] / my_dpi), dpi=96)
    plt.axis('off')
    nx.draw_networkx(WSI_graph, pos=pos_dict, node_size=30, width=0.2, node_color=(IG_normalized), vmin=0.0, vmax=1.0,
                     cmap=plt.cm.coolwarm, arrows=False, with_labels=False)
    plt.subplots_adjust(left=0., right=1., top=1., bottom=0.)
    plt.gca().invert_yaxis()
    plt.savefig(os.path.join(graph_w_IG_dir, patient + '_WSI_graph_w_IG.png'), transparent=True)
    plt.clf()
    plt.cla()
    plt.close()


def Visualize_IG(d, d_count, row, col, ig, global_ig_norm, local_ig_norm, p_id, Argument,
                 top_threshold, low_threshold, mid_threshold_top, mid_threshold_low, save_dir, dataset_root):
    id_path = d.absolute_path[d_count]
    id_sample = id_path.split('/')[-1].split('.')[0]
    root_dir = os.path.join('/'.join(id_path.split('/')[0:8]), Argument.pretrain)
    graph_dir = os.path.join(root_dir, 'graphs', Argument.dataset_type, Argument.spatial_threshold)
    feature_dir = os.path.join(root_dir, 'feature')

    graph_pt_dir = os.path.join(graph_dir, 'pt_files')
    graph_h5_dir = os.path.join(graph_dir, 'h5_files')
    feature_h5_dir = os.path.join(feature_dir, 'h5_files')

    graph_pt_file_list = os.listdir(graph_pt_dir)
    graph_h5_file_list = os.listdir(graph_h5_dir)
    feature_h5_file_list = os.listdir(feature_h5_dir)

    graph_pt = [f for f in graph_pt_file_list if id_sample in f][0]
    graph_h5 = [f for f in graph_h5_file_list if id_sample in f][0]
    feature_h5_file = [f for f in feature_h5_file_list if id_sample in f][0]

    d = torch.load(os.path.join(graph_pt_dir, graph_pt))
    edge_index = np.array(d.edge_index)
    h5_file = h5py.File(os.path.join(graph_h5_dir, graph_h5), 'r')
    feature_h5_file = h5py.File(os.path.join(feature_h5_dir, feature_h5_file), 'r')
    serialized_dict = h5_file['supernode_include'][()]
    node_dict = json.loads(serialized_dict)
    supernode_index = h5_file['supernode_index'][()]
    supernode_coords = h5_file['coords'][()]
    original_node_coords = feature_h5_file['coord_array'][()]
    high_ig = np.where(ig > top_threshold, 1, 0)
    pos_ig = np.where(ig > 0, ig, 0)
    pos_loc = np.where(ig > 0, 1, 0)
    pos_ig = pos_ig / np.max(pos_ig)

    global_save_dir = os.path.join(save_dir, 'global_normalize')
    local_save_dir = os.path.join(save_dir, 'local_normalize')
    high_ig_save_dir = os.path.join(save_dir, 'high_ig')
    pos_ig_save_dir = os.path.join(save_dir, 'pos_ig')
    pos_loc_save_dir = os.path.join(save_dir, 'pos_loc')

    if os.path.exists(global_save_dir) is False:
        os.mkdir(global_save_dir)
        os.mkdir(local_save_dir)
        os.mkdir(high_ig_save_dir)
        os.mkdir(pos_ig_save_dir)
        os.mkdir(pos_loc_save_dir)

    Whole_supernode_vis(id_sample, supernode_coords, supernode_index, node_dict, original_node_coords, edge_index,
                        pos_loc, Argument, pos_loc_save_dir)
    Whole_supernode_vis(id_sample, supernode_coords, supernode_index, node_dict, original_node_coords, edge_index,
                        pos_ig, Argument, pos_ig_save_dir)
    Whole_supernode_vis(id_sample, supernode_coords, supernode_index, node_dict, original_node_coords, edge_index,
                        global_ig_norm, Argument, global_save_dir)
    Whole_supernode_vis(id_sample, supernode_coords, supernode_index, node_dict, original_node_coords, edge_index,
                        local_ig_norm, Argument, local_save_dir)
    Whole_supernode_vis(id_sample, supernode_coords, supernode_index, node_dict, original_node_coords, edge_index,
                        high_ig, Argument, high_ig_save_dir)

    return 0


def model_forward(edge_mask, model, data):
    out = model(data, edge_mask)
    return out


def calculate_feature_IG(method, data, device, model):
    target = 0
    input_mask = torch.ones(data.x.shape[0], 1).requires_grad_(True).to(device)
    baseline = torch.zeros(data.x.shape[0], 1).requires_grad_(True).to(device)
    ig = IntegratedGradients(model_forward)
    mask = ig.attribute(input_mask, target=target, baselines=baseline,
                        additional_forward_args=(model, data), n_steps=50,
                        internal_batch_size=data.x.shape[0])

    edge_mask = mask.cpu().detach().numpy()

    return edge_mask


def Calculate_feature_attention(d, model):
    d = d.to(torch.device(0))
    input_mask = torch.ones(d.x.shape[0], 1).requires_grad_(True).to(torch.device(0))
    out, updated_feature, attention_list = model(d, input_mask, Interpretation_mode=True)

    return out, updated_feature, attention_list