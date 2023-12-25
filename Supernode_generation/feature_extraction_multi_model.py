#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 21:10:37 2021

@author: kyoungseob
"""
import sys
sys.path.append('/home/seob/script/DigitalPathology')
import os
import pandas as pd
import numpy as np
import h5py
from tqdm import tqdm
import torch
import openslide as osd
from torchvision import transforms
from EfficientNet import EfficientNet
from skimage.filters import threshold_multiotsu
import argparse
from utils import StitchCoords
from DataLoader import SlidePatchDataset

def feature_extraction(sample, he_model, ihc_model, device, Argument, save_dir):
    print(sample)
    svs_dir = os.path.join(Argument.svs_dir, Argument.database, Argument.cancertype, 'registered_image')
    downsampled_svs_dir = os.path.join(Argument.svs_dir, Argument.database, Argument.cancertype, 'registered_image_downsampled')

    he_image = os.path.join(svs_dir, sample + '_H&E_registered.ome.tiff')
    ihc_image = os.path.join(svs_dir, sample + '_Ki-67_registered.ome.tiff')
    he_downsampled = os.path.join(downsampled_svs_dir, sample + '_H&E_downsampled_registered.ome.tiff')
    ihc_downsampled = os.path.join(downsampled_svs_dir, sample + '_Ki-67_downsampled_registered.ome.tiff')
    try:
        he_slideimage = osd.OpenSlide(he_image)
    except:
        print('H&E openslide error')
    try:
        ihc_slideimage = osd.OpenSlide(ihc_image)
    except:
        print('IHC openslide error')
    try:
        he_downsampled_slideimage = osd.OpenSlide(he_downsampled)
    except:
        print('H&E downsampled openslide error')
    try:
        ihc_downsampled_slideimage = osd.OpenSlide(ihc_downsampled)
    except:
        print('IHC downsampled openslide error')

    he_downsampling_factor = int(he_slideimage.level_dimensions[0][0] / he_downsampled_slideimage.level_dimensions[0][0])
    ihc_downsampling_factor = int(ihc_slideimage.level_dimensions[0][0] / ihc_downsampled_slideimage.level_dimensions[0][0])

    # Get the image at the requested scale
    he_native_levelimg = he_downsampled_slideimage.read_region((0,0), 0, he_downsampled_slideimage.level_dimensions[0])
    ihc_native_levelimg = ihc_downsampled_slideimage.read_region((0,0), 0, ihc_downsampled_slideimage.level_dimensions[0])
    he_native_levelimg = he_native_levelimg.convert('L')
    ihc_native_levelimg = ihc_native_levelimg.convert('L')

    he_img = np.array(he_native_levelimg)
    ihc_img = np.array(ihc_native_levelimg)

    he_thresholds = threshold_multiotsu(he_img)
    he_regions = np.digitize(he_img, bins=he_thresholds)
    he_regions[he_regions == 1] = 0
    he_regions[he_regions == 2] = 1
    he_thresh_otsu = he_regions

    ihc_thresholds = threshold_multiotsu(ihc_img)
    ihc_regions = np.digitize(he_img, bins=ihc_thresholds)
    ihc_regions[ihc_regions == 1] = 0
    ihc_regions[ihc_regions == 2] = 1
    ihc_thresh_otsu = ihc_regions

    imagesize = Argument.patch_size
    he_downsampled_size = int(imagesize / he_downsampling_factor)
    ihc_downsampled_size = int(imagesize / ihc_downsampling_factor)
    Width = he_slideimage.dimensions[0]
    Height = he_slideimage.dimensions[1]
    num_row = int(Height / imagesize) + 1
    num_col = int(Width / imagesize) + 1
    counter = 0
    inside_counter = 0
    he_patch_list = []
    ihc_patch_list = []
    temp_x = []
    temp_y = []

    with tqdm(total=num_row * num_col) as pbar_image:
        for i in range(0, num_col):
            for j in range(0, num_row):
                if he_thresh_otsu.shape[1] >= (i + 1) * he_downsampled_size:
                    if he_thresh_otsu.shape[0] >= (j + 1) * he_downsampled_size:
                        he_cut_thresh = he_thresh_otsu[j * he_downsampled_size:(j + 1) * he_downsampled_size,
                                     i * he_downsampled_size:(i + 1) * he_downsampled_size]
                    else:
                        he_cut_thresh = he_thresh_otsu[(j) * he_downsampled_size:he_thresh_otsu.shape[0],
                                     i * he_downsampled_size:(i + 1) * he_downsampled_size]
                else:
                    if he_thresh_otsu.shape[0] >= (j + 1) * he_downsampled_size:
                        he_cut_thresh = he_thresh_otsu[j * he_downsampled_size:(j + 1) * he_downsampled_size,
                                     (i) * he_downsampled_size:he_thresh_otsu.shape[1]]
                    else:
                        he_cut_thresh = he_thresh_otsu[(j) * he_downsampled_size:he_thresh_otsu.shape[0],
                                     (i) * he_downsampled_size:he_thresh_otsu.shape[1]]

                if ihc_thresh_otsu.shape[1] >= (i + 1) * ihc_downsampled_size:
                    if ihc_thresh_otsu.shape[0] >= (j + 1) * ihc_downsampled_size:
                        ihc_cut_thresh = ihc_thresh_otsu[j * ihc_downsampled_size:(j + 1) * ihc_downsampled_size,
                                     i * ihc_downsampled_size:(i + 1) * ihc_downsampled_size]
                    else:
                        ihc_cut_thresh = ihc_thresh_otsu[(j) * ihc_downsampled_size:ihc_thresh_otsu.shape[0],
                                     i * ihc_downsampled_size:(i + 1) * ihc_downsampled_size]
                else:
                    if ihc_thresh_otsu.shape[0] >= (j + 1) * ihc_downsampled_size:
                        ihc_cut_thresh = ihc_thresh_otsu[j * ihc_downsampled_size:(j + 1) * ihc_downsampled_size,
                                     (i) * ihc_downsampled_size:ihc_thresh_otsu.shape[1]]
                    else:
                        ihc_cut_thresh = ihc_thresh_otsu[(j) * ihc_downsampled_size:ihc_thresh_otsu.shape[0],
                                     (i) * ihc_downsampled_size:ihc_thresh_otsu.shape[1]]

                if np.mean(he_cut_thresh) > 0.75 or np.mean(ihc_cut_thresh) > 0.9:
                        pbar_image.update()
                        pass
                else:
                    filter_location = (i * imagesize, j * imagesize)
                    level = 0
                    patch_size = (imagesize, imagesize)
                    location = (filter_location[0], filter_location[1])

                    he_CutImage = he_slideimage.read_region(location, level, patch_size)
                    ihc_CutImage = ihc_slideimage.read_region(location, level, patch_size)
                    he_patch_list.append(he_CutImage)
                    ihc_patch_list.append(ihc_CutImage)
                    temp_x.append(i * imagesize)
                    temp_y.append(j * imagesize)
                    counter += 1
                    batchsize = 256

                    if counter == batchsize:
                        Dataset = SlidePatchDataset([he_patch_list, ihc_patch_list], temp_x, temp_y, Argument.transform, multimodel = True)
                        dataloader = torch.utils.data.DataLoader(Dataset, batch_size=batchsize, num_workers=0,
                                                                 drop_last=False)
                        for sample_img in dataloader:
                            he_images = sample_img['he_image']
                            ihc_images = sample_img['ihc_image']
                            he_images = he_images.to(device)
                            ihc_images = ihc_images.to(device)
                            with torch.set_grad_enabled(False):
                                _, he_features = he_model(he_images)
                                _, ihc_features = ihc_model(ihc_images)
                                plus_graph_features = he_features + ihc_features

                        if inside_counter == 0:
                            he_feature_list = he_features.cpu().detach().numpy()
                            ihc_feature_list = ihc_features.cpu().detach().numpy()
                            plus_graph_feature_list = plus_graph_features.cpu().detach().numpy()
                            temp_x = np.reshape(np.array(temp_x), (len(temp_x), 1))
                            temp_y = np.reshape(np.array(temp_y), (len(temp_x), 1))

                            x_y_list = np.concatenate((temp_x, temp_y), axis=1)
                        else:
                            he_feature_list = np.concatenate((he_feature_list,
                                                        he_features.cpu().detach().numpy()), axis = 0)
                            ihc_feature_list = np.concatenate((ihc_feature_list,
                                                               ihc_features.cpu().detach().numpy()), axis = 0)
                            plus_graph_feature_list = np.concatenate((plus_graph_feature_list,
                                                                      plus_graph_features.cpu().detach().numpy()))
                            temp_x = np.reshape(np.array(temp_x), (len(temp_x), 1))
                            temp_y = np.reshape(np.array(temp_y), (len(temp_x), 1))

                            x_y_list = np.concatenate((x_y_list,
                                                       np.concatenate((temp_x, temp_y), axis=1)), axis=0)
                        inside_counter += 1
                        he_patch_list = []
                        ihc_patch_list = []
                        temp_x = []
                        temp_y = []
                        counter = 0

                    pbar_image.update()

        if counter < batchsize and counter > 0:
            Dataset = SlidePatchDataset([he_patch_list, ihc_patch_list], temp_x, temp_y, Argument.transform, multimodel= True)
            dataloader = torch.utils.data.DataLoader(Dataset, batch_size=batchsize, num_workers=0,
                                                     drop_last=False)
            for sample_img in dataloader:
                he_images = sample_img['he_image']
                ihc_images = sample_img['ihc_image']
                he_images = he_images.to(device)
                ihc_images = ihc_images.to(device)
                with torch.set_grad_enabled(False):
                    _, he_features = he_model(he_images)
                    _, ihc_features = ihc_model(ihc_images)
                    plus_graph_features = he_features + ihc_features

                if Argument.pretrained_model == 'Efficientnet':
                    he_feature_list = np.concatenate((he_feature_list,
                                                      he_features.cpu().detach().numpy()), axis=0)
                    ihc_feature_list = np.concatenate((ihc_feature_list,
                                                       ihc_features.cpu().detach().numpy()), axis=0)
                    plus_graph_feature_list = np.concatenate((plus_graph_feature_list,
                                                              plus_graph_features.cpu().detach().numpy()))
                    temp_x = np.reshape(np.array(temp_x), (len(temp_x), 1))
                    temp_y = np.reshape(np.array(temp_y), (len(temp_x), 1))

                    x_y_list = np.concatenate((x_y_list,
                                               np.concatenate((temp_x, temp_y), axis=1)), axis=0)

        he_feature_tensor = torch.tensor(he_feature_list)
        ihc_feature_tensor = torch.tensor(ihc_feature_list)
        plus_graph_feature_tensor = torch.tensor(plus_graph_feature_list)

        torch.save(he_feature_tensor, os.path.join(save_dir, 'pt_files', 'HE', sample + '.pt'))
        torch.save(ihc_feature_tensor, os.path.join(save_dir, 'pt_files', 'IHC', sample + '.pt'))
        torch.save(plus_graph_feature_tensor, os.path.join(save_dir, 'pt_files', 'plus_graph', sample + '.pt'))
        h5_file = os.path.join(save_dir, 'h5_files', sample + '.h5py')
        stitch_dir = os.path.join(save_dir, 'stitches')
        with h5py.File(h5_file, 'w') as h5f:
            dataset = h5f.create_dataset("coord_array", data=x_y_list)

        # he_heatmap = StitchCoords(h5_file, he_slideimage, downscale=2, bg_color=(0, 0, 0),
        #                        alpha=-1, draw_grid=False, patch_size=patch_size)
        # ihc_heatmap = StitchCoords(h5_file, ihc_slideimage, downscale=2, bg_color=(0, 0, 0),
        #                        alpha=-1, draw_grid=False, patch_size=patch_size)
        #
        # he_stitch_path = os.path.join(stitch_dir, 'HE', sample + '.jpg')
        # ihc_stitch_path = os.path.join(stitch_dir, 'IHC', sample + '.jpg')
        #
        # he_heatmap.save(he_stitch_path)
        # ihc_heatmap.save(ihc_stitch_path)

def Parser_main():
    parser = argparse.ArgumentParser(description="TEA-graph superpatch generation")
    parser.add_argument("--database", default='SNUH', help="Use in the savedir", type=str)
    parser.add_argument("--cancertype", default='GBM', help="cancer type", type=str)
    parser.add_argument("--magnification", default = '40x', help = "magnification", type = str)
    parser.add_argument("--save_dir", default="/mnt/disk2/TEAgraph_preprocessing/", help = 'root_dir', type = str)
    parser.add_argument("--svs_dir", default="/mnt/disk3/svs_data/", help="svs file location", type=str)
    parser.add_argument("--weight_path", default= "/mnt/disk2/result/Contrastive_learning/2023-12-11_21:52:54/epoch-6,loss-0.618223.pt", help="pretrained weight path")
    parser.add_argument("--patch_size", default=256, help="crop image size", type=int)
    parser.add_argument("--gpu", default='0', help="gpu device number", type=str)
    parser.add_argument("--pretrained_model", default = 'Contrastive', type=str)
    parser.add_argument("--group_num", default = 0, type = int)
    parser.add_argument("--group", default = None , type = int)
    return parser.parse_args()

def main():
    Argument = Parser_main()
    cancer_type = Argument.cancertype
    database = Argument.database
    root_dir = Argument.save_dir
    if os.path.exists(root_dir) is False:
        os.mkdir(root_dir)
    database_dir = os.path.join(root_dir, database)
    if os.path.exists(database_dir) is False:
        os.mkdir(database_dir)
    cancer_type_dir = os.path.join(database_dir, cancer_type)
    if os.path.exists(cancer_type_dir) is False:
        os.mkdir(cancer_type_dir)
    magnification_dir = os.path.join(cancer_type_dir, Argument.magnification)
    if os.path.exists(magnification_dir) is False:
        os.mkdir(magnification_dir)
    patch_size_dir = os.path.join(magnification_dir, str(Argument.patch_size))
    if os.path.exists(patch_size_dir) is False:
        os.mkdir(patch_size_dir)
    pretrained_model_dir = os.path.join(patch_size_dir, Argument.pretrained_model)
    if os.path.exists(pretrained_model_dir) is False:
        os.mkdir(pretrained_model_dir)
    feature_dir = os.path.join(pretrained_model_dir, 'feature')
    h5_dir = os.path.join(feature_dir, 'h5_files')
    pt_dir = os.path.join(feature_dir, 'pt_files')
    he_dir = os.path.join(pt_dir, 'HE')
    ihc_dir = os.path.join(pt_dir, 'IHC')
    plus_dir = os.path.join(pt_dir, 'plus_graph')
    stitch_dir = os.path.join(feature_dir, 'stitches')
    stitch_he_dir = os.path.join(stitch_dir, 'HE')
    stitch_ihc_dir = os.path.join(stitch_dir, 'IHC')

    gpu = Argument.gpu
    weight_path = Argument.weight_path

    svs_dir = os.path.join(Argument.svs_dir, database)
    svs_dir = os.path.join(svs_dir, cancer_type)
    svs_dir = os.path.join(svs_dir, 'registered_image')

    if os.path.exists(feature_dir) is False:
        os.mkdir(feature_dir)
        os.mkdir(h5_dir)
        os.mkdir(pt_dir)
        os.mkdir(he_dir)
        os.mkdir(ihc_dir)
        os.mkdir(plus_dir)
        os.mkdir(stitch_dir)
        os.mkdir(stitch_he_dir)
        os.mkdir(stitch_ihc_dir)

    svs_file_list = os.listdir(svs_dir)

    svs_sample_list = list(set([('_').join(svs.split('_')[0:2]) for svs in svs_file_list]))
    final_files = [os.path.join(svs_dir, sample + '_H&E_registered.ome.tiff') for sample in svs_sample_list]
    final_files.sort(key=lambda f: os.stat(f).st_size, reverse=True)
    if Argument.group_num > 1:
        group_list = [i % Argument.group_num for i in range(len(final_files))]
        processing_df = pd.DataFrame({'file_location' : final_files, 'group' : group_list})
        final_files = processing_df[processing_df['group'] == Argument.group]['file_location'].tolist()

    device = torch.device(int(gpu) if torch.cuda.is_available() else "cpu")
    he_model_ft = EfficientNet.from_pretrained('efficientnet-b4', num_classes=2)
    ihc_model_ft = EfficientNet.from_pretrained('efficientnet-b4', num_classes = 2)
    load_weight = torch.load(weight_path, map_location=device)
    load_weight_list = list(load_weight.keys())
    ihc_model = {}
    he_model = {}
    for item in load_weight_list:
        if 'ihc_model' in item:
            ihc_model[item.split('ihc_model.')[-1]] = load_weight[item]
        elif 'he_model' in item:
            he_model[item.split('he_model.')[-1]] = load_weight[item]
    del load_weight

    he_model_ft.load_state_dict(he_model)
    ihc_model_ft.load_state_dict(ihc_model)

    Argument.transform = transforms.Compose([
    transforms.Resize(320),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    he_model_ft = he_model_ft.to(device)
    ihc_model_ft = ihc_model_ft.to(device)

    he_model_ft.eval()
    ihc_model_ft.eval()

    with tqdm(total=len(final_files)) as pbar_tot:
        for image in final_files:
            sample = ('_').join(image.split('/')[-1].split('_')[0:2])
            feature_extraction(sample, he_model_ft, ihc_model_ft, device, Argument, feature_dir)
            pbar_tot.update()

if __name__ == "__main__":
    main()