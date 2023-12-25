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


def feature_extraction(image, model_ft, device, Argument, save_dir):
    if os.path.exists(save_dir) is False:
        os.mkdir(save_dir)
    sample = image.split('/')[-1].split('.svs')[0]
    print(sample)

    image_path = image
    try:
        slideimage = osd.OpenSlide(image_path)
    except:
        print('openslide error')
        return 0

    downsampling = slideimage.level_downsamples
    if len(downsampling) > 2:
        best_downsampling_level = 2 # downsampling_level 2 means convert 40x -> 10x
        downsampling_factor = int(slideimage.level_downsamples[best_downsampling_level])

        # Get the image at the requested scale
        svs_native_levelimg = slideimage.read_region((0, 0), best_downsampling_level, slideimage.level_dimensions[best_downsampling_level])
        svs_native_levelimg = svs_native_levelimg.convert('L')
        img = np.array(svs_native_levelimg)

        # Otsu thresholding
        thresholds = threshold_multiotsu(img)
        regions = np.digitize(img, bins=thresholds)
        regions[regions == 1] = 0
        regions[regions == 2] = 1
        thresh_otsu = regions

        imagesize = Argument.patch_size
        downsampled_size = int(imagesize / downsampling_factor)
        Width = slideimage.dimensions[0]
        Height = slideimage.dimensions[1]
        num_row = int(Height / imagesize) + 1
        num_col = int(Width / imagesize) + 1
        feature_list = []
        x_y_list = []
        counter = 0
        inside_counter = 0
        temp_patch_list = []
        temp_x = []
        temp_y = []

        # Get the patch the otsu threshold value is lower than 0.75 and extract the feature
        with tqdm(total=num_row * num_col) as pbar_image:
            for i in range(0, num_col):
                for j in range(0, num_row):
                    if thresh_otsu.shape[1] >= (i + 1) * downsampled_size:
                        if thresh_otsu.shape[0] >= (j + 1) * downsampled_size:
                            cut_thresh = thresh_otsu[j * downsampled_size:(j + 1) * downsampled_size,
                                         i * downsampled_size:(i + 1) * downsampled_size]
                        else:
                            cut_thresh = thresh_otsu[(j) * downsampled_size:thresh_otsu.shape[0],
                                         i * downsampled_size:(i + 1) * downsampled_size]
                    else:
                        if thresh_otsu.shape[0] >= (j + 1) * downsampled_size:
                            cut_thresh = thresh_otsu[j * downsampled_size:(j + 1) * downsampled_size,
                                         (i) * downsampled_size:thresh_otsu.shape[1]]
                        else:
                            cut_thresh = thresh_otsu[(j) * downsampled_size:thresh_otsu.shape[0],
                                         (i) * downsampled_size:thresh_otsu.shape[1]]

                    if np.mean(cut_thresh) > 0.75:
                        pbar_image.update()
                        pass
                    else:
                        filter_location = (i * imagesize, j * imagesize)
                        level = 0
                        patch_size = (imagesize, imagesize)
                        location = (filter_location[0], filter_location[1])

                        CutImage = slideimage.read_region(location, level, patch_size)
                        temp_patch_list.append(CutImage)
                        temp_x.append(i * imagesize)
                        temp_y.append(j * imagesize)
                        counter += 1
                        batchsize = 256

                        if counter == batchsize:

                            Dataset = SlidePatchDataset(temp_patch_list, temp_x, temp_y, Argument.transform)
                            dataloader = torch.utils.data.DataLoader(Dataset, batch_size=batchsize, num_workers=0,
                                                                     drop_last=False)
                            for sample_img in dataloader:
                                images = sample_img['image']
                                images = images.to(device)
                                with torch.set_grad_enabled(False):
                                    if Argument.pretrained_model == 'Efficientnet':
                                        classifier, features = model_ft(images)
                                    elif Argument.pretrained_model == 'resnet_pretrained':
                                        features = model_ft(images)

                            if inside_counter == 0:
                                if Argument.pretrained_model == 'Efficientnet':
                                    feature_list = np.concatenate((features.cpu().detach().numpy(),
                                                                   classifier.cpu().detach().numpy()), axis=1)
                                elif Argument.pretrained_model == 'resnet_pretrained':
                                    feature_list = features.cpu().detach().numpy()
                                temp_x = np.reshape(np.array(temp_x), (len(temp_x), 1))
                                temp_y = np.reshape(np.array(temp_y), (len(temp_x), 1))

                                x_y_list = np.concatenate((temp_x, temp_y), axis=1)
                            else:
                                if Argument.pretrained_model == 'Efficientnet':
                                    feature_list = np.concatenate((feature_list,
                                                               np.concatenate((features.cpu().detach().numpy(),
                                                                               classifier.cpu().detach().numpy()),
                                                                              axis=1)), axis=0)
                                elif Argument.pretrained_model == 'resnet_pretrained':
                                    feature_list = np.concatenate((feature_list, features.cpu().detach().numpy()), axis = 0)
                                temp_x = np.reshape(np.array(temp_x), (len(temp_x), 1))
                                temp_y = np.reshape(np.array(temp_y), (len(temp_x), 1))

                                x_y_list = np.concatenate((x_y_list,
                                                           np.concatenate((temp_x, temp_y), axis=1)), axis=0)
                            inside_counter += 1
                            temp_patch_list = []
                            temp_x = []
                            temp_y = []
                            counter = 0

                        pbar_image.update()

            if counter < batchsize and counter > 0:
                Dataset = SlidePatchDataset(temp_patch_list, temp_x, temp_y, Argument.transform)
                dataloader = torch.utils.data.DataLoader(Dataset, batch_size=batchsize, num_workers=0, drop_last=False)
                for sample_img in dataloader:
                    images = sample_img['image']
                    images = images.to(device)
                    with torch.set_grad_enabled(False):
                        if Argument.pretrained_model == 'Efficientnet':
                            classifier, features = model_ft(images)
                        elif Argument.pretrained_model == 'resnet_pretrained':
                            features = model_ft(images)

                    if Argument.pretrained_model == 'Efficientnet':
                        feature_list = np.concatenate((feature_list,
                                                       np.concatenate((features.cpu().detach().numpy(),
                                                                       classifier.cpu().detach().numpy()),
                                                                      axis=1)), axis=0)
                    elif Argument.pretrained_model == 'resnet_pretrained':
                        feature_list = np.concatenate((feature_list, features.cpu().detach().numpy()), axis=0)

                    temp_x = np.reshape(np.array(temp_x), (len(temp_x), 1))
                    temp_y = np.reshape(np.array(temp_y), (len(temp_x), 1))

                    x_y_list = np.concatenate((x_y_list,
                                               np.concatenate((temp_x, temp_y), axis=1)), axis=0)

        feature_tensor = torch.tensor(feature_list)
        torch.save(feature_tensor, os.path.join(save_dir, 'pt_files', sample + '.pt'))
        h5_file = os.path.join(save_dir, 'h5_files', sample + '.h5py')
        stitch_dir = os.path.join(save_dir, 'stitches')
        with h5py.File(h5_file, 'w') as h5f:
            dataset = h5f.create_dataset("coord_array", data=x_y_list)

        heatmap = StitchCoords(h5_file, slideimage, downscale=best_downsampling_level, bg_color=(0, 0, 0),
                               alpha=-1, draw_grid=False, patch_size=patch_size)
        stitch_path = os.path.join(stitch_dir, sample + '_H&E.jpg')
        heatmap.save(stitch_path)

def Parser_main():
    parser = argparse.ArgumentParser(description="TEA-graph superpatch generation")
    parser.add_argument("--database", default='TCGA', help="Use in the savedir", type=str)
    parser.add_argument("--cancertype", default='KIRC', help="cancer type", type=str)
    parser.add_argument("--magnification", default = '40x', help = "magnification", type = str)
    parser.add_argument("--save_dir", default="/mnt/disk2/TEAgraph_preprocessing/", help = 'root_dir', type = str)
    parser.add_argument("--svs_dir", default="/mnt/disk3/svs_data/", help="svs file location", type=str)
    parser.add_argument("--weight_path", default="/mnt/disk2/DSA_model/epoch-0,loss-0.012936,accuracy-0.094789.pt", help="pretrained weight path")
    parser.add_argument("--patch_size", default=256, help="crop image size", type=int)
    parser.add_argument("--gpu", default='0', help="gpu device number", type=str)
    parser.add_argument("--pretrained_model", default = 'Efficientnet', type=str)
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
    stitch_dir = os.path.join(feature_dir, 'stitches')

    gpu = Argument.gpu
    weight_path = Argument.weight_path

    svs_dir = os.path.join(Argument.svs_dir, database)
    svs_dir = os.path.join(svs_dir, cancer_type)

    if os.path.exists(feature_dir) is False:
        os.mkdir(feature_dir)
        os.mkdir(h5_dir)
        os.mkdir(pt_dir)
        os.mkdir(stitch_dir)

    svs_file_list = os.listdir(svs_dir)

    svs_sample_list = [svs.split('.svs')[0] for svs in svs_file_list]
    final_files = [os.path.join(svs_dir, sample + '.svs') for sample in svs_sample_list]
    final_files.sort(key=lambda f: os.stat(f).st_size, reverse=True)
    if Argument.group_num > 1:
        group_list = [i % Argument.group_num for i in range(len(final_files))]
        processing_df = pd.DataFrame({'file_location' : final_files, 'group' : group_list})
        final_files = processing_df[processing_df['group'] == Argument.group]['file_location'].tolist()

    device = torch.device(int(gpu) if torch.cuda.is_available() else "cpu")

    if Argument.pretrained_model == 'Efficientnet':
        model_ft = EfficientNet.from_pretrained('efficientnet-b4', num_classes=2)
        if weight_path is not None:
            load_weight = torch.load(weight_path, map_location=device)
            model_ft.load_state_dict(load_weight)
        Argument.transform = transforms.Compose([
        transforms.Resize(320),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    model_ft = model_ft.to(device)
    model_ft.eval()

    with tqdm(total=len(final_files)) as pbar_tot:
        for image in final_files:
            feature_extraction(image, model_ft, device, Argument, feature_dir)
            pbar_tot.update()

if __name__ == "__main__":
    main()