# Reference:
import h5py
import numpy as np
import os
import pdb
from PIL import Image
import math
import cv2
import openslide as osd


def DrawGrid(img, coord, shape, thickness=2, color=(0, 0, 0, 255)):
    cv2.rectangle(img, tuple(np.maximum([0, 0], coord - thickness // 2)),
                  tuple(coord - thickness // 2 + np.array(shape)), (0, 0, 0, 255), thickness=thickness)
    return img


def DrawMapFromCoords(canvas, wsi, coords, patch_size, downscale, indices=None, verbose=1, draw_grid=True):
    # downsamples = wsi_object.wsi.level_downsamples[0]
    if indices is None:
        indices = np.arange(len(coords))
    total = len(indices)
    if verbose > 0:
        ten_percent_chunk = math.ceil(total * 0.1)

    min_x = min(set(coords[:, 0]))
    min_y = min(set(coords[:, 1]))

    max_x = max(set(coords[:, 0]))
    max_y = max(set(coords[:, 1]))

    patch_size = tuple(np.ceil((np.array(patch_size) / np.array(downscale))).astype(np.int32))
    print('downscaled patch size: {}x{}'.format(patch_size[0], patch_size[1]))
    print(canvas.shape)
    for idx in range(total):
        if verbose > 0:
            if idx % ten_percent_chunk == 0:
                print('progress: {}/{} stitched'.format(idx, total))

        patch_id = indices[idx]
        patch_coord = coords[patch_id]
        coord = np.ceil(patch_coord / downscale).astype(np.int32)

        patch = np.array(wsi.read_region(tuple(patch_coord), 2, patch_size).convert("RGB"))
        # print(patch.shape, canvas[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0], :3])
        canvas_crop_shape = canvas[coord[1]:coord[1] + patch_size[1], coord[0]:coord[0] + patch_size[0], :3].shape[:2]
        # print(coord[1], coord[1]+patch_size[1], coord[0], coord[0]+patch_size[0])
        # canvas[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0], :3] = patch
        canvas[coord[1]:coord[1] + patch_size[1], coord[0]:coord[0] + patch_size[0], :3] = patch[:canvas_crop_shape[0],
                                                                                           :canvas_crop_shape[1], :]
        if draw_grid:
            DrawGrid(canvas, coord, patch_size)

    return Image.fromarray(canvas)


def StitchCoords(hdf5_file_path, wsi, downscale, draw_grid=False, bg_color=(0, 0, 0), alpha=-1, patch_size=256):
    file = h5py.File(hdf5_file_path, 'r')
    coords = file['coord_array'][()]

    # dset = file['coords']
    # coords = dset[:]
    w, h = wsi.level_dimensions[downscale]
    downsampling_factor = wsi.level_downsamples[downscale]

    print('downscaled size for stiching: {} x {}'.format(w, h))
    print('number of patches: {}'.format(len(coords)))

    # patch_size = dset.attrs['patch_size']
    # patch_level = dset.attrs['patch_level']
    # print('patch size: {}x{} patch level: {}'.format(patch_size, patch_size, patch_level))
    # patch_size = tuple(np.array((patch_size, patch_size)).astype(np.int32))
    # print('ref patch size: {}x{}'.format(patch_size, patch_size))

    if alpha < 0 or alpha == -1:
        heatmap = Image.new(size=(w, h), mode="RGB", color=bg_color)
    else:
        heatmap = Image.new(size=(w, h), mode="RGBA", color=bg_color + (int(255 * alpha),))

    heatmap = np.array(heatmap)
    heatmap = DrawMapFromCoords(heatmap, wsi, coords, patch_size, downsampling_factor, indices=None, draw_grid=draw_grid)

    file.close()
    return heatmap