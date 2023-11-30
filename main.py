# -*- coding: utf-8 -*-

import os
import sys
import argparse
from train import Train
import wandb

def Parser_main():
    parser = argparse.ArgumentParser(description="Deep cox analysis model")
    parser.add_argument("--DatasetType", default="SNUH", help="TCGA_BRCA or BORAME or BORAME_Meta or BORAME_Prog", type=str)
    parser.add_argument("--CancerType", default = "GBM", help= "Cancer type")
    parser.add_argument("--learning_rate", default=0.0001, help="Learning rate", type=float)
    parser.add_argument("--weight_decay", default=0.00005, help="Weight decay rate", type=float)
    parser.add_argument("--clip_grad_norm_value", default=2.0, help="Gradient clipping value", type=float)
    parser.add_argument("--batch_size", default=6, help="batch size", type=int)
    parser.add_argument("--num_epochs", default=1, help="Number of epochs", type=int)
    parser.add_argument("--dropedge_rate", default=0.25, help="Dropedge rate for GAT", type=float)
    parser.add_argument("--dropout_rate", default=0.25, help="Dropout rate for MLP", type=float)
    parser.add_argument("--graph_dropout_rate", default=0.25, help="Node/Edge feature dropout rate", type=float)
    parser.add_argument("--initial_dim", default=100, help="Initial dimension for the GAT", type=int)
    parser.add_argument("--attention_head_num", default=2, help="Number of attention heads for GAT", type=int)
    parser.add_argument("--number_of_layers", default=3, help="Whole number of layer of GAT", type=int)
    parser.add_argument("--FF_number", default=5, help="Selecting set for the five fold cross validation", type=int)
    parser.add_argument("--model", default="GAT_custom", help="GAT_custom/DeepGraphConv/PatchGCN/GIN/MIL/MIL-attention", type=str)
    parser.add_argument("--gpu", default = 2, help="Target gpu for calculating loss value", type=int)
    parser.add_argument("--norm_type", default="layer", help="BatchNorm=batch/LayerNorm=layer", type=str)
    parser.add_argument("--MLP_layernum", default=3, help="Number of layers for pre/pose-MLP", type=int)
    parser.add_argument("--with_distance", default="Y", help="Y/N; Including positional information as edge feature", type=str)
    parser.add_argument("--simple_distance", default="N", help="Y/N; Whether multiplying or embedding positional information", type=str)
    parser.add_argument("--loss_type", default="PRELU", help="RELU/Leaky/PRELU", type=str)
    parser.add_argument("--feature_type", default = 'only_HE', help= 'feature_type', type = str)
    parser.add_argument("--spatial_threshold", default = '0.75', help = 'Spatial_threshold', type = str)
    parser.add_argument("--residual_connection", default="Y", help="Y/N", type=str)

    parser.add_argument("--other", action = 'store_true', default = False)
    parser.add_argument("--without_biopsy", action='store_true', default=False)
    parser.add_argument("--gbl", action = 'store_true', default = False)
    parser.add_argument("--pretrain", default = 'Efficientnet', type = str)
    parser.add_argument("--dataset_type", default = 'ki-67', type = str)
    parser.add_argument("--random_seed", default = 1234567, type = int)
    parser.add_argument("--loss", default='CoxPH', type=str, help='Loss function(CoxPH, nnet, nll)')
    parser.add_argument("--duration", default = 0, type = int, help = 'Target duration (1, 5, all)')

    parser.add_argument("--wandb", action = 'store_true', default= False)
    parser.add_argument("--save", action = 'store_true', default = False)

    return parser.parse_args()

def main():
    Argument = Parser_main()
    best_model, bestepoch = Train(Argument)

if __name__ == "__main__":
    main()
