# -*- coding: utf-8 -*-

import argparse
from train import Train_cross_validation, Train
import numpy as np
from interpretation import IG_analysis

def Parser_main():
    parser = argparse.ArgumentParser(description="Deep cox analysis model")
    parser.add_argument("--mode", default = 'IG_analysis', type = str)
    parser.add_argument("--DatasetType", default="TCGA", help="TCGA_BRCA or BORAME or BORAME_Meta or BORAME_Prog", type=str)
    parser.add_argument("--CancerType", default = "KIRC", help= "Cancer type")
    parser.add_argument("--meta_root", default = "/mnt/disk2/metadata/", help = 'Metadata root', type = str)
    parser.add_argument("--graph_root", default = "/mnt/disk2/TEAgraph_preprocessing/", help = 'Graphdata root', type = str)
    parser.add_argument("--save_dir", default = "/mnt/disk2/result/TEA-graph", type = str)
    parser.add_argument("--magnification", default = "40x", help = 'resolution', type = str)
    parser.add_argument("--patch_size", default = '256', help = 'patch_size', type = str)
    parser.add_argument("--pretrain", default = 'Efficientnet', type = str)
    parser.add_argument("--feature_type", default = 'only_HE', help= 'feature_type', type = str)
    parser.add_argument("--spatial_threshold", default='0.75', help='Spatial_threshold', type=str)
    parser.add_argument("--feature_size", default = 1792, type = int)

    parser.add_argument("--learning_rate", default=0.0001, help="Learning rate", type=float)
    parser.add_argument("--weight_decay", default=0.00005, help="Weight decay rate", type=float)
    parser.add_argument("--clip_grad_norm_value", default=2.0, help="Gradient clipping value", type=float)
    parser.add_argument("--batch_size", default=6, help="batch size", type=int)
    parser.add_argument("--num_epochs", default=1, help="Number of epochs", type=int)
    parser.add_argument("--FF_number", default=5, help="Selecting set for the five fold cross validation", type=int)
    parser.add_argument("--model", default="GAT_custom", help="GAT_custom/DeepGraphConv/PatchGCN/GIN/MIL/MIL-attention", type=str)
    parser.add_argument("--gpu", default=2, help="Target gpu for calculating loss value", type=int)
    parser.add_argument("--random_seed", default = 1234567, type = int)
    parser.add_argument("--loss", default='CoxPH', type=str, help='Loss function(CoxPH, nnet, nll)')
    parser.add_argument("--sampler", action='store_true', default = True)

    parser.add_argument("--dropedge_rate", default=0.25, help="Dropedge rate for GAT", type=float)
    parser.add_argument("--dropout_rate", default=0.25, help="Dropout rate for MLP", type=float)
    parser.add_argument("--graph_dropout_rate", default=0.25, help="Node/Edge feature dropout rate", type=float)
    parser.add_argument("--initial_dim", default=100, help="Initial dimension for the GAT", type=int)
    parser.add_argument("--attention_head_num", default=2, help="Number of attention heads for GAT", type=int)
    parser.add_argument("--number_of_layers", default=3, help="Whole number of layer of GAT", type=int)
    parser.add_argument("--norm_type", default="layer", help="BatchNorm=batch/LayerNorm=layer", type=str)
    parser.add_argument("--MLP_layernum", default=3, help="Number of layers for pre/pose-MLP", type=int)
    parser.add_argument("--with_distance", default="Y", help="Y/N; Including positional information as edge feature", type=str)
    parser.add_argument("--simple_distance", default="N", help="Y/N; Whether multiplying or embedding positional information", type=str)
    parser.add_argument("--residual_connection", default="Y", help="Y/N", type=str)
    parser.add_argument("--loss_type", default="PRELU", help="RELU/Leaky/PRELU", type=str)

    parser.add_argument("--wandb", action = 'store_true', default= False)
    parser.add_argument("--save", action = 'store_true', default = False)
    parser.add_argument("--cross_val", action = 'store_true', default = True)

    parser.add_argument("--analysis_path", default = None, type = str)
    return parser.parse_args()

def main():
    Argument = Parser_main()
    if Argument.mode == 'Train':
        if Argument.cross_val:
            result_df = Train_cross_validation(Argument)
            mean = np.mean(result_df['C-index'])
            std_dev = np.std(result_df['C-index'])
            print(str(Argument.FF_number) + " fold cross validation c-index Average: ", mean)
            print(str(Argument.FF_number) + " fold cross validation c-index Standard Deviation: ", std_dev)
        else:
            best_acc = Train(Argument)
            print("Best test c-index: ", best_acc)
    elif Argument.mode == 'IG_analysis':
        IG_analysis(Argument)

if __name__ == "__main__":
    main()
