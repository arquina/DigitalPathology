import torch
import numpy as np
import lifelines.utils.concordance as LUC
import torch.nn as nn
import torch.nn.functional as F
import os
import datetime
import pytz

from models.GAT import GAT
from models.MLP import MLP
from models.GAT_custom import GAT as GAT_custom
from models.MLP_attention import MLP_attention as AttnMLP
from models.DeepGraphConv import DeepGraphConv_Surv as DeepGraphConv
from models.PatchGCN import PatchGCN
from torch_geometric.nn import GATConv as GATConv_v1
from torch_geometric.nn import GATv2Conv as GATConv

def model_selection(Argument):

    if Argument.model == "GAT":
        model = GAT(Argument.dropout_rate, Argument.dropedge_rate, Argument)
    elif Argument.model == "GAT_custom":
        model = GAT_custom(Argument.dropout_rate, Argument.dropedge_rate, Argument)
    elif Argument.model == "MLP":
        model = MLP(Argument.dropout_rate, Argument.dropedge_rate, Argument)
    elif Argument.model == "AttMLP":
        model = AttnMLP(Argument.dropout_rate, Argument.dropedge_rate, Argument)
    elif Argument.model == "PatchGCN":
        model = PatchGCN(Argument.dropout_rate, Argument.dropedge_rate, Argument)
    elif Argument.model == "DeepGraphConv":
        model = DeepGraphConv(Argument.dropout_rate, Argument.dropedge_rate, Argument)
    else:
        print("Enter the valid model type")
        model = None

    return model

def makecheckpoint_dir_graph(Argument):
    todaydata = datetime.datetime.now(pytz.timezone('Asia/Seoul')).strftime("%Y-%m-%d_%H:%M:%S")

    checkpoint_dir = os.path.join(Argument.save_dir, Argument.DatasetType)
    if os.path.exists(checkpoint_dir) is False:
        os.mkdir(checkpoint_dir)
    checkpoint_dir = os.path.join(checkpoint_dir, Argument.CancerType)
    if os.path.exists(checkpoint_dir) is False:
        os.mkdir(checkpoint_dir)
    checkpoint_dir = os.path.join(checkpoint_dir, Argument.model)
    if os.path.exists(checkpoint_dir) is False:
        os.mkdir(checkpoint_dir)
    checkpoint_dir = os.path.join(checkpoint_dir, todaydata)
    if os.path.exists(checkpoint_dir) is False:
        os.makedirs(checkpoint_dir)

    return checkpoint_dir


def ce_loss(hazards,Y, c, device, alpha = 0.7, eps = 1e-7):
    batch_size = len(Y)
    Y = Y.view(batch_size, 1).to(device)
    c = c.view(batch_size, 1).to(device)
    S = torch.cumprod(1 - hazards, dim = 1).to(device)
    S_padded = torch.cat([torch.ones_like(c), S], 1).to(device)
    reg = -(c) * (torch.log(torch.gather(S_padded, 1, Y) +eps) + torch.log(torch.gather(hazards, 1, Y).clamp(min = eps)))
    ce_l = -(1-c) * torch.log(torch.gather(S, 1, Y).clamp(min = eps)) - (c) * torch.log(1- torch.gather(S,1,Y).clamp(min = eps))
    loss = (1-alpha) * ce_l + alpha * reg
    loss = loss.mean()
    return loss

def cox_sort(out, tempsurvival, tempphase):

    sort_idx = torch.argsort(tempsurvival, descending=True)

    risklist = out[sort_idx]
    tempsurvival = tempsurvival[sort_idx]
    tempphase = tempphase[sort_idx]

    risklist = risklist.to(out.device)
    tempsurvival = tempsurvival.to(out.device)
    tempphase = tempphase.to(out.device)

    return risklist, tempsurvival, tempphase

def accuracytest(survivals, risk, censors):
    survlist = []
    risklist = []
    censorlist = []

    for riskval in risk:
        risklist.append(riskval.cpu().detach().item())

    for censorval in censors:
        censorlist.append(censorval.cpu().detach().item())

    for surval in survivals:
        survlist.append(surval.cpu().detach().item())

    C_value = LUC.concordance_index(survlist, -np.exp(risklist), censorlist)

    return C_value

class coxph_loss(torch.nn.Module):

    def __init__(self):
        super(coxph_loss, self).__init__()

    def forward(self, risk, phase, censors):

        #riskmax = risk
        riskmax = F.normalize(risk, p=2, dim=0)
        log_risk = torch.log((torch.cumsum(torch.exp(riskmax), dim=0)))

        uncensored_likelihood = torch.add(riskmax, -log_risk)
        resize_censors = censors.resize_(uncensored_likelihood.size()[0], 1)
        censored_likelihood = torch.mul(uncensored_likelihood, resize_censors)

        loss = -torch.sum(censored_likelihood) / float(censors.nonzero().size(0))
        #loss = -torch.sum(censored_likelihood) / float(censors.size(0))

        return loss

def non_decay_filter(model):

    no_decay = list()
    decay = list()

    for m in model.modules():
        if isinstance(m, nn.Linear):
            decay.append(m.weight)
            if m.bias != None:
                no_decay.append(m.bias)
        elif isinstance(m, nn.BatchNorm1d):
            no_decay.append(m.weight)
            if m.bias != None:
                no_decay.append(m.bias)
        elif isinstance(m, nn.LayerNorm):
            no_decay.append(m.weight)
            if m.bias != None:
                no_decay.append(m.bias)
        elif isinstance(m, nn.PReLU):
            no_decay.append(m.weight)
        elif isinstance(m, GATConv):
            decay.append(m.att)
            if m.bias != None:
                no_decay.append(m.bias)
            no_decay.append(m.position_bias)
        elif isinstance(m, GATConv_v1):
            decay.append(m.att_l)
            decay.append(m.att_r)
            if m.bias != None:
                no_decay.append(m.bias)
            no_decay.append(m.position_bias)
            no_decay.append(m.angle_bias)
            decay.append(m.att_edge_attr_pos)
            decay.append(m.att_edge_attr_angle)

    model_parameter_groups = [dict(params=decay), dict(params=no_decay, weight_decay=0.0)]

    return model_parameter_groups

def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        nn.init.normal_(m.weight.data)
        if m.bias is not None:
            nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        nn.init.normal_(m.weight.data)
        if m.bias is not None:
            nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.normal_(m.weight.data, mean=1, std=0.02)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, mean=1, std=0.02)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        nn.init.normal_(m.weight.data, mean=1, std=0.02)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data)
        nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.normal_(param.data)