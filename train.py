# -*- coding: utf-8 -*-

import os
import torch
import pandas as pd

import wandb

from torch import optim
from torch_geometric.data import DataListLoader
from torch_geometric.nn import DataParallel
from torch.optim.lr_scheduler import OneCycleLR

from tqdm import tqdm

from train_utils import accuracytest, cox_sort, coxph_loss, makecheckpoint_dir_graph, model_selection
from train_utils import non_decay_filter
from DataLoader import Sampler_custom, add_kfold_to_df, CoxGraphDataset, metadata_list_generation
import numpy as np

def Train_cross_validation(Argument):
    if Argument.save:
        checkpoint_dir = makecheckpoint_dir_graph(Argument)
    batch_num = int(Argument.batch_size)
    device = torch.device(int(Argument.gpu))
    metadata_root = os.path.join(Argument.meta_root, Argument.DatasetType, Argument.CancerType)
    Metadata = pd.read_csv(os.path.join(metadata_root, Argument.CancerType + '_clinical.tsv'), sep='\t')

    TrainRoot = os.path.join(Argument.graph_root, Argument.DatasetType, Argument.CancerType, Argument.magnification, Argument.patch_size, Argument.pretrain, 'graph', 'pt_files')
    Trainlist = os.listdir(TrainRoot)
    Trainlist, Train_survivallist, Train_censorlist, Train_stagelist = metadata_list_generation(Argument.DatasetType, Trainlist, Metadata)

    train_df = pd.DataFrame(zip(Trainlist, Train_survivallist, Train_censorlist, Train_stagelist), columns = ['File', 'Survival', 'Censor', 'Stage'])
    Fi = Argument.FF_number
    Argument.n_intervals = 1
    _ = add_kfold_to_df(train_df, Fi, Argument.random_seed)

    Argument.n_intervals = 1
    Fi = Argument.FF_number

    FFCV_accuracy = []
    FFCV_best_epoch = []
    for fold in range(Fi):
        if Argument.wandb:
            run = wandb.init(reinit=True, project='TEA-graph', group=checkpoint_dir.split('/')[-1])
            if Argument.DatasetType == 'SNUH':
                if Argument.without_biopsy:
                    wandb.run.name = checkpoint_dir.split('/')[-1] + '_total_only_' + Argument.pretrain + '_' + Argument.dataset_type + str(fold)
                else:
                    wandb.run.name = checkpoint_dir.split('/')[-1] + '_all_' + Argument.pretrain + '_' + Argument.dataset_type + str(fold)
            elif Argument.DatasetType == 'TCGA':
                wandb.run.name = checkpoint_dir.split('/')[-1] + Argument.pretrain + '_' + Argument.DatasetType + str(fold)
            wandb.run.save()

        if Argument.save:
            fold_checkpoint_dir = os.path.join(checkpoint_dir, str(fold))
            if os.path.exists(fold_checkpoint_dir) is False:
                os.mkdir(fold_checkpoint_dir)

        Train_df = train_df[train_df['kfold'] != fold]
        Val_df = train_df[train_df['kfold'] == fold]
        if Argument.save:
            Train_df.to_csv(os.path.join(fold_checkpoint_dir, 'Train_dataset.csv'), index=False)
            Val_df.to_csv(os.path.join(fold_checkpoint_dir, 'Validation_dataset.csv'), index=False)

        print(str(Fi))
        print('Train data: ', str(len(Train_df)))
        print('Val data: ', str(len(Val_df)))
        TrainDataset = CoxGraphDataset(df=Train_df, root_dir=TrainRoot, feature_size = Argument.feature_size)
        ValidDataset = CoxGraphDataset(df=Val_df, root_dir=TrainRoot, feature_size = Argument.feature_size)

        torch.manual_seed(Argument.random_seed)
        if Argument.sampler:
            Event_idx = np.where(np.array(Train_df['Censor']) == 1)[0]
            Censored_idx = np.where(np.array(Train_df['Censor']) == 0)[0]
            train_batch_sampler = Sampler_custom(Event_idx, Censored_idx, batch_num)
            train_loader = DataListLoader(TrainDataset, batch_sampler=train_batch_sampler, num_workers=8, pin_memory=True)
        else:
            train_loader = DataListLoader(TrainDataset, batch_size=batch_num, shuffle=True, num_workers=8, pin_memory=True)
        val_loader = DataListLoader(ValidDataset, batch_size=batch_num, shuffle=False, num_workers=8, pin_memory=True)

        model = model_selection(Argument)
        model_parameter_groups = non_decay_filter(model)

        if Argument.gpu == 0:
            model = DataParallel(model, device_ids=[0, 1], output_device=0)
        elif Argument.gpu == 2:
            model = DataParallel(model, device_ids=[2, 3], output_device=2)

        model = model.to(device)
        if Argument.wandb:
            wandb.watch(model, log="all", log_freq=1)

        if Argument.loss == 'CoxPH':
            criterion = coxph_loss()
            criterion = criterion.to(device)

        optimizer_ft = optim.AdamW(model_parameter_groups, lr=Argument.learning_rate,
                                   weight_decay=Argument.weight_decay)
        scheduler = OneCycleLR(optimizer_ft, max_lr=Argument.learning_rate, steps_per_epoch=len(train_loader),
                               epochs=Argument.num_epochs)
        bestloss = 100000
        bestacc = 0
        bestepoch = 0

        loader = {'train': train_loader, 'val': val_loader}
        BestAccDict = {'train': 0, 'val': 0}

        with tqdm(total=Argument.num_epochs) as pbar:
            for epoch in range(0, int(Argument.num_epochs)):
                phaselist = ['train', 'val']
                for mode in phaselist:
                    if mode == 'train':
                        model.train()
                        grad_flag = True
                    else:
                        model.eval()
                        grad_flag = False
                    with torch.set_grad_enabled(grad_flag):
                        EpochSurv = []
                        EpochPhase = []
                        EpochRisk = []
                        EpochID = []
                        Epochloss = 0
                        batchcounter = 1
                        pass_count = 0
                        for c, d in enumerate(loader[mode], 1):
                            optimizer_ft.zero_grad()
                            tempsurvival = torch.tensor([data.survival for data in d])
                            tempphase = torch.tensor([data.phase for data in d])
                            tempID = np.asarray([data.item for data in d])
                            out = model(d)
                            EpochSurv.extend(tempsurvival.cpu().detach().tolist())
                            EpochID.extend(tempID.tolist())
                            EpochPhase.extend(tempphase.cpu().detach().tolist())
                            EpochRisk.append(out.detach().cpu().numpy())

                            if torch.sum(tempphase).cpu().detach().item() < 1:
                                pass_count += 1
                            else:
                                if Argument.loss == 'CoxPH':
                                    risklist, tempsurvival, tempphase = cox_sort(out, tempsurvival, tempphase)
                                    loss = criterion(risklist, tempsurvival, tempphase)

                                if mode == 'train':
                                    loss.backward()
                                    torch.nn.utils.clip_grad_norm_(model_parameter_groups[0]['params'],
                                                                   max_norm=Argument.clip_grad_norm_value,
                                                                   error_if_nonfinite=True)
                                    torch.nn.utils.clip_grad_norm_(model_parameter_groups[1]['params'],
                                                                   max_norm=Argument.clip_grad_norm_value,
                                                                   error_if_nonfinite=True)
                                    optimizer_ft.step()
                                    scheduler.step()

                                Epochloss += loss.cpu().detach().item()
                                batchcounter += 1

                        EpochRisk = np.concatenate(EpochRisk)
                        if Argument.loss == 'CoxPH':
                            Epochacc = accuracytest(torch.tensor(EpochSurv), torch.tensor(EpochRisk),
                                                    torch.tensor(EpochPhase))
                        Epochloss = Epochloss / batchcounter

                        if mode == 'train':
                            if Epochacc > BestAccDict['train']:
                                BestAccDict['train'] = Epochacc
                        elif mode == 'val':
                            if Epochacc > BestAccDict['val']:
                                BestAccDict['val'] = Epochacc
                        print()
                        print('epoch:' + str(epoch))
                        print(" mode:" + mode)
                        print(" loss:" + str(Epochloss) + " acc:" + str(Epochacc) + " pass count:" + str(pass_count))

                        if mode == 'train':
                            if Argument.wandb:
                                wandb.log({
                                    'Training Accuracy': Epochacc,
                                    'Training loss': Epochloss
                                }, step=epoch)
                        elif mode == 'val':
                            if Argument.wandb:
                                wandb.log({
                                    'Validation Accuracy': Epochacc,
                                    'Validation loss': Epochloss
                                }, step=epoch)

                        checkpointinfo = 'epoch-{},acc-{:4f},loss-{:4f}.pt'
                        if mode == 'val':
                            if epoch == 0:
                                if Argument.save:
                                    torch.save(model.state_dict(), os.path.join(fold_checkpoint_dir,
                                                                                checkpointinfo.format(epoch, Epochacc,
                                                                                                      Epochloss)))
                            else:
                                if Epochacc > bestacc:
                                    bestepoch = epoch
                                    if epoch > 10:
                                        if Argument.save:
                                            torch.save(model.state_dict(), os.path.join(fold_checkpoint_dir,
                                                                                        checkpointinfo.format(epoch,
                                                                                                              Epochacc,
                                                                                                              Epochloss)))
                                if Epochacc > bestacc:
                                    bestacc = Epochacc
                                if Epochloss < bestloss:
                                    bestloss = Epochloss
                pbar.update()
        FFCV_accuracy.append(bestacc)
        FFCV_best_epoch.append(bestepoch)

    df = pd.DataFrame(list(zip(list(range(Argument.FF_number)), FFCV_accuracy, FFCV_best_epoch)),
                      columns=['Fold', 'C-index', 'Best_epoch'])
    if Argument.save:
        df.to_csv(os.path.join(checkpoint_dir, 'final_result.csv'), index=False)
        bestFi = np.argmax(FFCV_accuracy)
        best_checkpoint_dir = os.path.join(checkpoint_dir, str(bestFi))
        Argument.checkpoint_dir = best_checkpoint_dir

    return df

def Train(Argument):
    checkpoint_dir = makecheckpoint_dir_graph(Argument)
    batch_num = int(Argument.batch_size)
    device = torch.device(int(Argument.gpu))
    metadata_root = os.path.join(Argument.meta_root, Argument.DatasetType, Argument.CancerType)
    Metadata = pd.read_csv(os.path.join(metadata_root, Argument.CancerType + '_clinical.tsv'), sep='\t')

    TrainRoot = os.path.join(Argument.graph_root, Argument.DatasetType, Argument.CancerType)
    Trainlist = os.listdir(TrainRoot)
    Fi = Argument.FF_number
    Argument.n_intervals = 1
    _ = add_kfold_to_df(Metadata, Fi, Argument.random_seed)
    train_batch_sampler = Sampler_custom(Event_idx, Censored_idx, batch_num)
    Argument.n_intervals = 1
    Fi = Argument.FF_number
    _ = add_kfold_to_df(Metadata, Fi, Argument.random_seed)

    FFCV_accuracy = []
    FFCV_best_epoch = []
    for fold in range(Fi):
        if Argument.wandb:
            run = wandb.init(reinit=True, project='TEA-graph', group=checkpoint_dir.split('/')[-1])
            if Argument.without_biopsy:
                wandb.run.name = checkpoint_dir.split('/')[-1] + '_total_only_' + Argument.pretrain + '_' + Argument.dataset_type + str(fold)
            else:
                wandb.run.name = checkpoint_dir.split('/')[-1] + '_all_' + Argument.pretrain + '_' + Argument.dataset_type + str(fold)
            wandb.run.save()

        if Argument.save:
            fold_checkpoint_dir = os.path.join(checkpoint_dir, str(fold))
            if os.path.exists(fold_checkpoint_dir) is False:
                os.mkdir(fold_checkpoint_dir)

        Train_df = Metadata[Metadata['kfold'] != fold]
        Val_df = Metadata[Metadata['kfold'] == fold]
        if Argument.save:
            Train_df.to_csv(os.path.join(fold_checkpoint_dir, 'Train_dataset.csv'), index=False)
            Val_df.to_csv(os.path.join(fold_checkpoint_dir, 'Validation_dataset.csv'), index=False)

        print(str(Fi))
        print('Train data: ', str(len(Train_df)))
        print('Val data: ', str(len(Val_df)))
        TrainDataset = CoxGraphDataset(df=Train_df, root_dir=root_dir)
        ValidDataset = CoxGraphDataset(df=Val_df, root_dir=root_dir)

        torch.manual_seed(Argument.random_seed)
        train_loader = DataListLoader(TrainDataset, batch_size=batch_num, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataListLoader(ValidDataset, batch_size=batch_num, shuffle=False, num_workers=4, pin_memory=True)

        model = model_selection(Argument)
        model_parameter_groups = non_decay_filter(model)

        if Argument.gpu == 0:
            model = DataParallel(model, device_ids=[0, 1], output_device=0)
        elif Argument.gpu == 2:
            model = DataParallel(model, device_ids=[2, 3], output_device=2)

        model = model.to(device)
        if Argument.wandb:
            wandb.watch(model, log="all", log_freq=1)

        if Argument.loss == 'CoxPH':
            criterion = coxph_loss()
            criterion = criterion.to(device)

        optimizer_ft = optim.AdamW(model_parameter_groups, lr=Argument.learning_rate,
                                   weight_decay=Argument.weight_decay)
        scheduler = OneCycleLR(optimizer_ft, max_lr=Argument.learning_rate, steps_per_epoch=len(train_loader),
                               epochs=Argument.num_epochs)
        bestloss = 100000
        bestacc = 0
        bestepoch = 0

        loader = {'train': train_loader, 'val': val_loader}
        BestAccDict = {'train': 0, 'val': 0}

        with tqdm(total=Argument.num_epochs) as pbar:
            for epoch in range(0, int(Argument.num_epochs)):
                phaselist = ['train', 'val']
                for mode in phaselist:
                    if mode == 'train':
                        model.train()
                        grad_flag = True
                    else:
                        model.eval()
                        grad_flag = False
                    with torch.set_grad_enabled(grad_flag):
                        EpochSurv = []
                        EpochPhase = []
                        EpochRisk = []
                        EpochID = []
                        Epochloss = 0
                        batchcounter = 1
                        pass_count = 0
                        for c, d in enumerate(loader[mode], 1):
                            optimizer_ft.zero_grad()
                            tempsurvival = torch.tensor([data.survival for data in d])
                            tempphase = torch.tensor([data.phase for data in d])
                            tempID = np.asarray([data.item for data in d])
                            out = model(d)
                            EpochSurv.extend(tempsurvival.cpu().detach().tolist())
                            EpochID.extend(tempID.tolist())
                            EpochPhase.extend(tempphase.cpu().detach().tolist())
                            EpochRisk.append(out.detach().cpu().numpy())

                            if torch.sum(tempphase).cpu().detach().item() < 1:
                                pass_count += 1
                            else:
                                if Argument.loss == 'CoxPH':
                                    risklist, tempsurvival, tempphase = cox_sort(out, tempsurvival, tempphase)
                                    loss = criterion(risklist, tempsurvival, tempphase)

                                if mode == 'train':
                                    loss.backward()
                                    torch.nn.utils.clip_grad_norm_(model_parameter_groups[0]['params'],
                                                                   max_norm=Argument.clip_grad_norm_value,
                                                                   error_if_nonfinite=True)
                                    torch.nn.utils.clip_grad_norm_(model_parameter_groups[1]['params'],
                                                                   max_norm=Argument.clip_grad_norm_value,
                                                                   error_if_nonfinite=True)
                                    optimizer_ft.step()
                                    scheduler.step()

                                Epochloss += loss.cpu().detach().item()
                                batchcounter += 1

                        EpochRisk = np.concatenate(EpochRisk)
                        if Argument.loss == 'CoxPH':
                            Epochacc = accuracytest(torch.tensor(EpochSurv), torch.tensor(EpochRisk),
                                                    torch.tensor(EpochPhase))
                        Epochloss = Epochloss / batchcounter

                        if mode == 'train':
                            if Epochacc > BestAccDict['train']:
                                BestAccDict['train'] = Epochacc
                        elif mode == 'val':
                            if Epochacc > BestAccDict['val']:
                                BestAccDict['val'] = Epochacc
                        print()
                        print('epoch:' + str(epoch))
                        print(" mode:" + mode)
                        print(" loss:" + str(Epochloss) + " acc:" + str(Epochacc) + " pass count:" + str(pass_count))

                        if mode == 'train':
                            if Argument.wandb:
                                wandb.log({
                                    'Training Accuracy': Epochacc,
                                    'Training loss': Epochloss
                                }, step=epoch)
                        elif mode == 'val':
                            if Argument.wandb:
                                wandb.log({
                                    'Validation Accuracy': Epochacc,
                                    'Validation loss': Epochloss
                                }, step=epoch)

                        checkpointinfo = 'epoch-{},acc-{:4f},loss-{:4f}.pt'
                        if mode == 'val':
                            if epoch == 0:
                                if Argument.save:
                                    torch.save(model.state_dict(), os.path.join(fold_checkpoint_dir,
                                                                                checkpointinfo.format(epoch, Epochacc,
                                                                                                      Epochloss)))
                            else:
                                if Epochacc > bestacc:
                                    bestepoch = epoch
                                    if epoch > 10:
                                        if Argument.save:
                                            torch.save(model.state_dict(), os.path.join(fold_checkpoint_dir,
                                                                                        checkpointinfo.format(epoch,
                                                                                                              Epochacc,
                                                                                                              Epochloss)))
                                if Epochacc > bestacc:
                                    bestacc = Epochacc
                                if Epochloss < bestloss:
                                    bestloss = Epochloss
                pbar.update()
        FFCV_accuracy.append(bestacc)
        FFCV_best_epoch.append(bestepoch)

    df = pd.DataFrame(list(zip(list(range(Argument.FF_number)), FFCV_accuracy, FFCV_best_epoch)),
                      columns=['Fold', 'C-index', 'Best_epoch'])
    if Argument.save:
        df.to_csv(os.path.join(checkpoint_dir, 'final_result.csv'), index=False)
        bestFi = np.argmax(FFCV_accuracy)
        best_checkpoint_dir = os.path.join(checkpoint_dir, str(bestFi))
        Argument.checkpoint_dir = best_checkpoint_dir

    return df