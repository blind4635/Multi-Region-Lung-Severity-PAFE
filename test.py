##
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import cv2
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
import argparse

from skimage import exposure

import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


### dataset ###
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
from sklearn.model_selection import train_test_split # train , test 분리에 사용.
from sklearn.model_selection import KFold
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader


from PIL import Image
import cv2


##
import torch
import torch.nn.functional as F

import wandb
import time

### modules
from modules.model import CNN, Hybrid
from modules.dataset import BRIXIA
###



# dataset

def train(model, EPOCHS, epoch ,train_dataloader, optimizer, classify_criterion, DEVICE):
    model.train()
    train_correct = [0,0,0,0,0,0]
    
    
    total_train_loss = 0
    
    for idx, (img, label,label_float, filename) in enumerate(train_dataloader):
        img, label,label_float = img.to(DEVICE), label.to(DEVICE), label_float.to(DEVICE)
        optimizer.zero_grad()
        pred = model(img)
        # 출력값 : [batch_size, 6, 4] . 6개의 value를 가진 4개의 class
        
        #loss for classification per 6 position
        loss_classification = 0
        loss_regression = 0
        #print(pred.shape)
        for i in range(6):
            pos_probs = pred[:,i,:]
            #print(pos_probs[0].argmax(), label[0,i])
            loss_classification += classify_criterion( pos_probs, label[:,i])
            
            prediction = pos_probs.max(1,keepdim=True)[1]
            train_correct[i] += prediction.eq(label[:,i].view_as(prediction)).sum().item()
        
        loss_classification = loss_classification / 6
        
        #loss for regression per 6 position
        # for i in range(6):
        #     pos_max = pred[:,i,:].argmax(dim=1)
        #     loss_regression += regress_criterion(pos_max, label_float[:,i])
        
        # loss_regression = loss_regression / 6
        
        #loss = alpha * loss_classification + (1-alpha) * loss_regression
        loss = loss_classification
        
        loss.backward()
        optimizer.step()
        
        
        total_train_loss += loss.item()
    
    total_train_loss = total_train_loss / len(train_dataloader)
    
    
    train_acc_list = [0,0,0,0,0,0]
    for i in range(6):
        train_acc_list[i] = 100. * train_correct[i] / len(train_dataloader.dataset)
        print(f"EPOCH {epoch} / {EPOCHS}, Position {i} Train ACC : {train_acc_list[i]:.2f}")
    #mean train acc
    print(f"EPOCH {epoch} / {EPOCHS}, Mean Train ACC : {np.mean(train_acc_list):.2f}")
    print(f"EPOCH {epoch} / {EPOCHS}, Loss : {total_train_loss:.2f}")
    
    return total_train_loss, np.mean(train_acc_list)

def evaluate(model, EPOCHS, epoch ,valid_dataloader, classify_criterion, DEVICE):
    model.eval()
    valid_correct = [0,0,0,0,0,0]
    total_valid_loss = 0
    
    
    # validation
    with torch.no_grad():
        for idx, (img, label,label_float, filename) in enumerate(valid_dataloader):
            img, label,label_float = img.to(DEVICE), label.to(DEVICE),label_float.to(DEVICE)
            pred = model(img)
            loss_classification = 0
            loss_regression = 0
            
            for i in range(6):
                pos_probs = pred[:,i,:]
                # regression은 클래스의 가중합
                #pos_regress = pos_probs * torch.tensor([0,1,2,3],dtype=torch.float32).to(DEVICE)
                loss_classification += classify_criterion( pos_probs, label[:,i])
                
                
                
                prediction = pos_probs.max(1,keepdim=True)[1]
                valid_correct[i] += prediction.eq(label[:,i].view_as(prediction)).sum().item()
                
            loss_classification = loss_classification / 6
            
            #loss for regression per 6 position
            # for i in range(6):
            #     pos_max = pred[:,i,:].argmax(dim=1)
            #     loss_regression += regress_criterion(pos_max, label_float[:,i])
            
            # loss_regression = loss_regression / 6
            
            #loss = alpha * loss_classification + (1-alpha) * loss_regression
            loss = loss_classification
            
            total_valid_loss += loss.item()
        total_valid_loss = total_valid_loss / len(valid_dataloader)
        
        
        valid_acc_list = [0,0,0,0,0,0]
        for i in range(6):
            valid_acc_list[i] = 100. * valid_correct[i] / len(valid_dataloader.dataset)
            print(f"EPOCH {epoch} / {EPOCHS}, Position {i} Valid ACC : {valid_acc_list[i]:.2f}")
        #mean valid acc
        print(f"EPOCH {epoch} / {EPOCHS}, Mean Valid ACC : {np.mean(valid_acc_list):.2f}")
        print(f"EPOCH {epoch} / {EPOCHS}, Loss : {total_valid_loss:.2f}")
        
        return total_valid_loss, np.mean(valid_acc_list) 




def main(args):
    if torch.cuda.is_available():
        DEVICE = torch.device('cuda')
    else:
        DEVICE = torch.device('cpu')
    print('Using Pytorch version : ',torch.__version__,' Device : ',DEVICE)
    

    base_path = '/multi-region-severity-hybridNet/'
    data_path = '/data/'
    
    model_name = 'brixia_{}_{}_ep{}'.format(args.arch, args.backbone, args.epochs)
    
    
    wandb.login(key='') # your key.
    wandb.init(project="brixia")
    wandb.config.update(args)
    experiment_name = model_name # check point의 이름이 될 것.
    wandb.run.name = experiment_name
    wandb.run.save()
    
    
    ####
    # data loading
    print('Data Loading...')
    meta_data = pd.read_csv( os.path.join(data_path, 'metadata_global_v2_all.csv') )
    
    label_data = pd.read_csv( os.path.join(data_path,'metadata_global_v2_process.csv') )
    consensus_csv = pd.read_csv( os.path.join(data_path,'metadata_consensus_v1.csv'))
    # consensus_csv Filename과 겹치는 데이터 제거
    label_data = label_data[~label_data['Filename'].isin(consensus_csv['Filename'])]
    subject_array=label_data['Subject'].unique()
    
    train_subject_data, else_subject_data = train_test_split(subject_array, test_size=0.3, random_state=42)
    valid_subject_data, test_subject_data = train_test_split(else_subject_data, test_size=0.3, random_state=42)

    train_data = label_data[label_data['Subject'].isin(train_subject_data)]['Filename'].to_numpy().tolist()
    valid_data = label_data[label_data['Subject'].isin(valid_subject_data)]['Filename'].to_numpy().tolist()
    test_data = label_data[label_data['Subject'].isin(test_subject_data)]['Filename'].to_numpy().tolist()
    
    
    consensus_test_data = consensus_csv['Filename'].to_numpy().tolist()
    

    train_y_data = label_data[label_data['Filename'].isin(train_data)][['brixia1','brixia2',
                                                                        'brixia3','brixia4',
                                                                        'brixia5','brixia6']].to_numpy().tolist()
    valid_y_data = label_data[label_data['Filename'].isin(valid_data)][['brixia1','brixia2',
                                                                        'brixia3','brixia4',
                                                                        'brixia5','brixia6']].to_numpy().tolist()
    test_y_data = label_data[label_data['Filename'].isin(test_data)][['brixia1','brixia2',
                                                                        'brixia3','brixia4',
                                                                        'brixia5','brixia6']].to_numpy().tolist()    
    
    consensus_test_y_data = consensus_csv[['ModeA','ModeB','ModeC','ModeD','ModeE','ModeF']].to_numpy().tolist()


    
    
    train_dataset = BRIXIA(filename_list=train_data,
                                    label_list=train_y_data,
                                    label_df=label_data,
                                    prefix=os.path.join(data_path,"registration/images/"),
                                    transform=torch.nn.Sequential(
                                                            transforms.Resize(size=[512, 512], antialias=True),
                                                        ),
                                    augmentation=None,
                                    train=True)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, 
                                                batch_size=args.batch_size, 
                                                shuffle=True, 
                                                num_workers=args.num_workers
                                                )
    
    
    valid_dataset = BRIXIA(filename_list=valid_data,
                                    label_list=valid_y_data,
                                    label_df=label_data,
                                    prefix=os.path.join(data_path,"registration/images/"),
                                    transform=torch.nn.Sequential(
                                                            transforms.Resize(size=[512, 512], antialias=True),
                                                        ),
                                    augmentation=None,
                                    train=False)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, 
                                                batch_size=args.batch_size, 
                                                shuffle=True, 
                                                num_workers=args.num_workers
                                                )    
       
    test_dataset = BRIXIA(filename_list=test_data,
                                    label_list=test_y_data,
                                    label_df=label_data,
                                    prefix=os.path.join(data_path,"registration/images/"),
                                    transform=torch.nn.Sequential(
                                                            transforms.Resize(size=[512, 512], antialias=True),
                                                        ),
                                    augmentation=None,
                                    train=False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, 
                                                batch_size=args.batch_size, 
                                                shuffle=True, 
                                                num_workers=args.num_workers
                                                )        
    
    
    
    
    consensus_test_dataset = BRIXIA(filename_list=consensus_test_data,
                                    label_list=consensus_test_y_data,
                                    label_df=consensus_csv,
                                    prefix=os.path.join(data_path,"registration/images/"),
                                    transform=torch.nn.Sequential(
                                                            transforms.Resize(size=[512, 512], antialias=True),
                                                        ),
                                    augmentation=None,
                                    train=False)
    consensus_test_dataloader = torch.utils.data.DataLoader(consensus_test_dataset,
                                                batch_size=args.batch_size, 
                                                shuffle=False, 
                                                num_workers=args.num_workers
                                                )
    
    
    ##### model contruct
    
    EPOCHS = args.epochs
    lr = args.lr    
        
    if args.arch == 'cnn':
        model = CNN(backbone=args.backbone, num_classes = args.num_class, num_regions=args.num_regions).to(DEVICE)
    elif args.arch == 'hybrid':
        model = Hybrid(backbone=args.backbone, num_classes = args.num_class, num_regions=args.num_regions).to(DEVICE)

    print("load model checkpoint: ", args.checkpoint)
    model.load_state_dict( torch.load( os.path.join(base_path,'checkpoint','{}.pth'.format(args.checkpoint)) ) )
    
    # test
    # plot confusion matrix


    # test confusion matrix

    model.eval()
    # validation

    pred_list = []
    label_list = []
    with torch.no_grad():
        for idx, (img, label,label_float, filename) in enumerate(test_dataloader):
            img, label,label_float = img.to(DEVICE), label.to(DEVICE),label_float.to(DEVICE)
            pred = model(img)
            for i in range(6):
                pos_probs = pred[:,i,:]
                prediction = pos_probs.max(1,keepdim=True)[1].view(-1)
                pred_list += prediction.cpu().numpy().tolist()
                label_list += label[:,i].cpu().numpy().tolist()
        
        # print mean acc
        test_acc = 100. * sum([1 for i in range(len(pred_list)) if pred_list[i] == label_list[i]]) / len(pred_list)
        print(f"Test ACC : {test_acc:.2f}")
        abs_list = np.abs(np.array(pred_list) - np.array(label_list))
        print(f"Test MAE : {np.mean(abs_list):.2f}")
    
    
    pred_list = []
    label_list = []
    file_list = []
    with torch.no_grad():
        for idx, (img, label,label_float, filename) in enumerate(consensus_test_dataloader):
            img, label,label_float = img.to(DEVICE), label.to(DEVICE),label_float.to(DEVICE)
            pred = model(img)
            for i in range(6):
                pos_probs = pred[:,i,:]
                prediction = pos_probs.max(1,keepdim=True)[1].view(-1)
                pred_list += prediction.cpu().numpy().tolist()
                label_list += label[:,i].cpu().numpy().tolist()
                file_list += filename
        
        # print mean acc
        consensus_acc = 100. * sum([1 for i in range(len(pred_list)) if pred_list[i] == label_list[i]]) / len(pred_list)
        print(f"Consensus Test ACC : {consensus_acc:.2f}")
        abs_list = np.abs(np.array(pred_list) - np.array(label_list))
        print(f"Consensus Test MAE : {np.mean(abs_list):.2f}")
    
    # pandas. [filename, pred, label, abs(pred-label), position]
    report_df = pd.DataFrame()
    report_df['filename'] = file_list 
    report_df['pred'] = pred_list
    report_df['label'] = label_list
    report_df['abs'] = abs_list
    report_df['position'] = [1,2,3,4,5,6] * len(consensus_test_data)
    report_df.to_csv(os.path.join(base_path,'report','{}.csv'.format(model_name)),index=False)
    
    # get tsne embedding
    if args.embedding:
        print("Get Embedding on Consensus")
        model.eval()
        # validation
        embedding_list = []
        label_list = []
        position_list = []
        with torch.no_grad():
            for idx, (img, label,label_float, filename) in enumerate(consensus_test_dataloader):
                img, label,label_float = img.to(DEVICE), label.to(DEVICE),label_float.to(DEVICE)
                pred = model(img, embedding=True)
                for i in range(6):
                    pos_probs = pred[:,i,:]
                    embedding_list += pos_probs.cpu().numpy().tolist()
                    label_list += label[:,i].cpu().numpy().tolist()
                    position_list += [i+1] * len(label)
        
        embedding_list = np.array(embedding_list)
        print(embedding_list.shape)
        # t-sne
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, random_state=0)
        X_2d = tsne.fit_transform(embedding_list)
        
        # save embedding
        embedding_df = pd.DataFrame()
        embedding_df['filename'] = file_list
        embedding_df['embedding_x'] = X_2d[:,0].tolist()
        embedding_df['embedding_y'] = X_2d[:,1].tolist()
        embedding_df['label'] = label_list
        embedding_df['position'] = position_list
        
        embedding_df.to_csv(os.path.join(base_path,'report','{}_embedding.csv'.format(model_name)),index=False)
        
        
    
    
    
    
    
    wandb.log({
        "Test ACC": test_acc,
        "Consensus Test ACC": consensus_acc,
    })
    
    
    return



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--epochs', default=100, type=int, help='number of total epochs to run')
    parser.add_argument('--lr', default=0.0001, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight_decay', default=1e-5, type=float, help='weight decay')
    parser.add_argument('--num_workers', default=1, type=int, help='num workers')
    parser.add_argument('--num_class', default=4, type=int, help='num_class')
    parser.add_argument('--num_regions', default=6, type=int, help='num_class')
    parser.add_argument('--arch', default='cnn', type=str, help='[cnn, hybrid]')
    parser.add_argument('--backbone', default='resnet18', type=str, help='[ resnet18,resnet34,resnet50, mobilenet_v3_small, densenet121 ]')
    parser.add_argument('--checkpoint', default='', type=str, help='filename')
    parser.add_argument('--embedding', default=False, type=bool, help='get embedding')

    args = parser.parse_args()
    
    main(args)