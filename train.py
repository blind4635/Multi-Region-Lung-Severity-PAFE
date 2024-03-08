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
        
        #loss for classification per 6 position
        loss_classification = 0
        loss_regression = 0
        #print(pred.shape)
        for i in range(6):
            pos_probs = pred[:,i,:]
            loss_classification += classify_criterion( pos_probs, label[:,i])
            
            prediction = pos_probs.max(1,keepdim=True)[1]
            train_correct[i] += prediction.eq(label[:,i].view_as(prediction)).sum().item()
        
        loss_classification = loss_classification / 6
        
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
    

    base_path = '/Multi-Region-Lung-Severity-PAFE/'
    data_path = '/data/'
    
    model_name = 'brixia_{}_{}_ep{}'.format(args.arch, args.backbone, args.epochs)
    
    
    wandb.login(key='') # use your key
    wandb.init(project="brixia")
    wandb.config.update(args)
    experiment_name = model_name 
    wandb.run.name = experiment_name
    wandb.run.save()
    
    
    ####
    # data loading
    print('Data Loading...')
    meta_data = pd.read_csv( os.path.join(data_path, 'metadata_global_v2_all.csv') )
    
    label_data = pd.read_csv( os.path.join(data_path,'metadata_global_v2_process.csv') )
    consensus_csv = pd.read_csv( os.path.join(data_path,'metadata_consensus_v1.csv'))
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
    
    optimizer = torch.optim.SGD(model.parameters(), lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)    
    
    ### loss function ###
    classify_criterion = nn.CrossEntropyLoss()
    
    ####    
    
    
    ##### trainer
    #logger = tb_logger.Logger(logdir=os.path.join("./experiment/cxr",'tensorboard'), flush_secs=2)
    
    all_train_loss_list = []
    all_valid_loss_list = []

    all_train_acc_list = []
    all_valid_acc_list = []
    
    
    for epoch in range(EPOCHS):
        print(f"EPOCH {epoch} / {EPOCHS}")
        print(model_name)
        start_time = time.time()
        
        train_loss, train_acc = train(model, EPOCHS, epoch ,train_dataloader, optimizer, classify_criterion, DEVICE)
        valid_loss, valid_acc = evaluate(model, EPOCHS, epoch ,valid_dataloader, classify_criterion, DEVICE)
        
        all_train_loss_list.append(train_loss)
        all_valid_loss_list.append(valid_loss)
        
        all_train_acc_list.append(train_acc)
        all_valid_acc_list.append(valid_acc)
        
        print(f"Time taken : {time.time()-start_time:.2f}")
        print("=====================================")
        print("Train ACC : ",train_acc, "Valid ACC : ",valid_acc)
        print("Train Loss : ",train_loss, "Valid Loss : ",valid_loss)
        wandb.log({
            "Train Loss": train_loss,
            "Valid Loss": valid_loss,
            "Train ACC": train_acc,
            "Valid ACC": valid_acc,
        })
    
    
        
    torch.save(model.state_dict(), os.path.join(base_path,'checkpoint','{}.pth'.format(model_name)) )
    
    #####
    
    
    #### loss visualize
    
    # visualize
    plt.figure()
    plt.plot(all_train_loss_list, label='train loss')
    #legend
    plt.legend()
    plt.savefig(os.path.join(base_path,'plots',"{}_train_loss.png".format(model_name) ))
    
    # acc visualize
    plt.figure()
    plt.plot(all_train_acc_list, label='train acc')
    #legend
    plt.legend()    
    plt.savefig(os.path.join(base_path,'plots',"{}_train_acc.png".format(model_name) ))
    
    ####
    
    # visualize
    plt.figure()
    plt.plot(all_valid_loss_list, label='valid loss')
    #legend
    plt.legend()
    plt.savefig(os.path.join(base_path,'plots',"{}_valid_loss.png".format(model_name) ))
    
    # acc visualize
    plt.figure()
    plt.plot(all_valid_acc_list, label='valid acc')
    #legend
    plt.legend()    
    plt.savefig(os.path.join(base_path,'plots',"{}_valid_acc.png".format(model_name) ))
    
    ####
    
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
    
    
    pred_list = []
    label_list = []
    with torch.no_grad():
        for idx, (img, label,label_float, filename) in enumerate(consensus_test_dataloader):
            img, label,label_float = img.to(DEVICE), label.to(DEVICE),label_float.to(DEVICE)
            pred = model(img)
            for i in range(6):
                pos_probs = pred[:,i,:]
                prediction = pos_probs.max(1,keepdim=True)[1].view(-1)
                pred_list += prediction.cpu().numpy().tolist()
                label_list += label[:,i].cpu().numpy().tolist()
        
        # print mean acc
        consensus_acc = 100. * sum([1 for i in range(len(pred_list)) if pred_list[i] == label_list[i]]) / len(pred_list)
        print(f"Consensus Test ACC : {consensus_acc:.2f}")        
    
    
    
    
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
    parser.add_argument('--linux', default=False, type=bool, help='linux')
    parser.add_argument('--num_class', default=4, type=int, help='num_class')
    parser.add_argument('--num_regions', default=6, type=int, help='num_class')
    parser.add_argument('--arch', default='cnn', type=str, help='[cnn, hybrid]')
    parser.add_argument('--backbone', default='resnet18', type=str, help='[ resnet18,resnet34,resnet50]')

    args = parser.parse_args()
    
    main(args)