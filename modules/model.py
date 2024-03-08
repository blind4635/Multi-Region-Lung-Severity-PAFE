import torch
import torch.nn as nn
import torchvision

from .vit import PatchEmbedding, TransformerEncoderBlock
from .backbones import model_dict
import torch.nn.functional as F



class CNN(nn.Module):
    def __init__(self,backbone,num_classes=4,num_regions=6):
        super(CNN, self).__init__()
        self.backbone, self.num_features = model_dict[backbone]
        self.num_classes = num_classes
        self.num_regions = num_regions
    
        # remove GAP, FC layer
        if backbone == 'densenet121':
            self.backbone = nn.Sequential(self.backbone())
            self.img_size = 16
        else:
            self.backbone = nn.Sequential(*list(self.backbone().children())[:-2])
            self.img_size = 16
        
        self.score_GAP = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.num_features, self.num_classes)
        
        
    def pool_rois(self, x, crop_size=None):
        """
        From BSnet
        Pool the ROIs from the feature map.
        ex)
        0 51 0 64  (1)
        0 51 64 128 (2)
        38 89 0 64 (3)
        38 89 64 128 (4)
        76 128 0 64 (5)
        76 128 64 128 (6)
        """
        #print(x.shape   )
        if crop_size is None:
            crop_size = x.shape[2:4]

        if self.num_regions == 6:
            boxes = [torch.tensor([0, 0, 0.4, 0.5]), # A (1)
                    torch.tensor([0.3, 0, 0.7, 0.5]), # B (2)
                    torch.tensor([0.6, 0, 1, 0.5]), # C (3)
                    torch.tensor([0, 0.5, 0.4, 1]), # D (4)
                    torch.tensor([0.3, 0.5, 0.7, 1]), # E (5)
                    torch.tensor([0.6, 0.5, 1, 1]) # F (6)
                    ]
        elif self.num_regions == 4:
            boxes = [   torch.tensor([0, 0, 0.5, 0.5]), # A (1)
                        torch.tensor([0.5, 0, 1, 0.5]), # B (2)
                        torch.tensor([0, 0.5, 0.5, 1]), # C (3)
                        torch.tensor([0.5, 0.5, 1, 1]) # D (4)
                    ]

        out = []
        for b in boxes:
            # print(int(b[0]*x.shape[2]),int(b[2]*x.shape[2]), 
            #       int(b[1]*x.shape[3]),int(b[3]*x.shape[3]))
            car = F.interpolate(
                x[:, :,
                  int(b[0]*x.shape[2]):int(b[2]*x.shape[2]), # 세로
                  int(b[1]*x.shape[3]):int(b[3]*x.shape[3])  # 가로
                  ],
                size=crop_size,
                mode='bilinear',
                align_corners=False
            )
            
            out.append(car)
        return torch.stack(out,dim=1)
    
    def forward(self, x, embedding=False):        
        layer5_out = self.backbone(x)
        layer5_out = self.pool_rois(layer5_out) # torch.Size([1, 6, 512, 16, 16])
        
        
        b = layer5_out
        #print(b.shape)
        retina_net_class = []
        for i in range(self.num_regions):
            pred_i = b[:,i,:,:,:]
            #print('selection : ',pred_i.shape)
            pred_i = self.score_GAP(pred_i)
            pred_i = pred_i.view(-1,self.num_features)
            #print("gap : ",pred_i.shape)
            
            if embedding == False:
                pred_i = self.fc(pred_i)
                #print(pred_i.shape)
                pred_i = F.softmax(pred_i, dim=1)
            retina_net_class.append(pred_i)
        retina_net_class = torch.stack(retina_net_class,dim=1)
        return retina_net_class




class Hybrid(nn.Module):
    def __init__(self,backbone,num_classes=4,num_regions=6):
        super(Hybrid, self).__init__()
        self.backbone,num_features = model_dict[backbone]
        self.num_classes = num_classes
        self.num_regions = num_regions


        if backbone == 'densenet121':
            self.backbone = nn.Sequential(self.backbone())
            self.img_size = 16
        else:
            self.backbone = nn.Sequential(*list(self.backbone().children())[:-2])
            self.img_size = 16
        
        self.score_GAP = nn.AdaptiveAvgPool2d((1, 1))
        
        
        ###
        # https://github.com/FrancescoSaverioZuppichini/ViT
        # Transformer hybrid network
        
        self.patch_emb = PatchEmbedding(in_channels=num_features, patch_size=1, emb_size=768, img_size=self.img_size)
        
        # 2개의 transformer block
        self.transformer  = TransformerEncoderBlock()     
        self.fc = nn.Linear(768, self.num_classes)
    
        
    def pool_rois(self, x, crop_size=None):
        """
        Pool the ROIs from the feature map.
        ex)
        0 51 0 64  (1)
        0 51 64 128 (2)
        38 89 0 64 (3)
        38 89 64 128 (4)
        76 128 0 64 (5)
        76 128 64 128 (6)
        """
        #print(x.shape   )
        if crop_size is None:
            crop_size = x.shape[2:4]

        if self.num_regions == 6:
            boxes = [torch.tensor([0, 0, 0.4, 0.5]), # A (1)
                    torch.tensor([0.3, 0, 0.7, 0.5]), # B (2)
                    torch.tensor([0.6, 0, 1, 0.5]), # C (3)
                    torch.tensor([0, 0.5, 0.4, 1]), # D (4)
                    torch.tensor([0.3, 0.5, 0.7, 1]), # E (5)
                    torch.tensor([0.6, 0.5, 1, 1]) # F (6)
                    ]
        elif self.num_regions == 4:
            boxes = [   torch.tensor([0, 0, 0.5, 0.5]), # A (1)
                        torch.tensor([0.5, 0, 1, 0.5]), # B (2)
                        torch.tensor([0, 0.5, 0.5, 1]), # C (3)
                        torch.tensor([0.5, 0.5, 1, 1]) # D (4)
                    ]
        
        out = []
        for b in boxes:
            # print(int(b[0]*x.shape[2]),int(b[2]*x.shape[2]), 
            #       int(b[1]*x.shape[3]),int(b[3]*x.shape[3]))
            car = F.interpolate(
                x[:, :,
                  int(b[0]*x.shape[2]):int(b[2]*x.shape[2]), # 세로
                  int(b[1]*x.shape[3]):int(b[3]*x.shape[3])  # 가로
                  ],
                size=crop_size,
                mode='bilinear',
                align_corners=False
            )
            
            out.append(car)
        return torch.stack(out,dim=1)

    
    def forward(self, x, embedding=False):
        layer5_out = self.backbone(x)
        
        
        #print(b.shape)
        
        ###
        # print('before patch : ',layer5_out.shape)
        layer5_out = self.patch_emb(layer5_out)
        
        ###
        layer5_out = self.transformer(layer5_out)
        layer5_out = layer5_out.reshape(-1,16,16,768)
        layer5_out = layer5_out.permute(0,3,1,2)
        # print('after transformer : ',layer5_out.shape)
        
        
        b = self.pool_rois(layer5_out) # torch.Size([1, 6, 512, 16, 16])
        
        retina_net_class = []
        for i in range(self.num_regions):
            pred_i = b[:,i,:,:,:]
            #print('selection : ',pred_i.shape)
            pred_i = self.score_GAP(pred_i)
            pred_i = pred_i.view(-1,768)
            #print("gap : ",pred_i.shape)
            
            if embedding == False:                
                #print('transformer : ',pred_i.shape)
                pred_i = self.fc(pred_i)
                #print(pred_i.shape)
                pred_i = F.softmax(pred_i, dim=1)
            retina_net_class.append(pred_i)
        retina_net_class = torch.stack(retina_net_class,dim=1)
        return retina_net_class
    
    
    