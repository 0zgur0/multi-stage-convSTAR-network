import torch
import torch.nn
import torch.nn.functional as F
from models.TransformerModel import TransformerModel    


class model_GT(torch.nn.Module):
    def __init__(self, nclasses=36, num_classes_l1=6, num_classes_l2=20, s1_2_s3=None, s2_2_s3=None, transformer_type=1):
        super(model_GT, self).__init__()
        self.nclasses = nclasses
        self.hidden_dim = 2 * nclasses
        
        self.s1_2_s3 = s1_2_s3
        self.s2_2_s3 = s2_2_s3

        self.transformer = TransformerModel(input_dim=nclasses, sequencelength=3,
                               d_model=64, d_inner=256,
                               n_layers=3, n_head=8,
                               dropout=0.4, num_classes=nclasses)
   
        
    def forward(self, x, hidden=None):
        x1_, x2_, x3 = x
                
        x1_ = x1_.permute(0,2,3,1)
        x2_ = x2_.permute(0,2,3,1)
        x3 = x3.permute(0,2,3,1)
        
        x1_ = x1_.contiguous().view(-1,x1_.shape[3])
        x2_ = x2_.contiguous().view(-1,x2_.shape[3])
        x3 = x3.contiguous().view(-1,x3.shape[3])

        x1 = torch.zeros_like(x3)
        x2 = torch.zeros_like(x3)
    
        b, c = x3.shape

        if self.s1_2_s3[0] == None:        
            x1[:, :x1_.shape[1]] = x1_[:,:]
            x2[:, :x2_.shape[1]] = x2_[:,:]
        else:
            for i in range(self.s1_2_s3.shape[0]):
                x1[:,i] = x1_[:,int(self.s1_2_s3[i])]
                x2[:,i] = x2_[:,int(self.s2_2_s3[i])]
    
        
        x1 = x1.unsqueeze(1)
        x2 = x2.unsqueeze(1)
        x3 = x3.unsqueeze(1)

        x_concat = torch.cat((x1,x2,x3),dim=1)
        
        last = self.transformer(x_concat) 
                
        return last
    
    