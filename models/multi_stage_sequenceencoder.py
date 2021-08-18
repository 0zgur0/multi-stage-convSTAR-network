import torch
import torch.nn
import torch.nn.functional as F
from models.convstar import ConvSTAR, ConvSTAR_Res
from models.convgru import ConvGRU
from models.convlstm import ConvLSTM

class multistageSTARSequentialEncoder(torch.nn.Module):
    def __init__(self, height, width, input_dim=4, hidden_dim=64, nclasses=15,
                 nstage=3, nclasses_l1=3, nclasses_l2=7, kernel_size=(3,3), n_layers=6,
                 use_in_layer_norm=False, viz=False, test=False, wo_softmax=False, cell='star'):
        super(multistageSTARSequentialEncoder, self).__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.nstage = nstage
        self.viz = viz
        self.test = test
        self.wo_softmax = wo_softmax
        self.cell = cell
        #self.inconv = torch.nn.Conv3d(input_dim,hidden_dim,(1,3,3))
        
        self.use_in_layer_norm = use_in_layer_norm
        if use_in_layer_norm:
            self.in_layer_norm = torch.nn.LayerNorm(input_dim)
        

        if self.cell == 'gru':
            self.rnn = ConvGRU(input_size=input_dim,
                                hidden_sizes=hidden_dim,
                                kernel_sizes=kernel_size[0],
                                n_layers=n_layers)
        elif self.cell == 'star_res':
            self.rnn = ConvSTAR_Res(input_size=input_dim,
                                    hidden_sizes=hidden_dim,
                                    kernel_sizes=kernel_size[0],
                                    n_layers=n_layers)
        else:
            self.rnn = ConvSTAR(input_size=input_dim,
                                    hidden_sizes=hidden_dim,
                                    kernel_sizes=kernel_size[0],
                                    n_layers=n_layers)



        self.final = torch.nn.Conv2d(hidden_dim, nclasses, (3, 3), padding=1)
        self.final_local_1 = torch.nn.Conv2d(hidden_dim, nclasses_l1, (3, 3), padding=1)
        self.final_local_2 = torch.nn.Conv2d(hidden_dim, nclasses_l2, (3, 3), padding=1)


    def forward(self, x, hiddenS=None):
        
    
        if self.use_in_layer_norm:
            #(b x t x c x h x w) -> (b x t x h x w x c) -> (b x c x t x h x w)
            x = self.in_layer_norm(x.permute(0,1,3,4,2)).permute(0,4,1,2,3)
        else:
            # (b x t x c x h x w) -> (b x c x t x h x w)
            x = x.permute(0,2,1,3,4)
            
        #x = torch.nn.functional.pad(x, (1, 1, 1, 1), 'constant', 0)
        #x = self.inconv.forward(x)

        b, c, t, h, w = x.shape


        #convRNN step---------------------------------
        #hiddenS is a list (number of layer) of hidden states of size [b x c x h x w]
        if hiddenS is None:
            hiddenS = [torch.zeros((b, self.hidden_dim, h, w))] * self.n_layers
            
        if torch.cuda.is_available():
            for i in range(self.n_layers):
                hiddenS[i] = hiddenS[i].cuda()

        for iter in range(t):
            hiddenS = self.rnn.forward( x[:,:,iter,:,:], hiddenS )
                    
        

        if self.n_layers == 3:
            local_1 = hiddenS[0]
            local_2 = hiddenS[1]
        elif self.nstage==3:
            local_1 = hiddenS[1]
            local_2 = hiddenS[3]
        elif self.nstage==2: 
            local_1 = hiddenS[1]
            local_2 = hiddenS[2]
        elif self.nstage==1:
            local_1 = hiddenS[-1]
            local_2 = hiddenS[-1]            
            
        local_1 = self.final_local_1(local_1)
        local_2 = self.final_local_2(local_2)            

        last = hiddenS[-1]
        last = self.final(last)

        if self.viz:
            return  hiddenS[-1]
        elif self.test:
            return  F.softmax(last, dim=1), F.softmax(local_1, dim=1), F.softmax(local_2, dim=1) 
        elif self.wo_softmax:
            return  last, local_1, local_2 
        else:
            return F.log_softmax(last, dim=1), F.log_softmax(local_1, dim=1), F.log_softmax(local_2, dim=1)


class multistageLSTMSequentialEncoder(torch.nn.Module):
    def __init__(self, height, width, input_dim=4, hidden_dim=64, nclasses=15,
                 nstage=3, nclasses_l1=3, nclasses_l2=7, kernel_size=(3, 3), n_layers=6,
                 use_in_layer_norm=False, viz=False, test=False, wo_softmax=False):
        super(multistageLSTMSequentialEncoder, self).__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.nstage = nstage
        self.viz = viz
        self.test = test
        self.wo_softmax = wo_softmax
        # self.inconv = torch.nn.Conv3d(input_dim,hidden_dim,(1,3,3))

        self.use_in_layer_norm = use_in_layer_norm
        if use_in_layer_norm:
            self.in_layer_norm = torch.nn.LayerNorm(input_dim)

        self.rnn = ConvLSTM(input_size=[24,24],
                            input_dim=input_dim,
                            hidden_dim=hidden_dim,
                            kernel_size=kernel_size,
                            num_layers=n_layers)

        self.final = torch.nn.Conv2d(hidden_dim, nclasses, (3, 3), padding=1)
        self.final_local_1 = torch.nn.Conv2d(hidden_dim, nclasses_l1, (3, 3), padding=1)
        self.final_local_2 = torch.nn.Conv2d(hidden_dim, nclasses_l2, (3, 3), padding=1)

    def forward(self, x, hiddenS=None):

        if self.use_in_layer_norm:
            # (b x t x c x h x w) -> (b x t x h x w x c) -> (b x c x t x h x w)
            x = self.in_layer_norm(x.permute(0, 1, 3, 4, 2)).permute(0, 4, 1, 2, 3)
        # else:
        #     # (b x t x c x h x w) -> (b x c x t x h x w)
        #     x = x.permute(0, 2, 1, 3, 4)

        # x = torch.nn.functional.pad(x, (1, 1, 1, 1), 'constant', 0)
        # x = self.inconv.forward(x)
        b, t, c, h, w = x.shape

        # convRNN step---------------------------------
        # hiddenS is a list (number of layer) of hidden states of size [b x c x h x w]
        if hiddenS is None:
            hiddenS = [torch.zeros((b, self.hidden_dim, h, w))] * self.n_layers
            cellS = [torch.zeros((b, self.hidden_dim, h, w))] * self.n_layers

        if torch.cuda.is_available():
            for i in range(self.n_layers):
                hiddenS[i] = hiddenS[i].cuda()
                cellS[i] = cellS[i].cuda()

        #for iter in range(t):
        #    hiddenS, cellS = self.rnn.forward(x[:, :, iter, :, :], hiddenS, cellS)
        hiddenS, cellS = self.rnn.forward(x, hiddenS, cellS)

        if self.n_layers == 3:
            local_1 = hiddenS[0]
            local_2 = hiddenS[1]
        elif self.nstage==3:
            local_1 = hiddenS[1]
            local_2 = hiddenS[3]
        elif self.nstage==2:
            local_1 = hiddenS[1]
            local_2 = hiddenS[2]
        elif self.nstage==1:
            local_1 = hiddenS[-1]
            local_2 = hiddenS[-1]

        last = hiddenS[-1]

        local_1 = local_1[:,-1,:,:,:]
        local_2 = local_2[:,-1,:,:,:]
        last = last[:,-1,:,:,:]

        local_1 = self.final_local_1(local_1)
        local_2 = self.final_local_2(local_2)
        last = self.final(last)

        if self.viz:
            return hiddenS[-1]
        elif self.test:
            return F.softmax(last, dim=1), F.softmax(local_1, dim=1), F.softmax(local_2, dim=1)
        elif self.wo_softmax:
            return last, local_1, local_2
        else:
            return F.log_softmax(last, dim=1), F.log_softmax(local_1, dim=1), F.log_softmax(local_2, dim=1)
