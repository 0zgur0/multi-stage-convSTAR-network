import torch
import torch.nn
from models.convlstm.convlstm import ConvLSTMCell
import torch.nn.functional as F

class LSTMSequentialEncoder(torch.nn.Module):
    def __init__(self, height, width, input_dim=9, hidden_dim=64, nclasses=8, kernel_size=(3,3), bias=False):
        super(LSTMSequentialEncoder, self).__init__()

        self.inconv = torch.nn.Conv3d(input_dim,hidden_dim,(1,3,3))

        self.cell = ConvLSTMCell(input_size=(height, width),
                     input_dim=hidden_dim,
                     hidden_dim=hidden_dim,
                     kernel_size=kernel_size,
                     bias=bias)

        self.final = torch.nn.Conv2d(hidden_dim, nclasses, (3, 3))

    def forward(self, x, hidden=None, state=None):

        # (b x t x c x h x w) -> (b x c x t x h x w)
        x = x.permute(0,2,1,3,4)

        x = torch.nn.functional.pad(x, (1, 1, 1, 1), 'constant', 0)
        x = self.inconv.forward(x)

        b, c, t, h, w = x.shape

        if hidden is None:
            hidden = torch.zeros((b, c, h, w))
        if state is None:
            state = torch.zeros((b, c, h, w))

        if torch.cuda.is_available():
            hidden = hidden.cuda()
            state = state.cuda()

        for iter in range(t):

            hidden, state = self.cell.forward(x[:,:,iter,:,:], (hidden, state))

        x = torch.nn.functional.pad(state, (1, 1, 1, 1), 'constant', 0)
        x = self.final.forward(x)

        return F.log_softmax(x, dim=1)


if __name__=="__main__":


    b, t, c, h, w = 2, 10, 3, 320, 320

    model = LSTMSequentialEncoder(height=h, width=w, input_dim=c, hidden_dim=3)

    x = torch.randn((b, t, c, h, w))


    hidden, state = model.forward(x)


