import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as utils_data

class ConvBlock(nn.Module):
    """
    DeepLOB's fully convolutional block.
    """
    
    def __init__(self, input_cn, n_filters, input_kernel_size=(1,2), input_stride=(1,2), leaky_alpha=0.01):
        super(ConvBlock, self).__init__()
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(input_cn, n_filters, kernel_size=input_kernel_size, stride=input_stride),\
            nn.LeakyReLU(leaky_alpha),\
            nn.ZeroPad2d((0,0,1,2)),\
            nn.Conv2d(n_filters, n_filters, kernel_size=(4,1)),\
            nn.LeakyReLU(leaky_alpha),\
            nn.ZeroPad2d((0,0,1,2)),\
            nn.Conv2d(n_filters, n_filters, kernel_size=(4,1)),\
            nn.LeakyReLU(leaky_alpha)
        )
    
    def forward(self, x):
        return self.conv_block(x)
    
class InceptionBlock(nn.Module):
    """
    Inception v1 module.
    """
    
    def __init__(self, input_cn, n_filters, leaky_alpha=0.01):
        super(InceptionBlock, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_cn, n_filters, kernel_size=1),\
            nn.LeakyReLU(leaky_alpha),\
            nn.Conv2d(n_filters, n_filters, kernel_size=(3,1), padding=(1,0)),\
            nn.LeakyReLU(leaky_alpha)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(input_cn, n_filters, kernel_size=1),\
            nn.LeakyReLU(leaky_alpha),\
            nn.Conv2d(n_filters, n_filters, kernel_size=(5,1), padding=(2,0)),\
            nn.LeakyReLU(leaky_alpha)
        )
        
        self.conv3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3,1), stride=(1,1), padding=(1,0)),\
            nn.Conv2d(input_cn, n_filters, kernel_size=(3,1), padding=(1,0)),\
            nn.LeakyReLU(leaky_alpha)
        )
    
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)

        x  = torch.cat((x1,x2,x3), dim=1)
        return x
    
class DeepLOB(nn.Module):
    """
    PyTorch implementation of the model proposed in Zhang et al. DeepLOB: Deep Convolutional Neural Networks for Limit Order Books.
    """
    
    def __init__(self, input_cn=1, n_filters=16, inception_filters=32, lstm_units=64, alpha=0.01):
        super(DeepLOB, self).__init__()
        
        self.conv_block = nn.Sequential(
            ConvBlock(input_cn, n_filters),\
            ConvBlock(n_filters, n_filters),\
            ConvBlock(n_filters, n_filters, input_kernel_size=(1,10), input_stride=1)
        )
        
        self.inception = InceptionBlock(n_filters, inception_filters)
        
        self.lstm_input_dim = 3 * inception_filters
        self.lstm = nn.LSTM(self.lstm_input_dim, lstm_units, batch_first=True)
        
        self.out = nn.Linear(lstm_units, 3)
        self.apply(self.weights_init)

    def weights_init(self, m):
        """
        This model requires a specific weights allocation in order to function well and reproduce results close to the ones obtained in the paper.
        """
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'bias_ih' in name:
                    nn.init.zeros_(param)
                elif 'bias_hh' in name:
                    nn.init.zeros_(param)
                elif 'weight_ih' in name:
                    nn.init.xavier_uniform_(param)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param)
    
    def forward(self, x):
        # input size = (batch_size, T, features)
        x = self.conv_block(x)
        x = self.inception(x)
        x = x.squeeze(3).permute(0,2,1)
        x, _ = self.lstm(x)

        return self.out(x[:,-1,:])