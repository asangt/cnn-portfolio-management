import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as utils_data

class ResConv2d(nn.Module):
    """
    Mainly for convenience - combination of convolutional layer with zero padding and a leaky ReLU activation.
    """

    def __init__(self, in_channels, out_channels, kernel_size, leaky_alpha=0.01):
        super(ResConv2d, self).__init__()
        self.padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        self.layer_block  = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=self.padding),
            nn.LeakyReLU(leaky_alpha)
        )

    def forward(self, x):
        return self.layer_block(x)

class ResBlock(nn.Module):
    """
    The residual block used in the DeepResLOB model, architecture is as per our report.
    """

    def __init__(self, n_filters, num_layers=3, leaky_alpha=0.01, kernel_sizes=None):
        super(ResBlock, self).__init__()

        if kernel_sizes is None:
            self.kernel_sizes = [(3,1) for i in range(num_layers)]
        else:
            self.kernel_sizes = kernel_sizes
        
        layers = [ResConv2d(n_filters, n_filters, self.kernel_sizes[i], leaky_alpha) for i in range(num_layers)]
        self.res_block = nn.Sequential(
            *layers
        )
    
    def forward(self, x):
        residual = self.res_block(x)

        return residual + x

class InceptionBlock_v2(nn.Module):
    """
    Inception v2 module.
    """
    
    def __init__(self, input_cn, n_filters, leaky_alpha=0.01):
        super(InceptionBlock_v2, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_cn, n_filters, kernel_size=1),\
            nn.LeakyReLU(leaky_alpha),\
            nn.Conv2d(n_filters, n_filters, kernel_size=(3,1), padding=(1,0)),\
            nn.LeakyReLU(leaky_alpha)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(input_cn, n_filters, kernel_size=1),\
            nn.LeakyReLU(leaky_alpha),\
            nn.Conv2d(n_filters, n_filters, kernel_size=(3,1), padding=(1,0)),\
            nn.LeakyReLU(leaky_alpha),\
            nn.Conv2d(n_filters, n_filters, kernel_size=(3,1), padding=(1,0)),\
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

        x  = torch.cat((x1, x2, x3), dim=1)
        return x

class DeepResLOB(nn.Module):
    """
    Our modified and redesigned model, which uses residual connections and gated recurrent units. For architecture refer
    to the report.
    """

    def __init__(self, in_channels=1, gru_units=64, res_filters=16, inception_filters=32, res_layers=3, res_blocks=2, leaky_alpha=0.01):
        super(DeepResLOB, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, res_filters, kernel_size=(1,2), stride=(1,2)),
            nn.LeakyReLU(leaky_alpha)
        )

        res_layers1 = [ResBlock(res_filters, res_layers, leaky_alpha) for i in range(res_blocks)]

        self.res_block1 = nn.Sequential(
            *res_layers1
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(res_filters, res_filters, kernel_size=(1,2), stride=(1,2)),
            nn.LeakyReLU(leaky_alpha)
        )

        res_layers2 = [ResBlock(res_filters, res_layers, leaky_alpha) for i in range(res_blocks)]

        self.res_block2 = nn.Sequential(
            *res_layers2
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(res_filters, res_filters, kernel_size=(1,10)),
            nn.LeakyReLU(leaky_alpha)
        )

        res_layers3 = [ResBlock(res_filters, res_layers, leaky_alpha) for i in range(res_blocks)]

        self.res_block3 = nn.Sequential(
            *res_layers3
        )

        gru_input_dim = 3 * inception_filters
        self.inception = InceptionBlock_v2(res_filters, inception_filters, leaky_alpha)

        self.gru       = nn.GRU(gru_input_dim, gru_units, batch_first=True)

        self.fc_out    = nn.Linear(gru_units, 3)

        self.apply(self.weights_init)

    def weights_init(self, m):
        """
        For our model, this specific weight allocation is not necessary. However, we still perform it in order to directly compare the effectiveness
        of the architecture itself.
        """
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.GRU):
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
        x = self.res_block1(self.conv1(x))
        x = self.res_block2(self.conv2(x))
        x = self.res_block3(self.conv3(x))
        x = self.inception(x)

        x = x.squeeze(3).permute(0,2,1)
        x, _ = self.gru(x)

        return self.fc_out(x[:,-1,:])