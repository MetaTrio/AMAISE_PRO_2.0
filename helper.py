import numpy as np
import torch
import random
import torch.nn as nn
from Bio import SeqIO
from Bio.SeqIO.QualityIO import FastqGeneralIterator

torch.backends.cudnn.enabled = False
import joblib
from pcmer import features

# ensure reproducibility
seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# the thresholds used to convert AMAISE's output probabilities into classification labels
threshs = {
    25: 0.31313131313131315,
    50: 0.4141414141414142,
    100: 0.5454545454545455,
    150: 0.6262626262626263,
    200: 0.7070707070707072,
    250: 0.6363636363636365,
    300: 0.6666666666666667,
    500: 0.6464646464646465,
    1000: 0.4747474747474748,
    5000: 0.48484848484848486,
    10000: 0.4646464646464647,
}

def encodeLabel(num):
    encoded_l = [0] * 6
    encoded_l[num] = 1
    return torch.FloatTensor(encoded_l)


"""
Inputs:
none

Outputs:
TCN: AMAISE's architecture, which consists of 4 convolutional layers, a global average pooling layer, and 1 fully connected layer. Each convolutional layer in AMAISE contains 128 filters of length 15. We applied a rectified-linear unit activation function and an average pooling layer of length 5 after each convolutional layer.

The class TCN contains AMAISE's architecture
"""


class TCN(nn.Module):
    def __init__(self):
        num_input_channels = 1
        num_output_channels = 128
        filter_size = 15
        num_classes = 6
        pool_amt = 4

        super().__init__()
        self.c_in1 = nn.Conv1d(
            num_input_channels,
            num_output_channels,
            kernel_size=filter_size,
            padding=(filter_size - 1) // 2,
            padding_mode="zeros",
        )
        self.c_in2 = nn.Conv1d(
            num_output_channels,
            num_output_channels,
            kernel_size=filter_size,
            padding=(filter_size - 1) // 2,
            padding_mode="zeros",
        )
        self.c_in3 = nn.Conv1d(
            num_output_channels,
            num_output_channels,
            kernel_size=filter_size,
            padding=(filter_size - 1) // 2,
            padding_mode="zeros",
        )
        self.c_in4 = nn.Conv1d(
            num_output_channels,
            num_output_channels,
            kernel_size=filter_size,
            padding=(filter_size - 1) // 2,
            padding_mode="zeros",
        )
        self.fc = nn.Linear(num_output_channels, num_classes)
        self.pool = nn.AvgPool1d(pool_amt)
        self.pad = nn.ConstantPad1d((pool_amt - 1) // 2 + 1, 0)

        self.filter_size = filter_size
        self.pool_amt = pool_amt

    def forward(self, x):
        x = x.transpose(2, 1)

        old_shape = x.shape[2]
        if x.shape[2] < self.pool_amt:
            x = self.pad(x)
        new_shape = x.shape[2]

        # x1 = x
        output = self.c_in1(x)
        output = torch.relu(output)
        # output = output + x1
        output = self.pool(output) * (new_shape / old_shape)

        old_shape = output.shape[2]
        if output.shape[2] < self.pool_amt:
            output = self.pad(output)
        new_shape = output.shape[2]
        # x2 = self.pool(output)*(new_shape/old_shape)
        output = self.c_in2(output)
        output = torch.relu(output)
        # output = output + x2
        output = self.pool(output) * (new_shape / old_shape)

        old_shape = output.shape[2]
        if output.shape[2] < self.pool_amt:
            output = self.pad(output)
        new_shape = output.shape[2]
        # x3 = output
        output = self.c_in3(output)
        output = torch.relu(output)
        # output = output + x2
        # ************
        # output = self.pool(output) * (new_shape / old_shape)

        # old_shape = output.shape[2]
        # if output.shape[2] < self.pool_amt:
        #     output = self.pad(output)
        # new_shape = output.shape[2]
        # **********
        # x4 = output
        output = self.c_in4(output)
        output = torch.relu(output)

        # output = output + x4
        last_layer = nn.AvgPool1d(output.size(2))
        output = last_layer(output).reshape(output.size(0), output.size(1)) * (
            new_shape / old_shape
        )
        output = self.fc(output)
        return output
