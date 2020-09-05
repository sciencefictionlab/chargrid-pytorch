from __future__ import print_function, division
import torch.nn as nn
import torch.nn.functional as F
import torch


def add_block_a(inputs, outputs, s=1):
    return nn.Sequential(
        nn.Conv2d(in_channels=inputs, out_channels=outputs, stride=s, kernel_size=3, padding=1),
        nn.LeakyReLU(0.2),
        nn.BatchNorm2d(outputs),

        nn.Conv2d(in_channels=outputs, out_channels=outputs, kernel_size=3, padding=1),
        nn.LeakyReLU(0.2),
        nn.BatchNorm2d(outputs),

        nn.Conv2d(in_channels=outputs, out_channels=outputs, kernel_size=3, padding=1),
        nn.LeakyReLU(0.2),
        nn.BatchNorm2d(outputs),

        nn.Dropout(p=0.1)
    )


def add_block_a_dash(inputs, outputs):
    return nn.Sequential(
        nn.Conv2d(in_channels=inputs, out_channels=outputs, stride=2, kernel_size=3, padding=1),
        nn.LeakyReLU(0.2),
        nn.BatchNorm2d(outputs),

        nn.Conv2d(in_channels=outputs, out_channels=outputs, kernel_size=3, padding=1, dilation=1),
        nn.LeakyReLU(0.2),
        nn.BatchNorm2d(outputs),

        nn.Conv2d(in_channels=outputs, out_channels=outputs, kernel_size=3, padding=1, dilation=1),
        nn.LeakyReLU(0.2),
        nn.BatchNorm2d(outputs),

        nn.Dropout(p=0.1)
    )


def add_block_a_double_dash(inputs, outputs, d):
    return nn.Sequential(
        nn.Conv2d(in_channels=inputs, out_channels=outputs, kernel_size=3, padding=1, dilation=d),
        nn.LeakyReLU(0.2),
        nn.BatchNorm2d(outputs),

        nn.Conv2d(in_channels=outputs, out_channels=outputs, kernel_size=3, padding=1, dilation=d),
        nn.LeakyReLU(0.2),
        nn.BatchNorm2d(outputs),

        nn.Conv2d(in_channels=outputs, out_channels=outputs, kernel_size=3, padding=1, dilation=d),
        nn.LeakyReLU(0.2),
        nn.BatchNorm2d(outputs),

        nn.Dropout(p=0.1)
    )


def add_block_b(inputs, outputs):
    return nn.Sequential(
        nn.Conv2d(in_channels=inputs, out_channels=outputs, stride=2, kernel_size=1),
        nn.LeakyReLU(0.2),
        nn.BatchNorm2d(outputs),

        nn.ConvTranspose2d(in_channels=outputs, out_channels=outputs, kernel_size=3, padding=1),
        nn.LeakyReLU(0.2),
        nn.BatchNorm2d(outputs),

        nn.Conv2d(in_channels=outputs, out_channels=outputs, kernel_size=3, padding=1),
        nn.LeakyReLU(0.2),
        nn.BatchNorm2d(outputs),

        nn.Conv2d(in_channels=outputs, out_channels=outputs, kernel_size=3, padding=1),
        nn.LeakyReLU(0.2),
        nn.BatchNorm2d(outputs),

        nn.Dropout(p=0.1)
    )


def add_block_c(inputs, outputs):
    return nn.Sequential(
        nn.Conv2d(in_channels=inputs, out_channels=outputs, stride=2, kernel_size=1),
        nn.LeakyReLU(0.2),
        nn.BatchNorm2d(outputs),

        nn.ConvTranspose2d(in_channels=outputs, out_channels=outputs, kernel_size=3, padding=1),
        nn.LeakyReLU(0.2),
        nn.BatchNorm2d(outputs),
    )


def add_block_d_or_e(inputs, outputs):
    return nn.Sequential(
        nn.Conv2d(in_channels=inputs, out_channels=outputs, kernel_size=3, padding=1),
        nn.LeakyReLU(0.2),
        nn.BatchNorm2d(outputs),

        nn.Conv2d(in_channels=outputs, out_channels=outputs, kernel_size=3, padding=1),
        nn.LeakyReLU(0.2),
        nn.BatchNorm2d(outputs),

        nn.Conv2d(in_channels=outputs, out_channels=outputs, kernel_size=3, padding=1),
        nn.LeakyReLU(0.2),
        nn.BatchNorm2d(outputs),

        nn.Softmax(),
    )


def add_block_f(inputs, outputs):
    return nn.Sequential(
        nn.Conv2d(in_channels=inputs, out_channels=outputs, kernel_size=3, padding=1),
        nn.LeakyReLU(0.2),
        nn.BatchNorm2d(outputs),

        nn.Conv2d(in_channels=outputs, out_channels=outputs, kernel_size=3, padding=1),
        nn.LeakyReLU(0.2),
        nn.BatchNorm2d(outputs),

        nn.Conv2d(in_channels=outputs, out_channels=outputs, kernel_size=3, padding=1),
        nn.LeakyReLU(0.2),
        nn.BatchNorm2d(outputs),

        nn.Linear(in_features=inputs, out_features=outputs),
    )


def concatenate_tensors(tensor1, tensor2):
    diff_height = tensor2.size()[2] - tensor1.size()[2]
    diff_width = tensor2.size()[3] - tensor1.size()[3]

    tensor1 = F.pad(tensor1,
                    [diff_width // 2, diff_width - diff_width // 2, diff_height // 2, diff_height - diff_height // 2])
    return torch.cat([tensor1, tensor2], dim=1)


def shape_as_string(shape):
    return str(list(shape))


class ChargridNetwork(nn.Module):
    def __init__(self, HW, C, num_classes=5, num_anchors=4):
        super(ChargridNetwork, self).__init__()

        '''
        Encoder definition
        '''
        self.first_a_block = add_block_a(HW, C, 1)
        self.second_a_block = add_block_a(C, 2 * C, 2)
        self.a_dash_block = add_block_a_dash(2 * C, 4 * C)
        self.first_a_double_dash_block = add_block_a_double_dash(4 * C, 8 * C, d=2)
        self.second_a_double_dash_block = add_block_a_double_dash(8 * C, 8 * C, d=4)

        '''
        Semantic Segmentation Decoder (ssd)
        '''
        self.ssd_first_b_block = add_block_b(12 * C, 4 * C)
        self.ssd_second_b_block = add_block_b(6 * C, 2 * C)
        self.ssd_c_block = add_block_c(3 * C, C)
        self.ssd_d_block = add_block_d_or_e(C, num_classes)

        '''
        Bounding Box Regression Decoder (bbrd)
        '''

        self.bbrd_first_b_block = add_block_b(12 * C, 4 * C)
        self.bbrd_second_b_block = add_block_b(6 * C, 2 * C)
        self.bbrd_c_block = add_block_c(3 * C, C)
        self.bbrd_e_block = add_block_d_or_e(C, 2*num_anchors)
        self.bbrd_f_block = add_block_f(C, 4*num_anchors)

    def forward(self, x):
        """
        Forward pass function for the network
        :param x:
        :return:
        """

        '''
        Forward pass through Encoder
        '''
        # print("input shape " + shape_as_string(x.shape))

        first_a_block_output = self.first_a_block(x)
        # print("after first a block " + shape_as_string(first_a_block_output.shape))

        second_a_block_output = self.second_a_block(first_a_block_output)
        # print("after second a block " + shape_as_string(second_a_block_output.shape))

        a_dash_block_output = self.a_dash_block(second_a_block_output)
        # print("after a dash block " + shape_as_string(a_dash_block_output.shape))

        first_a_double_dash_block_output = self.first_a_double_dash_block(a_dash_block_output)
        # print("after first a double dash block " + shape_as_string(first_a_double_dash_block_output.shape))

        second_a_double_dash_block_output = self.second_a_double_dash_block(first_a_double_dash_block_output)
        # print("after second a double dash block " + shape_as_string(second_a_double_dash_block_output.shape))

        '''
        Forward pass through semantic segmentation decoder
        '''
        ssd_first_b_block_input = concatenate_tensors(second_a_double_dash_block_output, a_dash_block_output)
        ssd_first_b_block_output = self.ssd_first_b_block(ssd_first_b_block_input)
        # print("ssd after first b block " + shape_as_string(ssd_first_b_block_output.shape))

        ssd_second_b_block_input = concatenate_tensors(ssd_first_b_block_output, second_a_block_output)
        ssd_second_b_block_output = self.ssd_second_b_block(ssd_second_b_block_input)
        # print("ssd after second b block " + shape_as_string(ssd_second_b_block_output.shape))

        ssd_c_block_input = concatenate_tensors(ssd_second_b_block_output, first_a_block_output)
        ssd_c_block_output = self.ssd_c_block(ssd_c_block_input)
        # print("ssd after c block " + shape_as_string(ssd_c_block_output.shape))

        d_block_output = self.ssd_d_block(ssd_c_block_output)
        # print("ssd after d block " + shape_as_string(d_block_output.shape))

        '''
        Forward pass through bounding box regression decoder
        '''
        bbrd_first_b_block_input = concatenate_tensors(second_a_double_dash_block_output, a_dash_block_output)
        bbrd_first_b_block_output = self.bbrd_first_b_block(bbrd_first_b_block_input)
        # print("bbrd after first b block " + shape_as_string(bbrd_first_b_block_output.shape))

        bbrd_second_b_block_input = concatenate_tensors(bbrd_first_b_block_output, second_a_block_output)
        bbrd_second_b_block_output = self.bbrd_second_b_block(bbrd_second_b_block_input)
        # print("bbrd after second b block " + shape_as_string(bbrd_second_b_block_output.shape))

        bbrd_c_block_input = concatenate_tensors(bbrd_second_b_block_output, first_a_block_output)
        bbrd_c_block_output = self.bbrd_c_block(bbrd_c_block_input)
        # print("bbrd after c block " + shape_as_string(bbrd_c_block_output.shape))

        bbrd_e_block_output = self.bbrd_e_block(bbrd_c_block_output)
        # print("bbrd after e block " + shape_as_string(bbrd_e_block_output.shape))

        bbrd_f_block_output = self.bbrd_e_block(bbrd_c_block_output)
        # print("bbrd after f block " + shape_as_string(bbrd_f_block_output.shape))

        return d_block_output, bbrd_e_block_output, bbrd_f_block_output



import ChargridDataset

# trainloader, testloader = ChargridDataset.get_dataset()
# net = ChargridNetwork(3, 64, 5, 4)
# img, l1, l2, l3 = next(iter(trainloader))
# net(img)
