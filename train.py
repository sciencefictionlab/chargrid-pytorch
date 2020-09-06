from config import autoconfigure
import os
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch

import ChargridDataset
from ChargridNetwork import ChargridNetwork


def init_weights(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.uniform_(m.weight, a=0.0, b=1.0)


def init_weights_in_last_layers(net):
    torch.nn.init.constant_(net.ssd_d_block[6].weight, 1e-3)
    torch.nn.init.constant_(net.bbrd_e_block[6].weight, 1e-3)
    torch.nn.init.constant_(net.bbrd_f_block[6].weight, 1e-3)


if __name__ == '__main__':
    # in the name of reproducibility
    torch.manual_seed(0)
    autoconfigure()

    HW = 3
    C = 64
    num_classes = 5
    num_anchors = 4

    trainloader, testloader = ChargridDataset.get_dataset()

    net = ChargridNetwork(3, 64, 5, 4)
    net = net.apply(init_weights)
    init_weights_in_last_layers(net)

    model_dir = os.getenv('MODEL_OUTPUT_DIR')

    loss1 = nn.BCELoss()
    loss2 = nn.BCELoss()
    loss3 = nn.SmoothL1Loss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    losses = {
        'loss1': [],
        'loss2': [],
        'loss3': [],
        'combined_losses': []
    }

    num_epochs = 5
    for epoch in range(num_epochs):
        correct = 0
        total = 0
        final_loss = 0.0
        for i, data in enumerate(trainloader, 0):

            for inputs, label1, label2, label3 in trainloader:
                inputs = inputs

                optimizer_ft.zero_grad()
                output1, output2, output3 = net(inputs)

                loss_1 = loss1(output1, label1.float())
                print(loss_1)
                losses['loss1'].append(loss_1)

                loss_2 = loss2(output2, label2.float())
                print(loss_2)
                losses['loss2'].append(loss_2)

                loss_3 = loss3(output3, label3.float())
                print(loss_3)
                losses['loss3'].append(loss_3)

                final_loss = loss_1 + loss_2 + loss_3
                losses['combined_losses'].append(final_loss)
                final_loss.backward()
                optimizer_ft.step()

                # _, predicted = torch.max(output1.data, 1)

            exp_lr_scheduler.step()
            #                 total += labels.size(0)
            #                 correct += (predicted == labels).sum().item()

            # correct += (outputs == labels).float().sum()
        print("Epoch {}/{}, Loss: {:.3f}".format(epoch + 1, num_epochs, final_loss.item()))

        torch.save(net.state_dict(), os.path.join(model_dir, 'epoch-{}.pt'.format(epoch)))

    print('================')
    print('================')
    print('================')
    print('loss1: ' + str(losses['loss1']))
    print('loss2: ' + str(losses['loss2']))
    print('loss3: ' + str(losses['loss3']))
    print('combined: ' + str(losses['combined_losses']))
    print('Finished Training')
