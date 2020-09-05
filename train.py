import os
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch

import ChargridDataset
from ChargridNetwork import ChargridNetwork

HW = 3
C = 64
num_classes = 5
num_anchors = 4

trainloader, testloader = ChargridDataset.get_dataset()

net = ChargridNetwork(3, 64, 5, 4)

model_dir = './output/'

loss1 = nn.CrossEntropyLoss()
loss2 = nn.BCELoss()
loss3 = nn.SmoothL1Loss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

num_epochs = 10
for epoch in range(num_epochs):
    correct = 0
    total = 0
    for i, data in enumerate(trainloader, 0):
        exp_lr_scheduler.step()

        for inputs, label_1, label_2, label_3 in trainloader:
            inputs = inputs
            label1 = label_1
            label2 = label_2
            label3 = label_3
            print("label1", label1[0].shape)
            print("after squeeze", label1[0].squeeze(1).shape)

            optimizer_ft.zero_grad()
            output1, output2, output3 = net(inputs)
            print("output1", output1.shape)

            loss_1 = loss1(output1, label1)
            loss_2 = loss2(output2, label2)
            loss_3 = loss3(output3, label3)
            final_loss = loss_1 + loss_2 + loss_3
            final_loss.backward()
            optimizer_ft.step()

            _, predicted = torch.max(output1.data, 1)
        #                 total += labels.size(0)
        #                 correct += (predicted == labels).sum().item()

        # correct += (outputs == labels).float().sum()
        print("Epoch {}/{}, Loss: {:.3f}".format(epoch + 1, num_epochs, final_loss.item()))

        torch.save(net.state_dict(), os.path.join(model_dir, 'epoch-{}.pt'.format(epoch)))

print('Finished Training')
