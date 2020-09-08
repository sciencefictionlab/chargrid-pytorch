import torch
from ChargridNetwork import ChargridNetwork
import ChargridDataset
from train import init_weights, init_weights_in_last_layers
from config import autoconfigure
import os
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

autoconfigure()
HW = 3
C = 64
num_classes = 5
num_anchors = 4
net = ChargridNetwork(3, 64, 5, 4)
net = net.apply(init_weights)
init_weights_in_last_layers(net)

model = net.load_state_dict(torch.load('./output/epoch-0.pt', map_location=lambda storage, loc: storage))

model_dir = os.getenv('MODEL_OUTPUT_DIR')

trainloader, testloader = ChargridDataset.get_dataset()

loss1 = nn.BCELoss()
loss2 = nn.BCELoss()
loss3 = nn.SmoothL1Loss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

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
    final_loss = 0.0
    for i, data in enumerate(trainloader, 0):

        for inputs, label1, label2, label3 in trainloader:
            inputs = inputs

            optimizer_ft.zero_grad()
            output1, output2, output3 = model(inputs)

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

            # _, predicted1 = torch.max(output1.squeeze(), dim=0)
            # total1 += label1.size(0)

        exp_lr_scheduler.step()

    print("Epoch {}/{}, Loss: {:.3f}".format(epoch + 2, num_epochs, final_loss.item()))

    torch.save({
        'epoch': epoch + 2,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer_ft.state_dict(),
        'loss': losses,
    }, os.path.join(model_dir, 'epoch-{}.pt'.format(epoch)))

print('================')
print('================')
print('================')
print('loss1: ' + str(losses['loss1']))
print('loss2: ' + str(losses['loss2']))
print('loss3: ' + str(losses['loss3']))
print('combined: ' + str(losses['combined_losses']))
print('Finished Training')