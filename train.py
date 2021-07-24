from config import autoconfigure
import os
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch

import ChargridDataset
from ChargridNetwork import ChargridNetwork

from datetime import datetime

def init_weights(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.uniform_(m.weight, a=0.0, b=1.0)


def init_weights_in_last_layers(net):
    torch.nn.init.constant_(net.ssd_d_block[6].weight, 1e-3)
    torch.nn.init.constant_(net.bbrd_e_block[6].weight, 1e-3)
    torch.nn.init.constant_(net.bbrd_f_block[6].weight, 1e-3)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--restart", action="store_true", help="Restart the training, otherwise resume where it left.")
    parser.add_argument("-c", "--checkpoint", default=None, help="Checkpoint epoch to load from.", type=int)
    parser.add_argument("-e", "--epochs", default=10, help="Number of more epochs to run for.", type=int)
    args = parser.parse_args()

    # in the name of reproducibility
    # device = torch.device('cpu')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Running on device:", device)
    torch.manual_seed(0)
    autoconfigure()

    HW = 3
    C = 64
    num_classes = 5
    num_anchors = 4

    trainloader, testloader = ChargridDataset.get_dataset(train_batch_size=2)

    net = ChargridNetwork(3, 64, 5, 4)
    net = net.to(device)
    model_dir = os.getenv('MODEL_OUTPUT_DIR')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    optimizer_ft = optim.SGD(net.to(device).parameters(), lr=0.001, momentum=0.9)
    
    if(args.restart):
        net = net.apply(init_weights)
        init_weights_in_last_layers(net)
        start_epoch = 0
        losses = {
            'loss1': [],
            'loss2': [],
            'loss3': [],
            'combined_losses': []
        }
    elif(args.checkpoint is not None):
        print("Loading model from checkpoint-{}".format(args.checkpoint))
        checkpoint = torch.load(os.path.join(model_dir, 'epoch-'+str(args.checkpoint)+'.pt'))
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer_ft.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = args.checkpoint+1
        losses = torch.load(os.path.join(model_dir, 'losses'))['losses']
        # assert(len(losses['loss1']) == int(args.checkpoint)+1)
    else:
        print("Provide last checkpoint number to load from (using -c).")
        exit(1)
    
    loss1 = nn.BCELoss()
    loss2 = nn.BCELoss()
    loss3 = nn.SmoothL1Loss()

    # Observe that all parameters are being optimized
    

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    

    for epoch in range(start_epoch, start_epoch+args.epochs):
        final_loss = 0.0
        print("Epoch {}/{}. ".format(epoch, start_epoch+args.epochs-1), end='', flush=True)
        epoch_start_time = datetime.now()
        for batch_num, (inputs, label1, label2, label3) in enumerate(trainloader):
            # print(" > Started batch: {}".format(batch_num))

            inputs = inputs.to(device)
            label1 = label1.to(device)
            label2 = label2.to(device)
            label3 = label3.to(device)

            optimizer_ft.zero_grad()
            output1, output2, output3 = net(inputs)

            loss_1 = loss1(output1, label1.float())
            # print(loss_1)
            losses['loss1'].append(loss_1)

            loss_2 = loss2(output2, label2.float())
            # print(loss_2)
            losses['loss2'].append(loss_2)

            loss_3 = loss3(output3, label3.float())
            # print(loss_3)
            losses['loss3'].append(loss_3)

            final_loss = loss_1 + loss_2 + loss_3
            losses['combined_losses'].append(final_loss)
            final_loss.backward()
            optimizer_ft.step()

            # _, predicted1 = torch.max(output1.squeeze(), dim=0)
            # total1 += label1.size(0)

        exp_lr_scheduler.step()

        torch.save({
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer_ft.state_dict(),
        }, os.path.join(model_dir, 'epoch-{}.pt'.format(epoch)))
        torch.save({
            "losses": losses
        }, os.path.join(os.path.join(model_dir, 'losses')))
        
        print("Loss: {:.3f}, Time: {:.3f} s".format(final_loss.item(), (datetime.now() - epoch_start_time).total_seconds()))

        torch.cuda.empty_cache()
        
    print('================')
    print('================')
    # print('loss1: ' + str([l.item() for l in losses['loss1']]))
    # print('loss2: ' + str([l.item() for l in losses['loss2']]))
    # print('loss3: ' + str([l.item() for l in losses['loss3']]))
    # print('combined: ' + str([l.item() for l in losses['combined_losses']]))
    print('Finished Training')
