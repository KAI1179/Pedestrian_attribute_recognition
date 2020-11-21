# !/usr/local/bin/python3
import os
import time
import argparse
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from datafolder.folder import Train_Dataset
import numpy as np

import torch.utils.data
from multi_task_network_Duke import pretrain_model


use_gpu = True
dataset_dict = {
    'duke'  :  'DukeMTMC-reID',
}

######################################################################
# Argument
# --------
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--data-path', default='/home/xuk/dataset', type=str, help='path to the dataset')
parser.add_argument('--dataset', default='duke', type=str, help='dataset:duke')
parser.add_argument('--backbone', default='multi_task_network_Duke', type=str, help='backbone: multi_task_network_Market, multi_task_network_Duke')
parser.add_argument('--batch-size', default=2, type=int, help='batch size')
parser.add_argument('--num-epoch', default=60, type=int, help='num of epoch')
parser.add_argument('--num-workers', default=2, type=int, help='num_workers')
parser.add_argument('--load-path-name', default='net_10.pth', type= str , help='last epoch name')
parser.add_argument('--RESUME', default=False, type= bool , help='resume the train')

args = parser.parse_args()

assert args.dataset in ['duke']

already_epoch = 0
dataset_name = dataset_dict[args.dataset]
model_name = args.backbone
data_dir = args.data_path
model_dir = os.path.join('./checkpoints', args.dataset, model_name)
load_model_dir = os.path.join(model_dir, args.load_path_name)


if not os.path.isdir(model_dir):
    os.makedirs(model_dir)


######################################################################
# Function
# --------
def save_network(network, epoch_label):
    save_filename = 'net_%s.pth'% epoch_label
    save_path = os.path.join(model_dir, save_filename)
    # torch.save(network.cpu().state_dict(), save_path)

    checkpoint = {
        "net": model.state_dict(),
        'optimizer': optimizer.state_dict(),
        "epoch": epoch_label,
        'lr_schedule': lr_scheduler.state_dict()
    }
    torch.save(checkpoint, save_path)

    if use_gpu:
        network.cuda()
    print('Save model to {}'.format(save_path))


def load_network(network, load_path):

    network.load_state_dict(load_path)

    epoch = checkpoint['epoch']
    print('Load epoch:%d model down!' % epoch)

    return epoch


def load_network_notfull(network, load_path):

    pretrained_dict = torch.load(load_path)

    model_dict = network.state_dict()
    model_dict.update({k: v for k, v in pretrained_dict.items() if k in model_dict})
    network.load_state_dict(model_dict)

    print('Load {} model down!'.format(load_path))

    return network



######################################################################
# Draw Curve
#-----------
x_epoch = []
y_loss = {} # loss history
y_loss['train'] = []
y_loss['val'] = []
y_err = {}
y_err['train'] = []
y_err['val'] = []

fig = plt.figure()
ax0 = fig.add_subplot(121, title="loss")
ax1 = fig.add_subplot(122, title="top1err")
def draw_curve(current_epoch):
    x_epoch.append(current_epoch)
    ax0.plot(x_epoch, y_loss['train'], 'bo-', label='train')
    ax0.plot(x_epoch, y_loss['val'], 'ro-', label='val')
    ax1.plot(x_epoch, y_err['train'], 'bo-', label='train')
    ax1.plot(x_epoch, y_err['val'], 'ro-', label='val')
    if current_epoch == 0:
        ax0.legend()
        ax1.legend()
    fig.savefig( os.path.join(model_dir, 'train.jpg'))


######################################################################
# DataLoader
# ---------
image_datasets = {}
image_datasets['train'] = Train_Dataset(data_dir, dataset_name=dataset_name, train_val='train')
image_datasets['val'] = Train_Dataset(data_dir, dataset_name=dataset_name, train_val='query')
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch_size,
                                             shuffle=True, num_workers=args.num_workers, drop_last=True)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
num_label = image_datasets['train'].num_label()



labels_list = image_datasets['train'].labels()



######################################################################
# Model and Optimizer
# ------------------
model = pretrain_model(23)
if use_gpu:
    model = model.cuda()

# loss
criterion_bce = nn.BCELoss()  # only used in two classification
criterion_ce = nn.CrossEntropyLoss()
criterion_bceLogits = nn.BCEWithLogitsLoss()  # only used in two classification

# optimizer


ignored_params_1 = list(map(id, model.block_base.parameters()))
ignored_params_2 = list(map(id, model.block_mid_1_1.parameters()))
ignored_params_3 = list(map(id, model.block_mid_1_2.parameters()))
ignored_params_4 = list(map(id, model.block_mid_2_1.parameters()))
ignored_params_5 = list(map(id, model.block_mid_2_2.parameters()))
ignored_params_6 = list(map(id, model.block_mid_3_1.parameters()))
ignored_params_7 = list(map(id, model.block_mid_3_2.parameters()))
ignored_params_8 = list(map(id, model.block_mid_4_1.parameters()))
ignored_params_9 = list(map(id, model.block_mid_4_2.parameters()))
ignored_params_10 = list(map(id, model.block_group_1.parameters()))
ignored_params_11 = list(map(id, model.block_group_2.parameters()))
ignored_params_12 = list(map(id, model.block_group_3.parameters()))
ignored_params_13 = list(map(id, model.block_group_4.parameters()))

ignored_params = ignored_params_1 + ignored_params_2 + ignored_params_3 + ignored_params_4 + ignored_params_5 \
                 + ignored_params_6 + ignored_params_7 + ignored_params_8 + ignored_params_9 + ignored_params_10\
                 + ignored_params_11 + ignored_params_12 + ignored_params_13

train_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
classifier_params_list = list(map(id, train_params))
pretrained_params = filter(lambda p: id(p) not in classifier_params_list, model.parameters())

optimizer = torch.optim.SGD([
    {'params': model.block_group_1_add.parameters(), 'lr': 0.1},
    {'params': model.block_group_2_add.parameters(), 'lr': 0.1},
    {'params': model.block_group_3_add.parameters(), 'lr': 0.1},
    {'params': model.block_group_4_add.parameters(), 'lr': 0.1},
    {'params': model.block_group_1_classifier.parameters(), 'lr': 0.1},
    {'params': model.block_group_2_classifier.parameters(), 'lr': 0.1},
    {'params': model.block_group_3_classifier_1.parameters(), 'lr': 0.1},
    {'params': model.block_group_3_classifier_2.parameters(), 'lr': 0.1},
    {'params': model.block_group_3_classifier_3.parameters(), 'lr': 0.1},
    {'params': model.block_group_4_classifier.parameters(), 'lr': 0.1},
    {'params': model.mix_classifier.parameters(), 'lr': 0.1},
    {'params': pretrained_params, 'lr': 0.01},
], momentum=0.9, weight_decay=5e-4, nesterov=True)


# lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, eta_min=0)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
# lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, step_size=5, gamma=0.1)
# lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=0.01, eta_min=0.0001)




# if os.path.exists(load_model_dir):
#     already_epoch = load_network(model, load_model_dir)
#     print('加载 epoch {} 成功！'.format(already_epoch))
# else:
#     already_epoch = 0
#     print('无保存模型，将从头开始训练！')


if os.path.exists(load_model_dir):
    model = load_network_notfull(model, load_model_dir)
    print('加载 model {} 成功！'.format(load_model_dir))
else:

    print('无保存模型，将从头开始训练！')


if args.RESUME:

    checkpoint = torch.load(load_model_dir)  # 加载断点
    model.load_state_dict(checkpoint['net'])  # 加载模型可学习参数
    # optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
    already_epoch = checkpoint['epoch']  # 设置开始的epoch


######################################################################
def train_model(model, optimizer, scheduler, num_epochs, start_epoch = 1):
    since = time.time()

    print(start_epoch)

    # for epoch in range(1, num_epochs+1):
    for epoch in range(start_epoch, num_epochs + 1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)


        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for count, (images, indices, labels, ids, cams, names) in enumerate(dataloaders[phase]):

                labels = labels.float()
                
                if use_gpu:
                    images = images.cuda()
                    labels = labels.cuda()

                # zero the parameter gradients
                optimizer.zero_grad()
                # forward

                aux, pred_label = model(images)

                pred_loss = criterion_bce(pred_label, labels)
                aux_loss = criterion_bce(aux, labels)
                total_loss = pred_loss + aux_loss

                if phase == 'train':
                    total_loss.backward()
                    for name, parms in model.named_parameters():
                        if name == 'class_1.blockfc_2':
                            print('-->name:', name, '-->grad_requirs:', parms.requires_grad, \
                                 ' -->grad_value:', parms.grad)
                    optimizer.step()

                preds = torch.gt(pred_label, torch.ones_like(pred_label)/2 )  # 返回每个判断正误 Ture or False
                # statistics
                running_loss += total_loss.item()
                running_corrects += torch.sum(preds == labels.byte()).item() / num_label

                if count % 100 == 0:
                    print('step: ({}/{})  |  label loss: {:.4f}'.format(
                        count*args.batch_size, dataset_sizes[phase], total_loss.item()))

            epoch_loss = running_loss / len(dataloaders[phase])
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            print(lr_scheduler.get_lr())  # 打印lr
            y_loss[phase].append(epoch_loss)
            y_err[phase].append(1.0-epoch_acc)
            # deep copy the model
            if phase == 'val':
                last_model_wts = model.state_dict()
                if epoch % 10 == 0:
                    save_network(model, epoch)
                    
                draw_curve(epoch)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # load best model weights
    model.load_state_dict(last_model_wts)
    save_network(model, 'last')


######################################################################

# Main
# -----

train_model(model, optimizer, lr_scheduler, num_epochs=args.num_epoch, start_epoch = already_epoch + 1)

