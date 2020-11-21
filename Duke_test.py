import os
import argparse
import scipy.io
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# from sklearn.exceptions import UndefinedMetricWarning
from datafolder.folder import Test_Dataset
from multi_task_network_Duke import pretrain_model


######################################################################
# Settings
# ---------
use_gpu = True
dataset_dict = {
    'market'  :  'Market-1501',
    'duke'  :  'DukeMTMC-reID',
}
num_cls_dict = { 'market':30, 'duke':23 }
num_ids_dict = { 'market':751, 'duke':702 }


######################################################################
# Argument
# ---------
parser = argparse.ArgumentParser(description='Testing')
parser.add_argument('--data-path', default='/home/xuk/dataset/', type=str, help='path to the dataset')
parser.add_argument('--dataset', default='duke', type=str, help='dataset')
parser.add_argument('--backbone', default='multi_task_network_Duke', type=str, help='model')
parser.add_argument('--batch-size', default=50, type=int, help='batch size')
parser.add_argument('--num-epoch', default=60, type=int, help='num of epoch')
parser.add_argument('--num-workers', default=2, type=int, help='num_workers')
parser.add_argument('--which-epoch',default='40', type=str, help='0,1,2,3...or last')
parser.add_argument('--print-table',action='store_true', help='print results with table format')
parser.add_argument('--use-id', action='store_true', help='use identity loss')
args = parser.parse_args()

assert args.dataset in ['market', 'duke']
# assert args.backbone in ['resnet50', 'resnet34', 'resnet18', 'densenet121']

dataset_name = dataset_dict[args.dataset]
model_name = args.backbone
data_dir = args.data_path
model_dir = os.path.join('./checkpoints', args.dataset, model_name)
result_dir = os.path.join('./result', args.dataset, model_name)

if not os.path.isdir(result_dir):
    os.makedirs(result_dir)
if not os.path.isdir(model_dir):
    os.makedirs(model_dir)


######################################################################
# Function
# ---------
def load_network(network):
    save_path = os.path.join(model_dir,'net_%s.pth'%args.which_epoch)
    print(save_path)
    checkpoint = torch.load(save_path)  # 加载断点
    model.load_state_dict(checkpoint['net'])  # 加载模型可学习参数

    # network.load_state_dict(torch.load(save_path))
    print('Resume model from {}'.format(save_path))
    return network


def get_dataloader():
    image_datasets = {}
    image_datasets['gallery'] = Test_Dataset(data_dir, dataset_name=dataset_name, query_gallery='gallery')
    image_datasets['query'] = Test_Dataset(data_dir, dataset_name=dataset_name, query_gallery='query')
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch_size,
                                                 shuffle=True, num_workers=args.num_workers)
                  for x in ['gallery', 'query']}
    return dataloaders


def check_metric_vaild(y_pred, y_true):
    if y_true.min() == y_true.max() == 0:   # precision
        return False
    if y_pred.min() == y_pred.max() == 0:   # recall
        return False
    return True


######################################################################
# Load Data
# ---------
# Note that we only perform evaluation on gallery set.
test_loader = get_dataloader()['gallery']

attribute_list = test_loader.dataset.labels()

num_label = len(attribute_list)
num_sample = len(test_loader.dataset)



######################################################################
# Model
# ---------
# model = InceptionV3(num_label, stage= 'train')
model = pretrain_model(23)
model = load_network(model)
if use_gpu:
    model = model.cuda()
model.train(False)  # Set model to evaluate mode
# model.aux_logits = False



######################################################################
# Testing
# ---------

preds_tensor = np.empty(shape=[0, num_label], dtype=np.byte)   # shape = (num_sample, num_label)
labels_tensor = np.empty(shape=[0, num_label], dtype=np.byte)   # shape = (num_sample, num_label)

# Iterate over data.
with torch.no_grad():
    for count, (images, labels, ids, file_name) in enumerate(test_loader):
        # move input to GPU
        if use_gpu:
            images = images.cuda()

        aux, pred_label = model(images)

        preds = torch.gt(pred_label, torch.ones_like(pred_label)/2)  # 概率(小数)->True/False

        # transform to numpy formatSave model
        labels = labels.cpu().numpy()
        preds = preds.cpu().numpy()

        preds_tensor = np.append(preds_tensor, preds, axis=0)
        labels_tensor = np.append(labels_tensor, labels, axis=0)

        if count*args.batch_size % 5000 == 0:
            print('Step: {}/{}'.format(count*args.batch_size, num_sample))

# Evaluation.
accuracy_list = []
precision_list = []
recall_list = []
f1_score_list = []
average_precision = 0.0
average_recall = 0.0
average_f1score = 0.0
valid_count = 0
for i, name in enumerate(attribute_list):
    y_true, y_pred = labels_tensor[:, i], preds_tensor[:, i]
    accuracy_list.append(accuracy_score(y_true, y_pred))

    if check_metric_vaild(y_pred, y_true):    # exclude ill-defined cases
        precision_list.append(precision_score(y_true, y_pred, average='binary'))
        recall_list.append(recall_score(y_true, y_pred, average='binary'))
        f1_score_list.append(f1_score(y_true, y_pred, average='binary'))
        average_precision += precision_list[-1]
        average_recall += recall_list[-1]
        average_f1score += f1_score_list[-1]
        valid_count += 1
    else:
        precision_list.append(-1)
        recall_list.append(-1)
        f1_score_list.append(-1)

average_acc = np.mean(accuracy_list)
average_precision = average_precision / valid_count
average_recall = average_recall / valid_count
average_f1score = average_f1score / valid_count


######################################################################
# Print
# ---------
print("\n"
      "The Precision, Recall and F-score are ignored for some ill-defined cases."
      "\n")

if args.print_table:
    from prettytable import PrettyTable
    table = PrettyTable(['attribute', 'accuracy', 'precision', 'recall', 'f1 score'])
    for i, name in enumerate(attribute_list):
        table.add_row([name,
               '%.3f' % accuracy_list[i],
               '%.3f' % precision_list[i] if precision_list[i] >= 0.0 else '-',
               '%.3f' % recall_list[i] if recall_list[i] >= 0.0 else '-',
               '%.3f' % f1_score_list[i] if f1_score_list[i] >= 0.0 else '-',
               ])
    print(table)


print('Average accuracy: {:.4f}'.format(average_acc))
# print('Average precision: {:.4f}'.format(average_precision))
# print('Average recall: {:.4f}'.format(average_recall))
print('Average f1 score: {:.4f}'.format(average_f1score))

# Save results.
result = {
    'average_acc'       :   average_acc,
    'average_f1score'   :   average_f1score,
    'accuracy_list'     :   accuracy_list,
    'precision_list'    :   precision_list,
    'recall_list'       :   recall_list,
    'f1_score_list'     :   f1_score_list,
}
scipy.io.savemat(os.path.join(result_dir, 'acc.mat'), result)


