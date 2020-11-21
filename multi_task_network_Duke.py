import torch
import torchvision
import torch.nn as nn
from torch.nn import init
# from torchviz import make_dot
from block_base import block_base
from block_mid import block_mid_1, block_mid_2
from block_group import block_group
# from apex import amp


load_path = './checkpoints/vgg16_bn-6c64b313.pth'  # vgg16预训练权重文件
pretrained_dict = torch.load(load_path)
# print(pretrained_dict)
#

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


class classifier(nn.Module):
    def __init__(self, in_features, class_num=3):
        super(classifier, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=512),
            nn.Dropout(0.2),
            nn.Linear(in_features=512, out_features=class_num),
            nn.Sigmoid()
        )

        for m in self.classifier.children():
            m.apply(weights_init_classifier)

    def forward(self, x):
        x = self.classifier(x)

        return x


class pretrain_model(nn.Module):
    def __init__(self, class_num=23):
        super(pretrain_model, self).__init__()
        self.class_num = class_num

        # using_amp = True
        # if using_amp:
        #     amp.register_float_function(torch, 'sigmoid')

        self.block_base = block_base(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool_base = nn.MaxPool2d(2)
        #  #load weight**************************
        model_dict = self.block_base.state_dict()
        model_dict.update({k: v for k, v in pretrained_dict.items() if k in model_dict})
        self.block_base.load_state_dict(model_dict)
        #  #load weight**************************



        self.block_mid_1_1 = block_mid_1(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool_mid_1_1 = nn.MaxPool2d(2)
        #  #load weight**************************
        model_dict = self.block_mid_1_1.state_dict()
        model_dict.update({k: v for k, v in pretrained_dict.items() if k in model_dict})
        self.block_mid_1_1.load_state_dict(model_dict)
        #  #load weight**************************


        self.block_mid_1_2 = block_mid_2(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.pool_mid_1_2 = nn.MaxPool2d(2)
        #  #load weight**************************
        model_dict = self.block_mid_1_2.state_dict()
        model_dict.update({k: v for k, v in pretrained_dict.items() if k in model_dict})
        self.block_mid_1_2.load_state_dict(model_dict)
        #  #load weight**************************


        self.block_mid_2_1 = block_mid_1(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool_mid_2_1 = nn.MaxPool2d(2)
        #  #load weight**************************
        model_dict = self.block_mid_2_1.state_dict()
        model_dict.update({k: v for k, v in pretrained_dict.items() if k in model_dict})
        self.block_mid_2_1.load_state_dict(model_dict)
        #  #load weight**************************

        self.block_mid_2_2 = block_mid_2(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.pool_mid_2_2 = nn.MaxPool2d(2)
        #  #load weight**************************
        model_dict = self.block_mid_2_2.state_dict()
        model_dict.update({k: v for k, v in pretrained_dict.items() if k in model_dict})
        self.block_mid_2_2.load_state_dict(model_dict)
        #  #load weight**************************

        self.block_mid_3_1 = block_mid_1(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool_mid_3_1 = nn.MaxPool2d(2)
        #  #load weight**************************
        model_dict = self.block_mid_3_1.state_dict()
        model_dict.update({k: v for k, v in pretrained_dict.items() if k in model_dict})
        self.block_mid_3_1.load_state_dict(model_dict)
        #  #load weight**************************

        self.block_mid_3_2 = block_mid_2(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.pool_mid_3_2 = nn.MaxPool2d(2)
        #  #load weight**************************
        model_dict = self.block_mid_3_2.state_dict()
        model_dict.update({k: v for k, v in pretrained_dict.items() if k in model_dict})
        self.block_mid_3_2.load_state_dict(model_dict)
        #  #load weight**************************

        self.block_mid_4_1 = block_mid_1(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool_mid_4_1 = nn.MaxPool2d(2)
        #  #load weight**************************
        model_dict = self.block_mid_4_1.state_dict()
        model_dict.update({k: v for k, v in pretrained_dict.items() if k in model_dict})
        self.block_mid_4_1.load_state_dict(model_dict)
        #  #load weight**************************

        self.block_mid_4_2 = block_mid_2(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.pool_mid_4_2 = nn.MaxPool2d(2)
        #  #load weight**************************
        model_dict = self.block_mid_4_2.state_dict()
        model_dict.update({k: v for k, v in pretrained_dict.items() if k in model_dict})
        self.block_mid_4_2.load_state_dict(model_dict)
        #  #load weight**************************

        self.block_group_1 = block_group(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)

        #  #load weight**************************
        model_dict = self.block_group_1.state_dict()
        model_dict.update({k: v for k, v in pretrained_dict.items() if k in model_dict})
        self.block_group_1.load_state_dict(model_dict)
        #  #load weight**************************

        self.block_group_1_add = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=1, stride=1, padding=1),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        for m in self.block_group_1_add.children():
            m.apply(weights_init_kaiming)

        self.block_group_2 = block_group(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        # #  #load weight**************************
        # model_dict = self.block_group_2.state_dict()
        # model_dict.update({k: v for k, v in pretrained_dict.items() if k in model_dict})
        # self.block_group_2.load_state_dict(model_dict)
        # #  #load weight**************************
        self.block_group_2_add = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=1, stride=1, padding=1),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        for m in self.block_group_2_add.children():
            m.apply(weights_init_kaiming)

        self.block_group_3 = block_group(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        # #  #load weight**************************
        # model_dict = self.block_group_3.state_dict()
        # model_dict.update({k: v for k, v in pretrained_dict.items() if k in model_dict})
        # self.block_group_3.load_state_dict(model_dict)
        # #  #load weight**************************
        self.block_group_3_add = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=1, stride=1, padding=1),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        for m in self.block_group_3_add.children():
            m.apply(weights_init_kaiming)

        self.block_group_4 = block_group(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        # #  #load weight**************************
        # model_dict = self.block_group_4.state_dict()
        # model_dict.update({k: v for k, v in pretrained_dict.items() if k in model_dict})
        # self.block_group_4.load_state_dict(model_dict)
        # #  #load weight**************************
        self.block_group_4_add = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=1, stride=1, padding=1),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        for m in self.block_group_4_add.children():
            m.apply(weights_init_kaiming)

        self.block_group_1_classifier = nn.Sequential(  # group_1 :帽子，背包，手包  # 改成3个二分类
            classifier(in_features=2048, class_num=3),
        )
        self.block_group_2_classifier = nn.Sequential(  # group_2： 包，靴子，鞋子颜色
            classifier(in_features=2048, class_num=3),
        )
        self.block_group_3_classifier_1 = nn.Sequential(  # group_3： 上衣长度
            classifier(in_features=2048, class_num=1),
        )
        self.block_group_3_classifier_2 = nn.Sequential(  # group_3： 上衣颜色
            classifier(in_features=2048, class_num=8),
        )
        self.block_group_3_classifier_3 = nn.Sequential(  # group_3： 下衣颜色
            classifier(in_features=2048, class_num=7),
        )
        self.block_group_4_classifier = nn.Sequential(  # group_4： 性别
            classifier(in_features=2048, class_num=1),
        )

        self.mix_classifier = nn.Sequential(
            classifier(in_features=2048 * 4, class_num=23)
        )

    def forward(self, x):
        x = self.block_base(x)  # 输出 x =[b, 64, 144, 72]
        x = self.pool_base(x)

        x_1 = self.block_mid_1_1(x)
        x_1 = self.pool_mid_1_1(x_1)
        x_1 = self.block_mid_1_2(x_1)
        x_1 = self.pool_mid_1_2(x_1)
        x_1 = self.block_group_1(x_1)
        x_1 = self.block_group_1_add(x_1)
        x_1 = x_1.view(x_1.size(0), -1)
        x_1_out = self.block_group_1_classifier(x_1)

        x_2 = self.block_mid_2_1(x)
        x_2 = self.pool_mid_2_1(x_2)
        x_2 = self.block_mid_2_2(x_2)
        x_2 = self.pool_mid_2_2(x_2)
        x_2 = self.block_group_2(x_2)
        x_2 = self.block_group_2_add(x_2)
        x_2 = x_2.view(x_2.size(0), -1)
        x_2_out = self.block_group_2_classifier(x_2)

        x_3 = self.block_mid_3_1(x)
        x_3 = self.pool_mid_3_1(x_3)
        x_3 = self.block_mid_3_2(x_3)
        x_3 = self.pool_mid_3_2(x_3)
        x_3 = self.block_group_3(x_3)
        x_3 = self.block_group_3_add(x_3)
        x_3 = x_3.view(x_3.size(0), -1)
        x_3_1_out = self.block_group_3_classifier_1(x_3)  # x = [b, 1]
        x_3_2_out = self.block_group_3_classifier_2(x_3)  # x = [b, 8]
        x_3_3_out = self.block_group_3_classifier_3(x_3)

        x_4 = self.block_mid_4_1(x)
        x_4 = self.pool_mid_4_1(x_4)
        x_4 = self.block_mid_4_2(x_4)
        x_4 = self.pool_mid_4_2(x_4)
        x_4 = self.block_group_4(x_4)
        x_4 = self.block_group_4_add(x_4)
        x_4 = x_4.view(x_4.size(0), -1)

        # x_4 = torch.cat((x_1, x_2, x_3, x_4), dim=1)  # 输出 x_4 =[b, 2048]
        x_4_out = self.block_group_4_classifier(x_4)

        preds = []
        preds.append(x_1_out[:, 0].reshape([-1, 1]))
        preds.append(x_2_out[:, 0].reshape([-1, 1]))
        preds.append(x_1_out[:, 1].reshape([-1, 1]))
        preds.append(x_2_out[:, 1].reshape([-1, 1]))
        preds.append(x_4_out)
        preds.append(x_1_out[:, 2].reshape([-1, 1]))
        preds.append(x_2_out[:, 2].reshape([-1, 1]))
        preds.append(x_3_1_out)
        preds.append(x_3_2_out)
        preds.append(x_3_3_out)

        preds = torch.cat(preds, dim=1)

        mix = []
        mix.append(x_1)
        mix.append(x_2)
        mix.append(x_3)
        mix.append(x_4)
        out = torch.cat(mix, dim=1)
        out = self.mix_classifier(out)
        return preds, out

if __name__ == '__main__':
    model = pretrain_model()
    x = torch.randn([1, 3, 288, 144])
    y1, y = model(x)
    print(y.shape)
    print(y1.shape)






