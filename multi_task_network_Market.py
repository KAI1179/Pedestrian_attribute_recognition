import torch
import torchvision
import torch.nn as nn
from torch.nn import init
# from torchviz import make_dot
from block_base import block_base
from block_mid import block_mid_1, block_mid_2
from block_group import block_group


load_path = './checkpoints/vgg16_bn-6c64b313.pth'
pretrained_dict = torch.load(load_path, map_location='cpu')
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
            nn.Dropout(0.5),
            nn.Linear(in_features=512, out_features=class_num),
            nn.Sigmoid()
        )

        for m in self.classifier.children():
            m.apply(weights_init_classifier)

    def forward(self, x):
        x = self.classifier(x)

        return x


class pretrain_model(nn.Module):
    def __init__(self, class_num=30):
        super(pretrain_model, self).__init__()
        self.class_num = class_num


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

        self.block_mid_5_1 = block_mid_1(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool_mid_5_1 = nn.MaxPool2d(2)
        #  #load weight**************************
        model_dict = self.block_mid_5_1.state_dict()
        model_dict.update({k: v for k, v in pretrained_dict.items() if k in model_dict})
        self.block_mid_5_1.load_state_dict(model_dict)
        #  #load weight**************************

        self.block_mid_5_2 = block_mid_2(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.pool_mid_5_2 = nn.MaxPool2d(2)
        #  #load weight**************************
        model_dict = self.block_mid_5_2.state_dict()
        model_dict.update({k: v for k, v in pretrained_dict.items() if k in model_dict})
        self.block_mid_5_2.load_state_dict(model_dict)
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
        #  #load weight**************************
        model_dict = self.block_group_2.state_dict()
        model_dict.update({k: v for k, v in pretrained_dict.items() if k in model_dict})
        self.block_group_2.load_state_dict(model_dict)
        #  #load weight**************************
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
        #  #load weight**************************
        model_dict = self.block_group_3.state_dict()
        model_dict.update({k: v for k, v in pretrained_dict.items() if k in model_dict})
        self.block_group_3.load_state_dict(model_dict)
        #  #load weight**************************
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
        #  #load weight**************************
        model_dict = self.block_group_4.state_dict()
        model_dict.update({k: v for k, v in pretrained_dict.items() if k in model_dict})
        self.block_group_4.load_state_dict(model_dict)
        #  #load weight**************************
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

        self.block_group_5 = block_group(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        #  #load weight**************************
        model_dict = self.block_group_5.state_dict()
        model_dict.update({k: v for k, v in pretrained_dict.items() if k in model_dict})
        self.block_group_5.load_state_dict(model_dict)
        #  #load weight**************************
        self.block_group_5_add = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=1, stride=1, padding=1),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        for m in self.block_group_5_add.children():
            m.apply(weights_init_kaiming)

        self.block_group_1_classifier_1 = nn.Sequential(  # group_1 :背包，手包
            classifier(in_features=2048, class_num=2),
        )
        self.block_group_1_classifier_2 = nn.Sequential(  # group_1 :下衣颜色
            classifier(in_features=2048, class_num=9),
        )
        self.block_group_2_classifier_1 = nn.Sequential(  # group_2： 包，头发， 下衣长
            classifier(in_features=2048, class_num=3),
        )
        self.block_group_2_classifier_2 = nn.Sequential(  # group_2： 上衣颜色
            classifier(in_features=2048, class_num=8),
        )
        self.block_group_3_classifier = nn.Sequential(  # group_3： 帽子，袖长， 下衣类型
            classifier(in_features=2048, class_num=3),
        )
        self.block_group_4_classifier = nn.Sequential(  # group_4： 性别
            classifier(in_features=2048, class_num=1),
        )
        self.block_group_5_classifier = nn.Sequential(  # group_4： 年龄  4种
            classifier(in_features=2048, class_num=4),
        )
        self.mix_classifier = nn.Sequential(
            classifier(in_features=2048 * 5, class_num=30)
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
        x_1_out_1 = self.block_group_1_classifier_1(x_1)
        x_1_out_2 = self.block_group_1_classifier_2(x_1)

        x_2 = self.block_mid_2_1(x)
        x_2 = self.pool_mid_2_1(x_2)
        x_2 = self.block_mid_2_2(x_2)
        x_2 = self.pool_mid_2_2(x_2)
        x_2 = self.block_group_2(x_2)
        x_2 = self.block_group_2_add(x_2)
        x_2 = x_2.view(x_2.size(0), -1)
        x_2_out_1 = self.block_group_2_classifier_1(x_2)
        x_2_out_2 = self.block_group_2_classifier_2(x_2)

        x_3 = self.block_mid_3_1(x)
        x_3 = self.pool_mid_3_1(x_3)
        x_3 = self.block_mid_3_2(x_3)
        x_3 = self.pool_mid_3_2(x_3)
        x_3 = self.block_group_3(x_3)
        x_3 = self.block_group_3_add(x_3)
        x_3 = x_3.view(x_3.size(0), -1)
        x_3_out = self.block_group_3_classifier(x_3)  # x = [b, 1]


        x_4 = self.block_mid_4_1(x)
        x_4 = self.pool_mid_4_1(x_4)
        x_4 = self.block_mid_4_2(x_4)
        x_4 = self.pool_mid_4_2(x_4)
        x_4 = self.block_group_4(x_4)
        x_4 = self.block_group_4_add(x_4)
        x_4 = x_4.view(x_4.size(0), -1)
        x_4_out = self.block_group_4_classifier(x_4)

        x_5 = self.block_mid_5_1(x)
        x_5 = self.pool_mid_5_1(x_5)
        x_5 = self.block_mid_5_2(x_5)
        x_5 = self.pool_mid_5_2(x_5)
        x_5 = self.block_group_5(x_5)
        x_5 = self.block_group_5_add(x_5)
        x_5 = x_5.view(x_5.size(0), -1)
        x_5_out = self.block_group_5_classifier(x_5)



        preds = []
        preds.append(x_5_out)  # 年龄
        preds.append(x_1_out_1[:, 0].reshape([-1, 1]))  # 背包
        preds.append(x_2_out_1[:, 0].reshape([-1, 1]))  # 包
        preds.append(x_1_out_1[:, 1].reshape([-1, 1]))  # 手包
        preds.append(x_3_out[:, 0].reshape([-1, 1]))  # 下衣类型
        preds.append(x_2_out_1[:, 1].reshape([-1, 1]))  # 下衣长
        preds.append(x_3_out[:, 1].reshape([-1, 1]))  # 袖长
        preds.append(x_2_out_1[:, 2].reshape([-1, 1]))  # 头发
        preds.append(x_3_out[:, 2].reshape([-1, 1]))  # 帽子
        preds.append(x_4_out)  # 性别
        preds.append(x_2_out_2)  # 上衣颜色
        preds.append(x_1_out_2)  # 下衣颜色

        mix = []
        mix.append(x_1)
        mix.append(x_2)
        mix.append(x_3)
        mix.append(x_4)
        mix.append(x_5)
        out = torch.cat(mix, dim=1)
        out = self.mix_classifier(out)


        preds = torch.cat(preds, dim=1)
        return preds, out

if __name__ == '__main__':
    model = pretrain_model()
    print(model)
    x = torch.randn([1, 3, 288, 144])
    y1, y = model(x)
    print(y.shape)
