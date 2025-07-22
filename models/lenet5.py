import torch
import torch.nn as nn


class Lenet5(nn.Module):

    def __init__(self, num_classes):
        # 调用父类的构造函数
        super(Lenet5, self).__init__()
        # 第一层卷积池化， Sequential内的函数顺序执行
        # 原文中激活函数都是用的sigmoid，这里使用更好的ReLu
        self.conv_pool1 = nn.Sequential(
            nn.Conv2d(in_channels=3,  # input (1, 28, 28) padding to(1,32,32)
                      # 这里的input和output的值都是针对一个样本来说的，而训练时是一次输入一个batch
                      out_channels=6,
                      kernel_size=(5, 5),
                      padding=0  #针对32×32
                      #padding=2 #针对minist 28×28
                      ),  # output(6, 28, 28)
            nn.ReLU(),  # 激活函数
            nn.MaxPool2d(2, stride=2)  # output(6, 14, 14)
        )

        self.conv_pool2 = nn.Sequential(
            nn.Conv2d(in_channels=6,
                      out_channels=16,
                      kernel_size=(5, 5)
                      ),  # output(16, 10, 10)
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)  # output(16, 5, 5)
        )

        # 全连接层
        self.fc = nn.Sequential(
            #nn.Linear(576, 2000),  #?????????
            nn.Linear(16*5*5, 120),
            nn.Linear(120, 84),
            nn.Linear(84, num_classes)
        )


    # 前向传播
    def forward(self, x):
        x = self.conv_pool1(x)
        x = self.conv_pool2(x)
        x = x.view(x.size(0), -1)  # resize to 2-dims(batch_size, 16*5*5) 展平成1维
        x = self.fc(x)
        return x


def lenet5(**kwargs) -> Lenet5:
    model = Lenet5(**kwargs)
    return model
