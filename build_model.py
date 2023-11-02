import torch
import torch.nn as nn

class BuildModel(nn.Module):
    def __init__(self):
        super(BuildModel, self).__init__()

        self.feature = nn.Sequential(

            ####### First Phase##############
            #1st two convolution
            nn.Conv2d(in_channels=1,out_channels= 10, kernel_size=3),
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3),
            # 1st pooling
            nn.BatchNorm2d(10),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(.5),



            ####### Second Phase#########
            # 2nd two convolution
            nn.Conv2d(in_channels=10, out_channels=16, kernel_size=3),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.Dropout2d(.3),

        )
        ####### Third Phase#########
        self.classifier = nn.Sequential(
            # 5 FC layers
            nn.Linear(in_features=16*8*8,out_features= 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(.2),
            nn.Linear(in_features=128, out_features=64),
            nn.Linear(in_features=64, out_features=32),
            nn.Linear(in_features=32, out_features=16),
            nn.Linear(in_features=16, out_features=10)
        )


    def forward(self, x):
        # print(x.shape)
        first = self.feature(x)
        #print(first.shape)

        second = first.view(first.size(0), -1)
        # print(second.shape)

        f = self.classifier(second)
        return f

