import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(100, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(True),
            nn.Linear(100, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(True),
            nn.Linear(100, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(True),
            nn.Linear(100, 100)
        )

    def forward(self, input):
        output = self.main(input)
        return output

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main1 = nn.Sequential(
            nn.Linear(100, 100),
            nn.BatchNorm1d(100),
            nn.LeakyReLU(inplace=True),
            nn.Linear(100, 100)
        )

        self.main2 = nn.Sequential(
            nn.BatchNorm1d(100),
            nn.LeakyReLU(inplace=True),
            nn.Linear(100, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )



    def forward(self, input, intermediate=False):
        inter = self.main1(input)

        if intermediate:
            return inter
            
        output = self.main2(inter)
        return output
