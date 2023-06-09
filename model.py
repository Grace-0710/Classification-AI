import torch
import torch.nn as nn

class ImageClassifier(nn.Module):

    def __init__(self,
                 input_size, # 입력 ex. mnist = 784
                 output_size): # 출력 ex. mnist = 10
        self.input_size = input_size
        self.output_size = output_size

        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_size, 500), #  ex. mnist = 784 -> 500
            nn.LeakyReLU(),
            nn.BatchNorm1d(500),
            nn.Linear(500, 400),    #  ex. 500 -> 400
            nn.LeakyReLU(),
            nn.BatchNorm1d(400),
            nn.Linear(400, 300),    #  ex. 400 -> 300
            nn.LeakyReLU(),
            nn.BatchNorm1d(300),
            nn.Linear(300, 200),
            nn.LeakyReLU(),
            nn.BatchNorm1d(200),
            nn.Linear(200, 100),
            nn.LeakyReLU(),
            nn.BatchNorm1d(100),
            nn.Linear(100, 50),
            nn.LeakyReLU(),
            nn.BatchNorm1d(50),
            nn.Linear(50, output_size),
            nn.LogSoftmax(dim=-1),  # 마지막 dimension에서만 수행하라
        )

    def forward(self, x):
        # |x| = (batch_size, input_size) ex. mmist : input_size = 784

        y = self.layers(x)
        # |y| = (batch_size, output_size)

        return y
