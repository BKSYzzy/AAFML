import torch
from torch import nn


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.a = nn.ParameterList([nn.Parameter(torch.zeros(3, 4))])
        b = [torch.ones(2, 3), torch.ones(2, 3)]
        for i in range(2):
            self.register_buffer('b%d' % i, b[i])

    def forward(self, input):
        return self.a[0]


class MAML(nn.Module):

    def __init__(self):
        super(MAML, self).__init__()

        self.net = Net()

    def forward(self, input):
        return self.net(input)


def main():
    device = torch.device('cuda')
    maml = MAML().to(device)
    print(maml.net.a)
    print(maml.net.b0)
import torch

if torch.cuda.is_available():
    print("CUDA is available. GPU count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("CUDA is not available.")

if __name__ == '__main__':
    main()
