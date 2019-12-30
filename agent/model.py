import torch

import torch.nn as nn
import torch.nn.functional as F


class model(nn.Module):
    def __init__(self, in_channels, num_action):
        super(model, self).__init__()
        
        self.in_channels = in_channels
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.fc5 = nn.Linear(512, num_action)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        return self.fc5(x)

# Q(s,a) = V(s) + A(s,a)
class duelingDQN(nn.Module):
    def __init__(self, in_channels, num_action):
        super(duelingDQN, self).__init__()

        self.num_action = num_action
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        self.fc4_A = nn.Linear(7*7*64, 512)
        self.fc4_V = nn.Linear(7*7*64, 512)

        self.fc5_A = nn.Linear(512, num_action)
        self.fc5_V = nn.Linear(512, 1)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        
        a = F.relu(self.fc4_A(x))
        v = F.relu(self.fc4_V(x))

        a = self.fc5_A(a)
        v = self.fc5_V(v).expand(x.size(0), self.num_action)
        #正規化--->每個狀態的action加起來要為0，故會朝V去做更新
        '''
            v     [ 1  2  3  4]       ------> 由於下面有對a做限制，故更新的均是v
                +
            a     | 1  0 -1  3|
                  | 1  1  3 -1|       ------> 將原始的A做normalization後，列加起來會等於0，作為條件
                  |-2 -1 -2 -2|
                =
                  | 2  2  2  7|
            Q     | 2  3  6  3|
                  |-1  1  1  2|
        '''
        x = v + a -a.mean(1).unsqueeze(1).expand(x.size(0), self.num_action)
        return x