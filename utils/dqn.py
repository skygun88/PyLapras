import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):
    def __init__(self, n_action) -> None:
        super().__init__()
        conv11 = nn.Conv1d(in_channels=2, out_channels=4, kernel_size=6, stride=3) # (2, 15) -> (4, 4)
        conv12 = nn.Conv1d(in_channels=2, out_channels=4, kernel_size=6, stride=3) # (2, 15) -> (4, 4)
        conv13 = nn.Conv1d(in_channels=2, out_channels=4, kernel_size=6, stride=3) # (2, 15) -> (4, 4)
        conv2 = nn.Conv1d(in_channels=4*3, out_channels=24, kernel_size=2, stride=1) # (12, 4) -> (24, 3)
        fc1 = nn.Linear(24*3, 256)
        fc2 = nn.Linear(256, n_action)
        
        torch.nn.init.xavier_uniform_(conv11.weight)
        torch.nn.init.xavier_uniform_(conv12.weight)
        torch.nn.init.xavier_uniform_(conv13.weight)
        torch.nn.init.xavier_uniform_(conv2.weight)
        torch.nn.init.xavier_uniform_(fc1.weight)
        torch.nn.init.xavier_uniform_(fc2.weight)

        self.conv11_module = nn.Sequential(
            conv11,
            nn.ReLU(),
        )

        self.conv12_module = nn.Sequential(
            conv12,
            nn.ReLU(),
        )

        self.conv13_module = nn.Sequential(
            conv13,
            nn.ReLU(),
        )

        self.conv2_module = nn.Sequential(
            conv2,
            nn.ReLU(),
        )


        self.fc_module = nn.Sequential(
            fc1,
            nn.ReLU(),
            fc2
        )
        # print(f'Available device: {device}')
        
        # self.conv11_module = self.conv11_module.to(device=device)
        # self.conv12_module = self.conv12_module.to(device=device)
        # self.conv13_module = self.conv13_module.to(device=device)
        # self.conv2_module = self.conv2_module.to(device=device)
        # self.fc_module = self.fc_module.to(device=device)

    def forward(self, x):
        x1, x2, x3, = x[:, 0:2, :], x[:, 2:4, :], x[:, 4:6, :]
        
        out1 = self.conv11_module(x1)
        out2 = self.conv12_module(x2)
        out3 = self.conv13_module(x3)
        # print(x1.shape, out2.shape)
        # out = torch.concat([out1, out2, out3], dim=1)
        out = torch.cat([out1, out2, out3], dim=1)

        out = self.conv2_module(out)
        dim = 1
        for d in out.size()[1:]: #16, 4, 4
            dim = dim * d
        out = out.view(-1, dim)
        out = self.fc_module(out)
        return out
