import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from safebench.util.torch_util import to_device, to_tensor


# class Encoder(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
#         self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
#         self.pool1 = nn.MaxPool2d(kernel_size=3, stride=3, padding=0)
#         self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
#         self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
#         self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
#
#         self.fc1 = nn.Linear(1024, 512)
#         self.fc2 = nn.Linear(512, 256)
#         self.fc3 = nn.Linear(256, 128)
#
#     def forward(self, raw_obs):
#         obs_img = to_tensor(raw_obs["img"]).unsqueeze(0).permute(0, 3, 1, 2)
#         x = self.conv1(obs_img)
#         # print("1:", x.shape)
#         x = self.conv2(x)
#         # print("2:", x.shape)
#         x = self.pool1(x)
#         # print("4:", x.shape)
#         x = self.conv3(x)
#         # print("3:", x.shape)
#         x = self.conv4(x)
#         # print("5:", x.shape)
#         x = self.pool2(x)
#         # print("8:", x.shape)
#         x = torch.flatten(x, 1)
#         # print("9:", x.shape)
#         x = F.relu(self.fc1(x))
#         # print("6:", x.shape)
#         x = F.relu(self.fc2(x))
#         # print("7:", x.shape)
#         x = F.relu(self.fc3(x))
#         print("8:", x.shape)
#         obs_img = F.log_softmax(x, dim=1)
#         # print("9:", obs_img.shape)
#         obs_img = obs_img.detach().cpu().numpy()
#         obs_states = np.array([raw_obs["states"]]).repeat(32, axis=1)
#         # print("9:", obs_states.shape)
#         obs_total = np.concatenate((obs_img, obs_states), axis=1)
#
#         return obs_total


class Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = models.resnet101(pretrained=True, progress=True)
        self.fc = nn.Linear(1000, 128)

    def forward(self, raw_obs):
        obs_img = to_tensor(raw_obs["img"]).unsqueeze(0).permute(0, 3, 1, 2)
        x = self.fc(self.model(obs_img))
        obs_img = F.log_softmax(x, dim=1)
        # print("9:", obs_img.shape)
        obs_img = obs_img.detach().cpu().numpy()
        obs_states = np.array([raw_obs["states"]]).repeat(32, axis=1)
        # print("9:", obs_states.shape)
        obs_total = np.concatenate((obs_img, obs_states), axis=1)

        return obs_total
