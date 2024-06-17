''' 
Date: 2023-01-31 22:23:17
LastEditTime: 2023-04-04 01:03:34
Description: 
    Copyright (c) 2022-2023 Safebench Team

    This work is licensed under the terms of the MIT license.
    For a copy, see <https://opensource.org/licenses/MIT>
'''

import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributions as D
from matplotlib import pyplot as plt

from safebench.util.od_util import CUDA
from safebench.agent.object_detection.utils.general import cv2


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))


class ObjectDetection(object):
    def __init__(self, config, logger) -> None:

        self.ego_action_dim = config['ego_action_dim']
        self.batch_size = config['batch_size']
        self.type = config['type']
        self.texture_dir = os.path.join(config['ROOT_DIR'], config['texture_dir'])
        self.raw_image = cv2.imread(self.texture_dir)
        self._preprocess_img()

        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

        self.annotator = None 

        self.patch = CUDA(torch.rand(1, 256, 256)-0.5)
        self.patch.requires_grad = True
        self.optimizer = torch.optim.Adam([self.patch], lr=1e-3, betas=(0.9, 0.999))  # adjust beta1 to momentum
        self._dist = D.Bernoulli(torch.sigmoid(self.patch), )
    
    def _preprocess_img(self):
        self.raw_image = cv2.cvtColor(self.raw_image, cv2.COLOR_BGR2RGB)
        self.raw_image = cv2.resize(self.raw_image, (1024, 1024), interpolation=cv2.INTER_AREA)
        self.raw_image = np.rot90(self.raw_image,k=1)
        self.raw_image = cv2.flip(self.raw_image,1)
        self.raw_image = np.array(self.raw_image) / 255.
        self.raw_image = np.expand_dims(self.raw_image.transpose(2, 0, 1), 0)
        assert self.raw_image.shape == (1, 3, 1024, 1024), "shape should match template"
    
    def get_init_action(self, obs=None, deterministic=False):
        if self.mode == 'train':
            eps = CUDA(self._dist.sample((self.batch_size, )))
            img = CUDA(self.add_patch(CUDA(self.raw_image), CUDA(eps)))
            self._eps = eps
        elif self.mode == 'eval': # TODO
            eps = CUDA(self._dist.sample((self.batch_size, )))
            img = CUDA(self.add_patch(CUDA(self.raw_image), CUDA(eps)))
        # self._init_actions = [{'attack': eps, 'image': img} for _ in range(len(obs))]
        self._img = img
        return [{'attack': eps, 'image': img} for _ in range(len(obs))], [{} in range(len(obs))]
    
    def get_action(self, obs, infos, deterministic=False):
        return [{'attack': [], 'image': self._img} for _ in range(len(obs))]
    
    def load_model(self, scenario_configs=None):
        pass
    
    def set_mode(self, mode):
        self.mode = mode

    def train(self, replay_buffer):
        batch = replay_buffer.sample(self.batch_size)
        loss = CUDA(torch.FloatTensor(batch['loss']))

        log_prob = CUDA(self._dist.log_prob(self._eps).sum(-1).sum(-1).sum(-1))
        loss_pg = (-loss * log_prob).mean()
        
        self.optimizer.zero_grad()
        loss_pg.backward()
        self.optimizer.step()                             

        print('Attack_agents: ', loss.detach().cpu().numpy())
        print('Attack_agents PG: ', loss_pg.detach().cpu().numpy())

        # Update the distribution
        self._dist = D.Bernoulli(torch.sigmoid(self.patch), )
    
    def add_patch(self, img, input_patch):
        patch_mask = CUDA(self.create_patch_mask(img, input_patch))
        with_patch = img * patch_mask
        return with_patch

    def create_patch_mask(self, in_features, input_patch):
        width = in_features.size(2)
        height = in_features.size(3)

        patch_mask = torch.ones([input_patch.shape[0], 1, width, height])
        patch_size = 256
        patch_x = 512
        patch_y = 512
        p_w = patch_size + patch_x
        p_h = patch_size + patch_y
        patch_mask[:, :, int(patch_x):int(p_w), int(patch_y):int(p_h)] = input_patch

        return patch_mask

    def save_model(self, e_i):
        pass

if __name__ == '__main__':
    agent = ObjectDetection({
        'ego_action_dim': 2, 
        'model_path': None, 
        'batch_size': 16, 
        'ROOT_DIR': '/home/wenhao/7_carla/from_github_lhh/SafeBench_v2'
    }, None)
    ret = agent.get_init_action()
    cv2.imwrite('demo.jpg', np.array(ret['image'].detach().cpu().numpy()*255, dtype=np.int)[0].transpose(1, 2, 0))
