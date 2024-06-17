import time
import cv2
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.io.image import read_image
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image

from safebench.agent.base_policy import BasePolicy
from safebench.util.od_util import names_coco_paper, CUDA, CPU
from safebench.agent.object_detection.references_coco.detection.engine import train_one_epoch, evaluate

# /home/wenhao/.cache/torch/hub/checkpoints/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth


class FasterRCNNAgent(BasePolicy):
    def __init__(self, config, logger, train_mode='none') -> None:

        self.ego_action_dim = config['ego_action_dim']
        self.model_path = config['model_path']
        self.type = config['type']
        self.batch_size = config['batch_size']
        self.load_episode = 0
        self.continue_episode = 0

        self.mode = 'train'
        self.imgsz = (1024, 1024)
        self.model = CUDA(fasterrcnn_resnet50_fpn(pretrained=True))
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        # self.conf_thres = 0.3

    def get_action(self, obs, infos, deterministic=False):
        # print(len(obs), len(obs[0]), type(obs), type(obs[0]))
        self.model.eval()
        t1 = time.time()
        n_envs = len(obs)
        pred_list = []
        for i in range(n_envs):
            image = obs[i]['img']
            # print(image.shape)
            image = cv2.resize(image, self.imgsz, interpolation=cv2.INTER_LINEAR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            image = CUDA(torch.from_numpy(image).float().permute(2, 0, 1))
            image /= 255.
            if len(image.shape) == 3:
                image = image[None]
            
            pred = self.model(image)[0]
            pred = self._transform_predictions(pred)
            
            
            pred_list.append(pred)
            
            # TODO: CUDA Memory Management
            torch.cuda.empty_cache()
        return [{'ego_action': np.array([0.2, 0.0]), 'od_result': pred_list[i], 'annotated_image': []} for i in range(n_envs)]
    

    def load_model(self):
        pass

    def set_mode(self, mode):
        self.mode = mode

    def train(self, replay_buffer):
        # print('start training')
        self.model.train()

        batch = replay_buffer.sample(self.batch_size)

        img = CUDA(torch.FloatTensor(batch['image']))
        label = torch.FloatTensor(batch['label']).squeeze(1)
        label = CUDA(torch.cat([self._batch_id, label], dim=1))

        pred = self.model.model(img.permute(0, 3, 1, 2))
        
        self.optimizer.zero_grad()
        loss, loss_items = self.compute_loss(pred, label)
        loss_items = loss_items.detach()
        loss.backward()
        self.optimizer.step()
        torch.cuda.empty_cache()

    def smart_optimizer(self, model, name='Adam', lr=0.001, momentum=0.9, decay=1e-5):
        # YOLOv5 3-param group optimizer: 0) weights with decay, 1) weights no decay, 2) biases no decay
        g = [], [], []  # optimizer parameter groups
        bn = tuple(v for k, v in nn.__dict__.items() if 'Norm' in k)  # normalization layers, i.e. BatchNorm2d()
        for v in model.modules():
            for p_name, p in v.named_parameters(recurse=0):
                if p_name == 'bias':  # bias (no decay)
                    g[2].append(p)
                elif p_name == 'weight' and isinstance(v, bn):  # weight (no decay)
                    g[1].append(p)
                else:
                    g[0].append(p)  # weight (with decay)

        if name == 'Adam':
            optimizer = torch.optim.Adam(g[2], lr=lr, betas=(momentum, 0.999))  # adjust beta1 to momentum
        elif name == 'AdamW':
            optimizer = torch.optim.AdamW(g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
        elif name == 'RMSProp':
            optimizer = torch.optim.RMSprop(g[2], lr=lr, momentum=momentum)
        elif name == 'SGD':
            optimizer = torch.optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)
        else:
            raise NotImplementedError(f'Optimizer {name} not implemented.')

        optimizer.add_param_group({'params': g[0], 'weight_decay': decay})  # add g0 with weight_decay
        optimizer.add_param_group({'params': g[1], 'weight_decay': 0.0})  # add g1 (BatchNorm2d weights)
        
        return optimizer

    
    def add_patch(self, img, input_patch):
        # img: [1,3,416,416]
        # patch_size = cfg.patch_size
        patch_mask = self.create_patch_mask(img, 512)

        img_mask = self.create_img_mask(img, patch_mask)

        patch_mask = Variable(patch_mask.cuda(), requires_grad=False)
        img_mask = Variable(img_mask.cuda(), requires_grad=False)

        with_patch = img * img_mask + input_patch * patch_mask
        
        return with_patch

    def create_img_mask(self, img, patch_mask):
        mask = torch.ones([3, img.size(2), img.size(3)])
        img_mask = mask - patch_mask

        return img_mask

    def create_patch_mask(self, in_features, patch_size):
        width = in_features.size(2)
        height = in_features.size(3)
        patch_mask = torch.zeros([3, width, height])
        patch_x = 256
        patch_y = 256
        p_w = patch_size + patch_x
        p_h = patch_size + patch_y
        patch_mask[:, int(patch_x):int(p_w), int(patch_y):int(p_h)]= 1

        return patch_mask

    def save_model(self, e_i):
        pass
        
    def _transform_predictions(self, pred):
        if len(pred["scores"]) == 0:
            pred = {"scores": torch.Tensor([-1]), "labels": torch.Tensor([-1]), "boxes": torch.Tensor([-1, -1, -1, -1])}
        else:
            # index = torch.where(pred["scores"] > self.conf_thres)
            pred = {"scores": pred["scores"].detach().cpu(), 
                    "labels": [names_coco_paper[idx-1] for idx in pred["labels"].detach().cpu().numpy()],
                    "boxes": pred["boxes"].detach().cpu()}

        return pred