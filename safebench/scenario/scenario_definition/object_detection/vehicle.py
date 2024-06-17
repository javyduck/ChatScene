import carla
import os
import random
import numpy as np
import cv2

from safebench.scenario.scenario_manager.carla_data_provider import CarlaDataProvider
from safebench.scenario.scenario_definition.basic_scenario import BasicScenario
from safebench.util.od_util import *


class Detection_Vehicle(BasicScenario):
    """
        This scenario create car textures in the current scenarios.
    """

    def __init__(self, world, ego_vehicle, config, timeout=60):
        self._map = CarlaDataProvider.get_map()
        self.ego_id = config.ego_id
        self.ego_vehicle = ego_vehicle
        self.world = world
        self.timeout = timeout
        self.object_list=list(filter(lambda k: 'SM_Tesla' in k, world.get_names_of_all_objects()))
        self.image_path_list = [config.texture_dir]
        self.image_list = [cv2.imread(image_file) for image_file in self.image_path_list]
        self.image_list = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in self.image_list]
        resized = cv2.resize(self.image_list[0], (1024,1024), interpolation=cv2.INTER_AREA)
        resized = np.rot90(resized,k=1)
        self.resized = cv2.flip(resized,1)
        super(Detection_Vehicle, self).__init__("Detection_Vehicle", config, world)

    def initialize_actors(self):
        """
        Initialize some background autopilot vehicles.
        """
        # actors = CarlaDataProvider.request_new_batch_actors(
        #     'vehicle.*', 
        #     amount=self.number_of_vehicles,
        #     spawn_points=None, autopilot=True,
        #     random_location=True, 
        #     rolename='autopilot'
        # )
        # self.other_actors = actors

        # the trigger distance will always be 0, trigger at the beginning
        self.reference_actor = self.ego_vehicle 
    
    def create_behavior(self, scenario_init_action):
        if self.ego_id == 0:
            inputs = np.array(scenario_init_action['image'].detach().cpu().numpy()*255, dtype=np.int)[0].transpose(1, 2, 0)
            height = 1024
            texture = carla.TextureColor(height,height)
            # TODO: run in multi-processing?
            for x in range(height):
                for y in range(height):
                    r = int(inputs[x,y,0])
                    g = int(inputs[x,y,1])
                    b = int(inputs[x,y,2])
                    a = int(255)
                    # texture.set(x,height -0-y - 1, carla.Color(r,g,b,a))
                    texture.set(height-x-1, height-y-1, carla.Color(r,g,b,a))
                    # texture.set(x, y, carla.Color(r,g,b,a))
            for o_name in self.object_list:
                # print('initialize_actors: ', o_name)
                self.world.apply_color_texture_to_object(o_name, carla.MaterialParameter.Diffuse, texture)
        else:
            return
    
    def update_behavior(self, scenario_action):
        pass

    def check_stop_condition(self):
        return False


    def eval(self, bbox_pred, bbox_gt):
        '''
            bbox_pred: dictionary from detection modules (torch)
            bbox_gt: dictionary from carla envs (numpy)
        '''
        types = bbox_pred['labels']
        types_flag = -torch.ones_like(bbox_pred['scores'])

        if isinstance(types, torch.Tensor) or 'car' not in types:
            return {
                "iou": 0., 
                "logits": types_flag, 
                "gt": bbox_gt['car'], 
                "scores": bbox_pred['scores'],
                "pred": bbox_pred['boxes'], 
                "class": types, 
                "TP+FP": 0,
                "TP+FP+FN": len(bbox_pred['scores']),
            }
        
        index = torch.LongTensor(self._find_indices(types, 'car'))
        pred = bbox_pred['boxes'][index]
        types_flag[index] += 1
        match_ret = 0.
        
        if 'car' in bbox_gt.keys():
            box_true = bbox_gt['car']
            if len(box_true) > 0:
                b_true = torch.from_numpy(np.concatenate(box_true))
                ret = box_iou(pred, b_true)
                match_ret = ret.max().item()
                logits, idx_tp = torch.max(ret, dim=0)
                types_flag[index[idx_tp]] = logits
        
        return {
            "iou": match_ret, 
            "logits": types_flag, 
            "gt": bbox_gt['car'], 
            "scores": bbox_pred['scores'],
            "pred": bbox_pred['boxes'],
            "class": types,
        }
    
    def _find_indices(self, types, name):
        index = []
        for i in range(len(types)):
            if types[i] == name:
                index.append(i)
        return index
    
    def _try_spawn_random_walker_at(self, transform):
        """
            Try to spawn a walker at specific transform with random bluprint.
            Args:
                transform: the carla transform object.
            Returns:
                walker_actor: Bool indicating whether the spawn is successful.
        """
        walker_bp = random.choice(self.world.get_blueprint_library().filter('walker.*'))
        # set as not invencible
        if walker_bp.has_attribute('is_invincible'):
            walker_bp.set_attribute('is_invincible', 'false')
        walker_actor = self.world.try_spawn_actor(walker_bp, transform)

        if walker_actor is not None:
            walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
            walker_controller_actor = self.world.spawn_actor(walker_controller_bp, carla.Transform(), walker_actor)
            # start walker
            walker_controller_actor.start()
            # set walk to random point
            walker_controller_actor.go_to_location(self.world.get_random_location_from_navigation())
            # random max speed
            walker_controller_actor.set_max_speed(1 + random.random())  # max speed between 1 and 2 (default is 1.4 m/s)
        return walker_actor

    def _try_spawn_random_vehicle_at(self, transform, number_of_wheels=[4]):
        """
            Try to spawn a surrounding vehicle at specific transform with random bluprint.
            Args:
                transform: the carla transform object.
            Returns:
                vehicle: Bool indicating whether the spawn is successful.
        """
        blueprint = self._create_vehicle_bluepprint('vehicle.*', number_of_wheels=number_of_wheels)
        blueprint.set_attribute('role_name', 'autopilot')
        vehicle = self.world.try_spawn_actor(blueprint, transform)
        if vehicle is not None:
            vehicle.set_autopilot()
        return vehicle

    def _create_vehicle_bluepprint(self, actor_filter, color=None, number_of_wheels=[4]):
        """
            Create the blueprint for a specific actor type.
            Args:
                actor_filter: a string indicating the actor type, e.g, 'vehicle.lincoln*'.
            Returns:
                bp: the blueprint object of carla.
        """
        blueprints = self.world.get_blueprint_library().filter(actor_filter)
        blueprint_library = []
        for nw in number_of_wheels:
            blueprint_library = blueprint_library + [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == nw]
        bp = random.choice(blueprint_library)
        if bp.has_attribute('color'):
            if not color:
                color = random.choice(bp.get_attribute('color').recommended_values)
            bp.set_attribute('color', color)
        return bp