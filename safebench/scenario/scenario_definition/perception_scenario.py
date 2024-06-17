''' 
Date: 2023-01-31 22:23:17
LastEditTime: 2023-04-03 19:36:06
Description: 
    Copyright (c) 2022-2023 Safebench Team

    This work is licensed under the terms of the MIT license.
    For a copy, see <https://opensource.org/licenses/MIT>
'''

import os
import traceback

import carla
import numpy as np
import cv2

from safebench.util.run_util import class_from_path
from safebench.scenario.scenario_manager.scenario_config import PerceptionScenarioConfig
from safebench.scenario.scenario_manager.timer import GameTime
from safebench.scenario.scenario_definition.atomic_criteria import Status
from safebench.scenario.scenario_definition.route_scenario import RouteScenario
from safebench.scenario.tools.scenario_utils import convert_json_to_transform
from safebench.util.od_util import *


class PerceptionScenario(RouteScenario):
    """
        This class creates scenario where ego vehicle  is required to conduct pass-by testing.
    """

    def __init__(self, world, config, ego_id, ROOT_DIR, logger):
        self.world = world
        self.logger = logger
        self.config = config
        self.route = None
        self.ego_id = ego_id
        self.other_actors = []

        self.texture_dir = os.path.join(ROOT_DIR, config.texture_dir)
        self.sampled_scenarios_definitions = None
        self.vehicle_spawn_points = list(self.world.get_map().get_spawn_points())
        self.route, self.ego_vehicle, scenario_definitions = self._update_route_and_ego(timeout=self.timeout)

        self.list_scenarios = self._build_scenario_instances(scenario_definitions)
        self.n_step = 0
        self.object_dict = dict(
            stopsign=list(filter(lambda k: 'BP_Stop' in k, world.get_names_of_all_objects())),
            car=list(filter(lambda k: 'SM_Tesla' in k or 'SM_Jeep' in k, world.get_names_of_all_objects())),
            ad=list(filter(lambda k: 'AD' in k, world.get_names_of_all_objects()))
        )

        self.bbox_ground_truth = {}
        self.ground_truth_bbox = {}

        self.criteria = self._create_criteria()
        self._iou = 0.0
    
    def get_running_status(self, running_record):
        running_status = {
            "iou": self._iou, 
            'gt': self._gt,
            'scores': self._scores,
            'logits': self._logits,
            'pred': self._pred,
            'class': self._class,
            'scores': self._scores,
            'current_game_time': GameTime.get_time()
        }

        for criterion_name, criterion in self.criteria.items():
            running_status[criterion_name] = criterion.update()

        stop = False
        if running_status['collision'] == Status.FAILURE:
            stop = True
            self.logger.log('>> Scenario stops due to collision', color='yellow')
        if self.config.scenario_id != 0:  # only check when evaluating
            if running_status['route_complete'] == 100:
                stop = True
                self.logger.log('>> Scenario stops due to route completion', color='yellow')
        else:
            if len(running_record) >= self.max_running_step:  # stop at max step when training
                stop = True
                self.logger.log('>> Scenario stops due to max steps', color='yellow')

        for scenario in self.list_scenarios:
            if self.config.scenario_id != 0:  # only check when evaluating
                if running_status['driven_distance'] >= scenario.ego_max_driven_distance:
                    stop = True
                    self.logger.log('>> Scenario stops due to max driven distance', color='yellow')
                    break
            if running_status['current_game_time'] >= scenario.timeout:
                stop = True
                self.logger.log('>> Scenario stops due to timeout', color='yellow') 
                break

        return running_status, stop

    def _build_scenario_instances(self, scenario_definitions):
        """
            Based on the parsed route and possible scenarios, build all the scenario classes.
        """
        scenario_instance_list = []
        for scenario_number, definition in enumerate(scenario_definitions):
            # get the class of the scenario
            scenario_path = [
                'safebench.scenario.scenario_definition',
                self.config.scenario_folder,
                definition['name'],
            ]
            scenario_class = class_from_path('.'.join(scenario_path))

            # create the other actors that are going to appear
            if definition['other_actors'] is not None:
                list_of_actor_conf_instances = self._get_actors_instances(definition['other_actors'])
            else:
                list_of_actor_conf_instances = []

            # create an actor configuration for the ego-vehicle trigger position
            egoactor_trigger_position = convert_json_to_transform(definition['trigger_position'])
            perception_config = PerceptionScenarioConfig()
            perception_config.other_actors = list_of_actor_conf_instances
            perception_config.trigger_points = [egoactor_trigger_position]
            perception_config.parameters = self.config.parameters
            perception_config.num_scenario = self.config.num_scenario
            perception_config.texture_dir = self.texture_dir
            perception_config.ego_id = self.ego_id
            if self.config.weather is not None:
                perception_config.weather = self.config.weather

            try:
                scenario_instance = scenario_class(self.world, self.ego_vehicle, perception_config, timeout=self.timeout)
            except Exception as e:   
                traceback.print_exc()
                print("Skipping scenario '{}' due to setup error: {}".format(definition['name'], e))
                continue

            scenario_instance_list.append(scenario_instance)
        return scenario_instance_list

    def get_bbox(self, world_2_camera, image_w, image_h, fov): 
        def get_image_point(loc, K, w2c):
            # Calculate 2D projection of 3D coordinate

            # Format the input coordinate (loc is a carla.Position object)
            point = np.array([loc.x, loc.y, loc.z, 1])
            # transform to camera coordinates
            point_camera = np.dot(w2c, point)

            # New we must change from UE4's coordinate system to an "standard"
            # (x, y ,z) -> (y, -z, x)
            # and we remove the fourth componebonent also
            point_camera = [point_camera[1], -point_camera[2], point_camera[0]]

            # now project 3D->2D using the camera matrix
            point_img = np.dot(K, point_camera)
            # normalize
            point_img[0] /= point_img[2]
            point_img[1] /= point_img[2]
            return point_img[0:2]

        self.ground_truth_bbox = {}
        self.K = build_projection_matrix(image_w, image_h, fov)
        self.bbox_ground_truth['stopsign'] = self.world.get_level_bbs(carla.CityObjectLabel.TrafficSigns)
        self.bbox_ground_truth['car'] = self.world.get_level_bbs(carla.CityObjectLabel.Vehicles)
        self.bbox_ground_truth['person'] = self.world.get_level_bbs(carla.CityObjectLabel.Pedestrians)

        label_verts = list(range(8))

        for key, bounding_box_set in self.bbox_ground_truth.items():
            self.ground_truth_bbox.setdefault(key, [])
            for bbox in bounding_box_set:
                if bbox.location.distance(self.ego_vehicle.get_transform().location) < 50:
                    forward_vec = self.ego_vehicle.get_transform().get_forward_vector()
                    ray = bbox.location - self.ego_vehicle.get_transform().location
                    if forward_vec.dot(ray) > 1.0:
                        verts = [v for v in bbox.get_world_vertices(carla.Transform())]
                        # get the contour of the cube in 2D
                        box_cnt = []
                        for v in label_verts:
                            p = get_image_point(verts[v], self.K, world_2_camera)
                            box_cnt.append(np.array(p, dtype=np.int32))
                        box_cnt = np.expand_dims(np.array(box_cnt), axis=1)
                        # get bounding rectangle of the cube as labels
                        x, y, w, h = cv2.boundingRect(box_cnt)
                        self.ground_truth_bbox[key].append(np.array([[x, y, x+w, y+h]]))

    def eval(self, bbox_pred, bbox_gt):
        scenario = self.list_scenarios[0]
        ret_dict = scenario.eval(bbox_pred, bbox_gt)
        return ret_dict

    def evaluate(self, ego_action, world_2_camera, image_w, image_h, fov, obs):
        bbox_pred = ego_action['od_result']
        self.get_bbox(world_2_camera, image_w, image_h, fov)
        bbox_label = self.ground_truth_bbox
        ret = self.eval(bbox_pred, bbox_label)
        self._iou = ret['iou']
        self._gt = ret['gt']
        self._scores = ret['scores']
        self._logits = ret['logits']
        self._pred = ret['pred']
        self._class = ret['class']
        self._scores = ret['scores']

    def update_info(self):
        return {
            # "bbox_label": self.ground_truth_bbox, 
            "iou_loss": 1-self._iou, 
            "iou": self._iou, 
        }