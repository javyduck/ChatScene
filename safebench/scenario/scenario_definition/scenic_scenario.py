import numpy as np
from safebench.scenario.scenario_manager.timer import GameTime
from safebench.scenario.scenario_definition.atomic_criteria import Status
from safebench.scenario.scenario_manager.carla_data_provider import CarlaDataProvider
from safebench.gym_carla.envs.route_planner import RoutePlanner
from safebench.scenario.tools.scenario_utils import convert_transform_to_location
from safebench.scenario.scenario_definition.scenic.dynamic_scenic import DynamicScenic as scenario_scenic
from safebench.scenario.tools.route_manipulation import interpolate_trajectory

SECONDS_GIVEN_PER_METERS = 1
from safebench.scenario.scenario_definition.atomic_criteria import (
    Status,
    CollisionTest,
    DrivenDistanceTest,
    AverageVelocityTest,
    OffRoadTest,
    KeepLaneTest,
    InRouteTest,
    RouteCompletionTest,
    RunningRedLightTest,
    RunningStopTest,
    ActorSpeedAboveThresholdTest
)


class ScenicScenario():
    """
        Implementation of a ScenicScenario, i.e., a scenario that is controlled by scenic
    """

    def __init__(self, world, config, ego_id, logger, max_running_step):
        self.world = world
        self.logger = logger
        self.config = config
        self.ego_id = ego_id
        self.max_running_step = max_running_step
        self.timeout = 60

        self.route, self.ego_vehicle = self._update_route_and_ego()
        self.other_actors = []
        self.list_scenarios = [scenario_scenic(world, self.ego_vehicle, self.config, timeout=self.timeout)]
        self.criteria = self._create_criteria()
                
    def _update_route_and_ego(self, timeout=None):
        ego_vehicle = self.world.scenic.simulation.ego.carlaActor
        actor = ego_vehicle
        CarlaDataProvider._carla_actor_pool[actor.id] = actor
        CarlaDataProvider.register_actor(actor)       
        
        self.adv_actors = []
        for other_actor in self.world.scenic.simulation.objects:
            if 'Adv' in str(other_actor.behavior):
                adv_actor = other_actor.carlaActor
                self.adv_actors.append(adv_actor)
                CarlaDataProvider._carla_actor_pool[actor.id] = adv_actor
                CarlaDataProvider.register_actor(adv_actor)      
        
        if len(self.config.trajectory) == 0:
            # coarse traj ##
            routeplanner = RoutePlanner(ego_vehicle, 200, [])

            _waypoint_buffer = []
            while len(_waypoint_buffer) < 50:
                pop = routeplanner._waypoints_queue.popleft()
                _waypoint_buffer.append(pop[0].transform.location)

            ### dense route planning ###
            route = interpolate_trajectory(self.world, _waypoint_buffer)
            index = 1
            prev_wp = route[0][0].location
            _accum_meters = 0
            while _accum_meters < 100:
                pop = route[index]
                wp = pop[0].location
                d = wp.distance(prev_wp)
                _accum_meters += d
                prev_wp = wp
                index += 1
            route = route[:index]
        else:
            route = interpolate_trajectory(self.world, self.config.trajectory)
            
        CarlaDataProvider.set_ego_vehicle_route(convert_transform_to_location(route))
        CarlaDataProvider.set_scenario_config(self.config)

        # Timeout of scenario in seconds
        self.timeout = self._estimate_route_timeout(route) if timeout is None else timeout
        return route, ego_vehicle

    def _estimate_route_timeout(self, route):
        route_length = 0.0  # in meters
        min_length = 100.0

        if len(route) == 1:
            return int(SECONDS_GIVEN_PER_METERS * min_length)

        prev_point = route[0][0]
        for current_point, _ in route[1:]:
            dist = current_point.location.distance(prev_point.location)
            route_length += dist
            prev_point = current_point
        return int(SECONDS_GIVEN_PER_METERS * route_length)

    def initialize_actors(self):
        """
            Set other_actors to the superset of all scenario actors
        """
        pass

    def get_running_status(self, running_record):
        running_status = {
            'ego_velocity': CarlaDataProvider.get_velocity(self.ego_vehicle),
            'ego_acceleration_x': self.ego_vehicle.get_acceleration().x,
            'ego_acceleration_y': self.ego_vehicle.get_acceleration().y,
            'ego_acceleration_z': self.ego_vehicle.get_acceleration().z,
            'ego_x': CarlaDataProvider.get_transform(self.ego_vehicle).location.x,
            'ego_y': CarlaDataProvider.get_transform(self.ego_vehicle).location.y,
            'ego_z': CarlaDataProvider.get_transform(self.ego_vehicle).location.z,
            'ego_roll': CarlaDataProvider.get_transform(self.ego_vehicle).rotation.roll,
            'ego_pitch': CarlaDataProvider.get_transform(self.ego_vehicle).rotation.pitch,
            'ego_yaw': CarlaDataProvider.get_transform(self.ego_vehicle).rotation.yaw,
            'current_game_time': GameTime.get_time()
        }
        adv_actor_status = {}
        for i, adv_actor in enumerate(self.adv_actors):
            adv_actor_status[f'adv_agent_{i}'] = {
            'velocity': CarlaDataProvider.get_velocity(adv_actor),
            'acceleration_x': adv_actor.get_acceleration().x,
            'acceleration_y': adv_actor.get_acceleration().y,
            'acceleration_z': adv_actor.get_acceleration().z,
            'x': CarlaDataProvider.get_transform(adv_actor).location.x,
            'y': CarlaDataProvider.get_transform(adv_actor).location.y,
            'z': CarlaDataProvider.get_transform(adv_actor).location.z,
            'roll': CarlaDataProvider.get_transform(adv_actor).rotation.roll,
            'pitch': CarlaDataProvider.get_transform(adv_actor).rotation.pitch,
            'yaw': CarlaDataProvider.get_transform(adv_actor).rotation.yaw,
        }
        running_status.update(adv_actor_status)
            
        for criterion_name, criterion in self.criteria.items():
            running_status[criterion_name] = criterion.update()

        stop = False
        # collision with other objects
        if running_status['collision'] == Status.FAILURE:
            stop = True
            self.logger.log('>> Scenario stops due to collision', color='yellow')

        # out of the road detection
        if running_status['off_road'] == Status.FAILURE:
            stop = True
            self.logger.log('>> Scenario stops due to off road', color='yellow')

        # only check when evaluating
        if self.config.scenario_id != 0:  
            # route completed
            if running_status['route_complete'] == 100:
                stop = True
                self.logger.log('>> Scenario stops due to route completion', color='yellow')

        # stop at max step
        if len(running_record) >= self.max_running_step: 
            stop = True
            self.logger.log('>> Scenario stops due to max steps', color='yellow')

        for scenario in self.list_scenarios:
            # only check when evaluating
            if self.config.scenario_id != 0:  
                if running_status['driven_distance'] >= scenario.ego_max_driven_distance:
                    stop = True
                    self.logger.log('>> Scenario stops due to max driven distance', color='yellow')
                    break
            if running_status['current_game_time'] >= scenario.timeout:
                stop = True
                self.logger.log('>> Scenario stops due to timeout', color='yellow') 
                break
            if scenario.check_scenic_terminate():
                self.logger.log('>> Scenario stops due to scenic termination', color='yellow') 
                stop = True
                break
        return running_status, stop

    def _create_criteria(self):
        criteria = {}
        route = convert_transform_to_location(self.route)

        criteria['driven_distance'] = DrivenDistanceTest(actor=self.ego_vehicle, distance_success=1e4, distance_acceptable=1e4, optional=True)
        criteria['average_velocity'] = AverageVelocityTest(actor=self.ego_vehicle, avg_velocity_success=1e4, avg_velocity_acceptable=1e4, optional=True)
        criteria['lane_invasion'] = KeepLaneTest(actor=self.ego_vehicle, optional=True)
        criteria['off_road'] = OffRoadTest(actor=self.ego_vehicle, optional=True)
        criteria['collision'] = CollisionTest(actor=self.ego_vehicle, terminate_on_failure=True)
        criteria['run_red_light'] = RunningRedLightTest(actor=self.ego_vehicle)
        criteria['run_stop'] = RunningStopTest(actor=self.ego_vehicle)
        if self.config.scenario_id != 0:  # only check when evaluating
            criteria['distance_to_route'] = InRouteTest(self.ego_vehicle, route=route, offroad_max=30)
            criteria['route_complete'] = RouteCompletionTest(self.ego_vehicle, route=route)
        return criteria

    @staticmethod
    def _get_actor_state(actor):
        actor_trans = actor.get_transform()
        actor_x = actor_trans.location.x
        actor_y = actor_trans.location.y
        actor_yaw = actor_trans.rotation.yaw / 180 * np.pi
        yaw = np.array([np.cos(actor_yaw), np.sin(actor_yaw)])
        velocity = actor.get_velocity()
        acc = actor.get_acceleration()
        return [actor_x, actor_y, actor_yaw, yaw[0], yaw[1], velocity.x, velocity.y, acc.x, acc.y]

    def update_info(self):
        ego_state = self._get_actor_state(self.ego_vehicle)
        actor_info = [ego_state]
        for s_i in self.list_scenarios:
            for a_i in s_i.other_actors:
                actor_state = self._get_actor_state(a_i)
                actor_info.append(actor_state)

        actor_info = np.array(actor_info)
        # get the info of the ego vehicle and the other actors
        return {
            'actor_info': actor_info
        }

    def clean_up(self):
        # stop criterion and destroy sensors
        for _, criterion in self.criteria.items():
            criterion.terminate()
        # clean all actors
        self.world.scenic.endSimulation()
