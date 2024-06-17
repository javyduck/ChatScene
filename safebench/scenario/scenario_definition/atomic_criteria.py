''' 
Date: 2023-01-31 22:23:17
LastEditTime: 2023-03-01 19:47:15
Description: 
    Copyright (c) 2022-2023 Safebench Team

    This file is modified from <https://github.com/carla-simulator/scenario_runner/blob/master/srunner/scenariomanager/scenarioatomics/atomic_criteria.py>
    Copyright (c) 2018-2020 Intel Corporation

    This work is licensed under the terms of the MIT license.
    For a copy, see <https://opensource.org/licenses/MIT>
'''

import weakref
import math

import numpy as np
from enum import Enum
import shapely
import carla

from safebench.scenario.scenario_manager.carla_data_provider import CarlaDataProvider
from safebench.scenario.scenario_manager.timer import GameTime
from safebench.scenario.scenario_manager.traffic_events import TrafficEvent, TrafficEventType


class Status(Enum):
    SUCCESS = 'SUCCESS'
    FAILURE = 'FAILURE'
    RUNNING = 'RUNNING'
    INVALID = 'INVALID'


class Criterion:
    """
        Base class for all criteria used to evaluate a scenario for success/failure

        Important parameters (PUBLIC):
        - name: Name of the criterion
        - expected_value_success:    Result in case of success (e.g. max_speed, zero collisions, ...)
        - expected_value_acceptable: Result that does not mean a failure, but is not good enough for a success
        - actual_value: Actual result after running the scenario
        - test_status: Used to access the result of the criterion
        - optional: Indicates if a criterion is optional (not used for overall analysis)
    """

    def __init__(
        self,
        name,
        actor,
        expected_value_success,
        expected_value_acceptable=None,
        optional=False,
        terminate_on_failure=False
    ):
        self._terminate_on_failure = terminate_on_failure

        self.name = name
        self.actor = actor
        self.test_status = "INIT"
        self.expected_value_success = expected_value_success
        self.expected_value_acceptable = expected_value_acceptable
        self.actual_value = 0
        self.optional = optional
        self.list_traffic_events = []

    def terminate(self):
        """
        Terminate the criterion. Can be extended by the user-derived class
        """
        if (self.test_status == "RUNNING") or (self.test_status == "INIT"):
            self.test_status = "SUCCESS"


class DrivenDistanceTest(Criterion):
    """
        This class contains an atomic test to check the driven distance

        Important parameters:
        - actor: CARLA actor to be used for this test
        - distance_success: If the actor's driven distance is more than this value (in meters), the test result is SUCCESS
        - distance_acceptable: If the actor's driven distance is more than this value (in meters), the test result is ACCEPTABLE
        - optional [optional]: If True, the result is not considered for an overall pass/fail result
    """

    def __init__(self, actor, distance_success, distance_acceptable=None, optional=False,name="CheckDrivenDistance"):
        super(DrivenDistanceTest, self).__init__(name, actor, distance_success, distance_acceptable, optional)
        self._last_location = None
        self._last_location = CarlaDataProvider.get_location(self.actor)

    def update(self):
        if self.actor is None:
            return self.actual_value

        location = CarlaDataProvider.get_location(self.actor)
        if location is None:
            return self.actual_value

        if self._last_location is None:
            self._last_location = location
            return self.actual_value

        self.actual_value += location.distance(self._last_location)
        self._last_location = location

        if self.actual_value > self.expected_value_success:
            self.test_status = "SUCCESS"
        elif (self.expected_value_acceptable is not None and self.actual_value > self.expected_value_acceptable):
            self.test_status = "ACCEPTABLE"
        else:
            self.test_status = "RUNNING"

        return self.actual_value

    def terminate(self):
        if self.test_status != "SUCCESS":
            self.test_status = "FAILURE"
        self.actual_value = round(self.actual_value, 2)
        super(DrivenDistanceTest, self).terminate()


class AverageVelocityTest(Criterion):
    """
        This class contains an atomic test for average velocity.

        Important parameters:
        - actor: CARLA actor to be used for this test
        - avg_velocity_success: If the actor's average velocity is more than this value (in m/s), the test result is SUCCESS
        - avg_velocity_acceptable: If the actor's average velocity is more than this value (in m/s), the test result is ACCEPTABLE
        - optional [optional]: If True, the result is not considered for an overall pass/fail result
    """

    def __init__(self, actor, avg_velocity_success, avg_velocity_acceptable=None, optional=False, name="CheckAverageVelocity"):
        """
            Setup actor and average velovity expected
        """
        super(AverageVelocityTest, self).__init__(name, actor, avg_velocity_success, avg_velocity_acceptable, optional)
        self._last_location = None
        self._distance = 0.0
        self._last_location = CarlaDataProvider.get_location(self.actor)

    def update(self):
        """
            Check velocity
        """
        if self.actor is None:
            return self.actual_value

        location = CarlaDataProvider.get_location(self.actor)
        if location is None:
            return self.actual_value

        if self._last_location is None:
            self._last_location = location
            return self.actual_value

        self._distance += location.distance(self._last_location)
        self._last_location = location

        elapsed_time = GameTime.get_time()
        if elapsed_time > 0.0:
            self.actual_value = self._distance / elapsed_time

        if self.actual_value > self.expected_value_success:
            self.test_status = "SUCCESS"
        elif (self.expected_value_acceptable is not None and self.actual_value > self.expected_value_acceptable):
            self.test_status = "ACCEPTABLE"
        else:
            self.test_status = "RUNNING"
        return self.actual_value

    def terminate(self):
        """
            Set final status
        """
        if self.test_status == "RUNNING":
            self.test_status = "FAILURE"
        super(AverageVelocityTest, self).terminate()


class CollisionTest(Criterion):
    """
        This class contains an atomic test for collisions.

        Args:
        - actor (carla.Actor): CARLA actor to be used for this test
        - other_actor (carla.Actor): only collisions with this actor will be registered
        - other_actor_type (str): only collisions with actors including this type_id will count.
            Additionally, the "miscellaneous" tag can also be used to include all static objects in the scene
        - terminate_on_failure [optional]: If True, the complete scenario will terminate upon failure of this test
        - optional [optional]: If True, the result is not considered for an overall pass/fail result
    """

    MIN_AREA_OF_COLLISION = 3       # If closer than this distance, the collision is ignored
    MAX_AREA_OF_COLLISION = 5       # If further than this distance, the area is forgotten
    MAX_ID_TIME = 5                 # Amount of time the last collision if is remembered

    def __init__(self, actor, other_actor=None, other_actor_type=None, optional=False, name="CollisionTest", terminate_on_failure=False):
        """
            Construction with sensor setup
        """
        super(CollisionTest, self).__init__(name, actor, 0, None, optional, terminate_on_failure)

        world = self.actor.get_world()
        blueprint = world.get_blueprint_library().find('sensor.other.collision')
        self._collision_sensor = world.spawn_actor(blueprint, carla.Transform(), attach_to=self.actor)
        self._collision_sensor.listen(lambda event: self._count_collisions(weakref.ref(self), event))

        self.other_actor = other_actor
        self.other_actor_type = other_actor_type
        self.registered_collisions = []
        self.last_id = None
        self.collision_time = None

    def update(self):
        """
            Check collision count
        """
        new_status = Status.RUNNING

        if self._terminate_on_failure and (self.test_status == "FAILURE"):
            new_status = Status.FAILURE

        actor_location = CarlaDataProvider.get_location(self.actor)
        new_registered_collisions = []

        # Loops through all the previous registered collisions
        for collision_location in self.registered_collisions:
            # Get the distance to the collision point
            distance_vector = actor_location - collision_location
            distance = math.sqrt(math.pow(distance_vector.x, 2) + math.pow(distance_vector.y, 2))

            # If far away from a previous collision, forget it
            if distance <= self.MAX_AREA_OF_COLLISION:
                new_registered_collisions.append(collision_location)

        self.registered_collisions = new_registered_collisions
        if self.last_id and GameTime.get_time() - self.collision_time > self.MAX_ID_TIME:
            self.last_id = None

        return new_status

    def terminate(self):
        """
            Cleanup sensor
        """
        if self._collision_sensor is not None and self._collision_sensor.is_alive:
            self._collision_sensor.destroy()
        self._collision_sensor = None
        super(CollisionTest, self).terminate()

    @staticmethod
    def _count_collisions(weak_self, event):     # pylint: disable=too-many-return-statements
        """
            Callback to update collision count
        """
        self = weak_self()
        if not self:
            return

        actor_location = CarlaDataProvider.get_location(self.actor)

        # Ignore the current one if it is the same id as before
        if self.last_id == event.other_actor.id:
            return

        # Filter to only a specific actor
        if self.other_actor and self.other_actor.id != event.other_actor.id:
            return

        # Filter to only a specific type
        if self.other_actor_type:
            if self.other_actor_type == "miscellaneous":
                if "traffic" not in event.other_actor.type_id and "static" not in event.other_actor.type_id:
                    return
            else:
                if self.other_actor_type not in event.other_actor.type_id:
                    return

        # Ignore it if its too close to a previous collision (avoid micro collisions)
        for collision_location in self.registered_collisions:
            distance_vector = actor_location - collision_location
            distance = math.sqrt(math.pow(distance_vector.x, 2) + math.pow(distance_vector.y, 2))
            if distance <= self.MIN_AREA_OF_COLLISION:
                return

        if ('static' in event.other_actor.type_id or 'traffic' in event.other_actor.type_id) and 'sidewalk' not in event.other_actor.type_id:
            actor_type = TrafficEventType.COLLISION_STATIC
        elif 'vehicle' in event.other_actor.type_id:
            actor_type = TrafficEventType.COLLISION_VEHICLE
        elif 'walker' in event.other_actor.type_id:
            actor_type = TrafficEventType.COLLISION_PEDESTRIAN
        else:
            return

        collision_event = TrafficEvent(event_type=actor_type)
        collision_event.set_dict({
            'type': event.other_actor.type_id,
            'id': event.other_actor.id,
            'x': actor_location.x,
            'y': actor_location.y,
            'z': actor_location.z
        })
        collision_event.set_message(
            "Agent collided against object with type={} and id={} at (x={}, y={}, z={})".format(
                event.other_actor.type_id,
                event.other_actor.id,
                round(actor_location.x, 3),
                round(actor_location.y, 3),
                round(actor_location.z, 3)
            )
        )

        self.test_status = "FAILURE"
        self.actual_value += 1
        self.collision_time = GameTime.get_time()

        self.registered_collisions.append(actor_location)
        self.list_traffic_events.append(collision_event)

        # Number 0: static objects -> ignore it
        if event.other_actor.id != 0:
            self.last_id = event.other_actor.id


class ActorSpeedAboveThresholdTest(Criterion):
    """
        This test will fail if the actor has had its linear velocity lower than a specific value for a specific amount of time

        Important parameters:
        - actor: CARLA actor to be used for this test
        - speed_threshold: speed required
        - below_threshold_max_time: Maximum time (in seconds) the actor can remain under the speed threshold
        - terminate_on_failure [optional]: If True, the complete scenario will terminate upon failure of this test
    """

    def __init__(self, actor, speed_threshold, below_threshold_max_time, name="ActorSpeedAboveThresholdTest", terminate_on_failure=False):
        """
            Class constructor
        """
        super(ActorSpeedAboveThresholdTest, self).__init__(name, actor, 0, terminate_on_failure=terminate_on_failure)
        self._actor = actor
        self._speed_threshold = speed_threshold
        self._below_threshold_max_time = below_threshold_max_time
        self._time_last_valid_state = None

    def update(self):
        """
            Check if the actor speed is above the speed_threshold
        """
        new_status = Status.RUNNING

        linear_speed = CarlaDataProvider.get_velocity(self._actor)
        if linear_speed is not None:
            if linear_speed < self._speed_threshold and self._time_last_valid_state:
                if (GameTime.get_time() - self._time_last_valid_state) > self._below_threshold_max_time:
                    # Game over. The actor has been "blocked" for too long
                    self.test_status = "FAILURE"

                    # record event
                    vehicle_location = CarlaDataProvider.get_location(self._actor)
                    blocked_event = TrafficEvent(event_type=TrafficEventType.VEHICLE_BLOCKED)
                    ActorSpeedAboveThresholdTest._set_event_message(blocked_event, vehicle_location)
                    ActorSpeedAboveThresholdTest._set_event_dict(blocked_event, vehicle_location)
                    self.list_traffic_events.append(blocked_event)
            else:
                self._time_last_valid_state = GameTime.get_time()

        if self._terminate_on_failure and (self.test_status == "FAILURE"):
            new_status = Status.FAILURE
        return new_status

    @staticmethod
    def _set_event_message(event, location):
        """
            Sets the message of the event
        """
        event.set_message('Agent got blocked at (x={}, y={}, z={})'.format(round(location.x, 3), round(location.y, 3), round(location.z, 3)))

    @staticmethod
    def _set_event_dict(event, location):
        """
            Sets the dictionary of the event
        """
        event.set_dict({'x': location.x, 'y': location.y, 'z': location.z})


class KeepLaneTest(Criterion):
    """
        This class contains an atomic test for keeping lane.

        Important parameters:
        - actor: CARLA actor to be used for this test
        - optional [optional]: If True, the result is not considered for an overall pass/fail result
    """

    def __init__(self, actor, optional=False, name="CheckKeepLane"):
        """
            Construction with sensor setup
        """
        super(KeepLaneTest, self).__init__(name, actor, 0, None, optional)

        world = self.actor.get_world()
        blueprint = world.get_blueprint_library().find('sensor.other.lane_invasion')
        self._lane_sensor = world.spawn_actor(blueprint, carla.Transform(), attach_to=self.actor)
        self._lane_sensor.listen(lambda event: self._count_lane_invasion(weakref.ref(self), event))

    def update(self):
        """
            Check lane invasion count
        """
        new_status = Status.RUNNING

        if self.actual_value > 0:
            self.test_status = "FAILURE"
        else:
            self.test_status = "SUCCESS"

        if self._terminate_on_failure and (self.test_status == "FAILURE"):
            new_status = Status.FAILURE
        return self.actual_value

    def terminate(self):
        """
            Cleanup sensor
        """
        if self._lane_sensor is not None:
            self._lane_sensor.destroy()
        self._lane_sensor = None
        super(KeepLaneTest, self).terminate()

    @staticmethod
    def _count_lane_invasion(weak_self, event):
        """
            Callback to update lane invasion count
        """
        self = weak_self()
        if not self:
            return
        self.actual_value += 1


class OffRoadTest(Criterion):
    """
        Atomic containing a test to detect when an actor deviates from the driving lanes. This atomic can
        fail when actor has spent a specific time outside driving lanes (defined by OpenDRIVE). Simplified
        version of OnSidewalkTest, and doesn't relly on waypoints with *Sidewalk* lane types

        Args:
            actor (carla.Actor): CARLA actor to be used for this test
            duration (float): Time spent at sidewalks before the atomic fails. If terminate_on_failure isn't active, this is ignored.
            optional (bool): If True, the result is not considered for an overall pass/fail result when using the output argument
            terminate_on_failure (bool): If True, the atomic will fail when the duration condition has been met.
    """

    def __init__(self, actor, duration=0, optional=False, terminate_on_failure=False, name="OffRoadTest"):
        """
            Setup of the variables
        """
        super(OffRoadTest, self).__init__(name, actor, 0, None, optional, terminate_on_failure)

        self._map = CarlaDataProvider.get_map()
        self._offroad = False

        self._duration = duration
        self._prev_time = None
        self._time_offroad = 0

    def update(self):
        """
            First, transforms the actor's current position to its corresponding waypoint. This is
            filtered to only use waypoints of type Driving or Parking. Depending on these results,
            the actor will be considered to be outside (or inside) driving lanes.

            returns:
                Status.FAILURE: when the actor has spent a given duration outside driving lanes
                Status.RUNNING: the rest of the time
        """
        new_status = Status.RUNNING

        current_location = CarlaDataProvider.get_location(self.actor)

        # Get the waypoint at the current location to see if the actor is offroad
        drive_waypoint = self._map.get_waypoint(current_location, project_to_road=False)
        park_waypoint = self._map.get_waypoint(current_location, project_to_road=False, lane_type=carla.LaneType.Parking)
        if drive_waypoint or park_waypoint:
            self._offroad = False
        else:
            self._offroad = True

        # Counts the time offroad
        if self._offroad:
            if self._prev_time is None:
                self._prev_time = GameTime.get_time()
            else:
                curr_time = GameTime.get_time()
                self._time_offroad += curr_time - self._prev_time
                self._prev_time = curr_time
        else:
            self._prev_time = None

        if self._time_offroad > self._duration:
            self.test_status = "FAILURE"

        if self._terminate_on_failure and self.test_status == "FAILURE":
            new_status = Status.FAILURE

        return self._offroad


class InRouteTest(Criterion):
    """
        The test is a success if the actor is never outside route. The actor can go outside of the route
        but only for a certain amount of distance

        Important parameters:
        - actor: CARLA actor to be used for this test
        - route: Route to be checked
        - offroad_max: Maximum distance (in meters) the actor can deviate from the route
        - offroad_min: Maximum safe distance (in meters). Might eventually cause failure
        - terminate_on_failure [optional]: If True, the complete scenario will terminate upon failure of this test
    """

    MAX_ROUTE_PERCENTAGE = 30  # %
    WINDOWS_SIZE = 5  # Amount of additional waypoints checked

    def __init__(self, actor, route, offroad_min=-1, offroad_max=30, name="InRouteTest", terminate_on_failure=False):
        super(InRouteTest, self).__init__(name, actor, 0, terminate_on_failure=terminate_on_failure)
        self._actor = actor
        self._route = route
        self._offroad_max = offroad_max
        # Unless specified, halve of the max value
        if offroad_min == -1:
            self._offroad_min = self._offroad_max / 2
        else:
            self._offroad_min = self._offroad_min

        self._world = CarlaDataProvider.get_world()
        self._waypoints, _ = zip(*self._route)
        self._route_length = len(self._route)
        self._current_index = 0
        self._out_route_distance = 0
        self._in_safe_route = True

        self._accum_meters = []
        prev_wp = self._waypoints[0]
        for i, wp in enumerate(self._waypoints):
            d = wp.distance(prev_wp)
            if i > 0:
                accum = self._accum_meters[i - 1]
            else:
                accum = 0

            self._accum_meters.append(d + accum)
            prev_wp = wp

    def update(self):
        """
            Check if the actor location is within trigger region
        """
        new_status = Status.RUNNING

        location = CarlaDataProvider.get_location(self._actor)
        if location is None:
            return 0

        if self._terminate_on_failure and (self.test_status == "FAILURE"):
            new_status = Status.FAILURE

        off_route = True
        shortest_distance = float('inf')
        closest_index = -1

        # Get the closest distance
        for index in range(self._current_index, self._route_length):
            ref_waypoint = self._waypoints[index]
            distance = math.sqrt(((location.x - ref_waypoint.x) ** 2) + ((location.y - ref_waypoint.y) ** 2))
            if distance <= shortest_distance:
                closest_index = index
                shortest_distance = distance

        if closest_index == -1 or shortest_distance == float('inf'):
            self.shortest_distance = 0
            return self.shortest_distance
        else:
            self.shortest_distance = shortest_distance

        # Check if the actor is out of route
        if shortest_distance < self._offroad_max:
            off_route = False
            self._in_safe_route = bool(shortest_distance < self._offroad_min)

        # If actor advanced a step, record the distance
        if self._current_index != closest_index:
            new_dist = self._accum_meters[closest_index] - self._accum_meters[self._current_index]

            # If too far from the route, add it and check if its value
            if not self._in_safe_route:
                self._out_route_distance += new_dist
                out_route_percentage = 100 * self._out_route_distance / self._accum_meters[-1]
                if out_route_percentage > self.MAX_ROUTE_PERCENTAGE:
                    off_route = True

            self._current_index = closest_index

        if off_route:
            route_deviation_event = TrafficEvent(event_type=TrafficEventType.ROUTE_DEVIATION)
            route_deviation_event.set_message(
                "Agent deviated from the route at (x={}, y={}, z={})".format(
                    round(location.x, 3),
                    round(location.y, 3),
                    round(location.z, 3)
                )
            )
            route_deviation_event.set_dict({'x': location.x, 'y': location.y, 'z': location.z})
            self.list_traffic_events.append(route_deviation_event)
            self.test_status = "FAILURE"
            self.actual_value += 1
            new_status = Status.FAILURE

        return self.shortest_distance


class RouteCompletionTest(Criterion):
    """
        Check at which stage of the route is the actor at each tick

        Important parameters:
        - actor: CARLA actor to be used for this test
        - route: Route to be checked
        - terminate_on_failure [optional]: If True, the complete scenario will terminate upon failure of this test
    """

    DISTANCE_THRESHOLD = 10.0  # meters
    WINDOWS_SIZE = 2
    
    def __init__(self, actor, route, name="RouteCompletionTest", terminate_on_failure=False):
        super(RouteCompletionTest, self).__init__(name, actor, 100, terminate_on_failure=terminate_on_failure)
        self._actor = actor
        self._route = route
        self._map = CarlaDataProvider.get_map()

        self._wsize = self.WINDOWS_SIZE
        self._current_index = 0
        self._route_length = len(self._route)
        self._waypoints, _ = zip(*self._route)
        self.target = self._waypoints[-1]

        self._accum_meters = []
        prev_wp = self._waypoints[0]
        for i, wp in enumerate(self._waypoints):
            d = wp.distance(prev_wp)
            if i > 0:
                accum = self._accum_meters[i - 1]
            else:
                accum = 0

            self._accum_meters.append(d + accum)
            prev_wp = wp

        self._traffic_event = TrafficEvent(event_type=TrafficEventType.ROUTE_COMPLETION)
        self.list_traffic_events.append(self._traffic_event)
        self._percentage_route_completed = 0.0

    def update(self):
        """
            Check if the actor location is within trigger region
        """
        new_status = Status.RUNNING

        location = CarlaDataProvider.get_location(self._actor)
        if location is None:
            return self._percentage_route_completed

        if self._terminate_on_failure and (self.test_status == "FAILURE"):
            new_status = Status.FAILURE
        elif self.test_status == "RUNNING" or self.test_status == "INIT":
            for index in range(self._current_index, min(self._current_index + self._wsize + 1, self._route_length)):
                # Get the dot product to know if it has passed this location
                ref_waypoint = self._waypoints[index]
                wp = self._map.get_waypoint(ref_waypoint)
                wp_dir = wp.transform.get_forward_vector()          # Waypoint's forward vector
                wp_veh = location - ref_waypoint                    # vector waypoint - vehicle
                dot_ve_wp = wp_veh.x * wp_dir.x + wp_veh.y * wp_dir.y + wp_veh.z * wp_dir.z

                if dot_ve_wp > 0:
                    # good! segment completed!
                    self._current_index = index
                    self._percentage_route_completed = 100.0 * float(self._accum_meters[self._current_index]) / float(self._accum_meters[-1])
                    self._traffic_event.set_dict({'route_completed': self._percentage_route_completed})
                    self._traffic_event.set_message("Agent has completed > {:.2f}% of the route".format(self._percentage_route_completed))

            if self._percentage_route_completed > 99.0 and location.distance(self.target) < self.DISTANCE_THRESHOLD:
                route_completion_event = TrafficEvent(event_type=TrafficEventType.ROUTE_COMPLETED)
                route_completion_event.set_message("Destination was successfully reached")
                self.list_traffic_events.append(route_completion_event)
                self.test_status = "SUCCESS"
                self._percentage_route_completed = 100

        elif self.test_status == "SUCCESS":
            new_status = Status.SUCCESS

        return self._percentage_route_completed

    def terminate(self):
        """
            Set test status to failure if not successful and terminate
        """
        self.actual_value = round(self._percentage_route_completed, 2)

        if self.test_status == "INIT":
            self.test_status = "FAILURE"
        super(RouteCompletionTest, self).terminate()


class RunningRedLightTest(Criterion):
    """
        Check if an actor is running a red light

        Important parameters:
        - actor: CARLA actor to be used for this test
        - terminate_on_failure [optional]: If True, the complete scenario will terminate upon failure of this test
    """
    
    DISTANCE_LIGHT = 15  # m

    def __init__(self, actor, name="RunningRedLightTest", terminate_on_failure=False):
        """
        Init
        """
        super(RunningRedLightTest, self).__init__(name, actor, 0, terminate_on_failure=terminate_on_failure)
        self._actor = actor
        self._world = actor.get_world()
        self._map = CarlaDataProvider.get_map()
        self._list_traffic_lights = []
        self._last_red_light_id = None
        self.actual_value = 0
        self.debug = False

        all_actors = self._world.get_actors()
        for _actor in all_actors:
            if 'traffic_light' in _actor.type_id:
                center, waypoints = self.get_traffic_light_waypoints(_actor)
                self._list_traffic_lights.append((_actor, center, waypoints))

    def is_vehicle_crossing_line(self, seg1, seg2):
        """
            Check if vehicle crosses a line segment
        """
        line1 = shapely.geometry.LineString([(seg1[0].x, seg1[0].y), (seg1[1].x, seg1[1].y)])
        line2 = shapely.geometry.LineString([(seg2[0].x, seg2[0].y), (seg2[1].x, seg2[1].y)])
        inter = line1.intersection(line2)
        return not inter.is_empty

    def update(self):
        """
            Check if the actor is running a red light
        """
        new_status = Status.RUNNING

        transform = CarlaDataProvider.get_transform(self._actor)
        location = transform.location
        if location is None:
            return self.actual_value

        veh_extent = self._actor.bounding_box.extent.x

        tail_close_pt = self.rotate_point(carla.Vector3D(-0.8 * veh_extent, 0.0, location.z), transform.rotation.yaw)
        tail_close_pt = location + carla.Location(tail_close_pt)

        tail_far_pt = self.rotate_point(carla.Vector3D(-veh_extent - 1, 0.0, location.z), transform.rotation.yaw)
        tail_far_pt = location + carla.Location(tail_far_pt)

        for traffic_light, center, waypoints in self._list_traffic_lights:
            if self.debug:
                z = 2.1
                if traffic_light.state == carla.TrafficLightState.Red:
                    color = carla.Color(155, 0, 0)
                elif traffic_light.state == carla.TrafficLightState.Green:
                    color = carla.Color(0, 155, 0)
                else:
                    color = carla.Color(155, 155, 0)
                self._world.debug.draw_point(center + carla.Location(z=z), size=0.2, color=color, life_time=0.01)
                for wp in waypoints:
                    text = "{}.{}".format(wp.road_id, wp.lane_id)
                    self._world.debug.draw_string(wp.transform.location + carla.Location(x=1, z=z), text, color=color, life_time=0.01)
                    self._world.debug.draw_point(wp.transform.location + carla.Location(z=z), size=0.1, color=color, life_time=0.01)

            center_loc = carla.Location(center)
            if self._last_red_light_id and self._last_red_light_id == traffic_light.id:
                continue
            if center_loc.distance(location) > self.DISTANCE_LIGHT:
                continue
            if traffic_light.state != carla.TrafficLightState.Red:
                continue

            for wp in waypoints:
                tail_wp = self._map.get_waypoint(tail_far_pt)

                # Calculate the dot product (Might be unscaled, as only its sign is important)
                ve_dir = CarlaDataProvider.get_transform(self._actor).get_forward_vector()
                wp_dir = wp.transform.get_forward_vector()
                dot_ve_wp = ve_dir.x * wp_dir.x + ve_dir.y * wp_dir.y + ve_dir.z * wp_dir.z

                # Check the lane until all the "tail" has passed
                if tail_wp.road_id == wp.road_id and tail_wp.lane_id == wp.lane_id and dot_ve_wp > 0:
                    # This light is red and is affecting our lane
                    yaw_wp = wp.transform.rotation.yaw
                    lane_width = wp.lane_width
                    location_wp = wp.transform.location

                    lft_lane_wp = self.rotate_point(carla.Vector3D(0.4 * lane_width, 0.0, location_wp.z), yaw_wp + 90)
                    lft_lane_wp = location_wp + carla.Location(lft_lane_wp)
                    rgt_lane_wp = self.rotate_point(carla.Vector3D(0.4 * lane_width, 0.0, location_wp.z), yaw_wp - 90)
                    rgt_lane_wp = location_wp + carla.Location(rgt_lane_wp)

                    # Is the vehicle traversing the stop line?
                    if self.is_vehicle_crossing_line((tail_close_pt, tail_far_pt), (lft_lane_wp, rgt_lane_wp)):
                        self.test_status = "FAILURE"
                        self.actual_value += 1
                        location = traffic_light.get_transform().location
                        red_light_event = TrafficEvent(event_type=TrafficEventType.TRAFFIC_LIGHT_INFRACTION)
                        red_light_event.set_message(
                            "Agent ran a red light {} at (x={}, y={}, z={})".format(
                                traffic_light.id,
                                round(location.x, 3),
                                round(location.y, 3),
                                round(location.z, 3)
                            )
                        )
                        red_light_event.set_dict({'id': traffic_light.id, 'x': location.x, 'y': location.y, 'z': location.z})

                        self.list_traffic_events.append(red_light_event)
                        self._last_red_light_id = traffic_light.id
                        break

        if self._terminate_on_failure and (self.test_status == "FAILURE"):
            new_status = Status.FAILURE
        return self.actual_value

    def rotate_point(self, point, angle):
        """
            Rotate a given point by a given angle
        """
        x_ = math.cos(math.radians(angle)) * point.x - math.sin(math.radians(angle)) * point.y
        y_ = math.sin(math.radians(angle)) * point.x + math.cos(math.radians(angle)) * point.y
        return carla.Vector3D(x_, y_, point.z)

    def get_traffic_light_waypoints(self, traffic_light):
        """
            Get area of a given traffic light
        """
        base_transform = traffic_light.get_transform()
        base_rot = base_transform.rotation.yaw
        area_loc = base_transform.transform(traffic_light.trigger_volume.location)

        # Discretize the trigger box into points
        area_ext = traffic_light.trigger_volume.extent
        x_values = np.arange(-0.9 * area_ext.x, 0.9 * area_ext.x, 1.0)  # 0.9 to avoid crossing to adjacent lanes

        area = []
        for x in x_values:
            point = self.rotate_point(carla.Vector3D(x, 0, area_ext.z), base_rot)
            point_location = area_loc + carla.Location(x=point.x, y=point.y)
            area.append(point_location)

        # Get the waypoints of these points, removing duplicates
        ini_wps = []
        for pt in area:
            wpx = self._map.get_waypoint(pt)
            # As x_values are arranged in order, only the last one has to be checked
            if not ini_wps or ini_wps[-1].road_id != wpx.road_id or ini_wps[-1].lane_id != wpx.lane_id:
                ini_wps.append(wpx)

        # Advance them until the intersection
        wps = []
        for wpx in ini_wps:
            while not wpx.is_intersection:
                next_wp = wpx.next(0.5)[0]
                if next_wp and not next_wp.is_intersection:
                    wpx = next_wp
                else:
                    break
            wps.append(wpx)

        return area_loc, wps


class RunningStopTest(Criterion):
    """
        Check if an actor is running a stop sign

        Important parameters:
        - actor: CARLA actor to be used for this test
        - terminate_on_failure [optional]: If True, the complete scenario will terminate upon failure of this test
    """

    PROXIMITY_THRESHOLD = 50.0  # meters
    SPEED_THRESHOLD = 0.1
    WAYPOINT_STEP = 1.0  # meters

    def __init__(self, actor, name="RunningStopTest", terminate_on_failure=False):
        super(RunningStopTest, self).__init__(name, actor, 0, terminate_on_failure=terminate_on_failure)
        self._actor = actor
        self._world = CarlaDataProvider.get_world()
        self._map = CarlaDataProvider.get_map()
        self._list_stop_signs = []
        self._target_stop_sign = None
        self._stop_completed = False
        self._affected_by_stop = False
        self.actual_value = 0

        all_actors = self._world.get_actors()
        for _actor in all_actors:
            if 'traffic.stop' in _actor.type_id:
                self._list_stop_signs.append(_actor)

    @staticmethod
    def point_inside_boundingbox(point, bb_center, bb_extent):
        # pylint: disable=invalid-name
        A = carla.Vector2D(bb_center.x - bb_extent.x, bb_center.y - bb_extent.y)
        B = carla.Vector2D(bb_center.x + bb_extent.x, bb_center.y - bb_extent.y)
        D = carla.Vector2D(bb_center.x - bb_extent.x, bb_center.y + bb_extent.y)
        M = carla.Vector2D(point.x, point.y)

        AB = B - A
        AD = D - A
        AM = M - A
        am_ab = AM.x * AB.x + AM.y * AB.y
        ab_ab = AB.x * AB.x + AB.y * AB.y
        am_ad = AM.x * AD.x + AM.y * AD.y
        ad_ad = AD.x * AD.x + AD.y * AD.y

        return am_ab > 0 and am_ab < ab_ab and am_ad > 0 and am_ad < ad_ad

    def is_actor_affected_by_stop(self, actor, stop, multi_step=20):
        """
        Check if the given actor is affected by the stop
        """
        affected = False
        # first we run a fast coarse test
        current_location = actor.get_location()
        stop_location = stop.get_transform().location
        if stop_location.distance(current_location) > self.PROXIMITY_THRESHOLD:
            return affected

        stop_t = stop.get_transform()
        transformed_tv = stop_t.transform(stop.trigger_volume.location)

        # slower and accurate test based on waypoint's horizon and geometric test
        list_locations = [current_location]
        waypoint = self._map.get_waypoint(current_location)
        for _ in range(multi_step):
            if waypoint:
                next_wps = waypoint.next(self.WAYPOINT_STEP)
                if not next_wps:
                    break
                waypoint = next_wps[0]
                if not waypoint:
                    break
                list_locations.append(waypoint.transform.location)

        for actor_location in list_locations:
            if self.point_inside_boundingbox(actor_location, transformed_tv, stop.trigger_volume.extent):
                affected = True

        return affected

    def _scan_for_stop_sign(self):
        target_stop_sign = None

        ve_tra = CarlaDataProvider.get_transform(self._actor)
        ve_dir = ve_tra.get_forward_vector()

        wp = self._map.get_waypoint(ve_tra.location)
        wp_dir = wp.transform.get_forward_vector()

        dot_ve_wp = ve_dir.x * wp_dir.x + ve_dir.y * wp_dir.y + ve_dir.z * wp_dir.z

        if dot_ve_wp > 0:  # Ignore all when going in a wrong lane
            for stop_sign in self._list_stop_signs:
                if self.is_actor_affected_by_stop(self._actor, stop_sign):
                    # this stop sign is affecting the vehicle
                    target_stop_sign = stop_sign
                    break

        return target_stop_sign

    def update(self):
        """
            Check if the actor is running a red light
        """
        new_status = Status.RUNNING

        location = self._actor.get_location()
        if location is None:
            return self.actual_value

        if not self._target_stop_sign:
            # scan for stop signs
            self._target_stop_sign = self._scan_for_stop_sign()
        else:
            # we were in the middle of dealing with a stop sign
            if not self._stop_completed:
                # did the ego-vehicle stop?
                current_speed = CarlaDataProvider.get_velocity(self._actor)
                if current_speed < self.SPEED_THRESHOLD:
                    self._stop_completed = True

            if not self._affected_by_stop:
                stop_location = self._target_stop_sign.get_location()
                stop_extent = self._target_stop_sign.trigger_volume.extent

                if self.point_inside_boundingbox(location, stop_location, stop_extent):
                    self._affected_by_stop = True

            if not self.is_actor_affected_by_stop(self._actor, self._target_stop_sign):
                # is the vehicle out of the influence of this stop sign now?
                if not self._stop_completed and self._affected_by_stop:
                    # did we stop?
                    self.actual_value += 1
                    self.test_status = "FAILURE"
                    stop_location = self._target_stop_sign.get_transform().location
                    running_stop_event = TrafficEvent(event_type=TrafficEventType.STOP_INFRACTION)
                    running_stop_event.set_message(
                        "Agent ran a stop with id={} at (x={}, y={}, z={})".format(
                            self._target_stop_sign.id,
                            round(stop_location.x, 3),
                            round(stop_location.y, 3),
                            round(stop_location.z, 3)))
                    running_stop_event.set_dict({
                        'id': self._target_stop_sign.id,
                        'x': stop_location.x,
                        'y': stop_location.y,
                        'z': stop_location.z})

                    self.list_traffic_events.append(running_stop_event)

                # reset state
                self._target_stop_sign = None
                self._stop_completed = False
                self._affected_by_stop = False

        if self._terminate_on_failure and (self.test_status == "FAILURE"):
            new_status = Status.FAILURE

        return self.actual_value
