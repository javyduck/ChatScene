'''The ego attempts an unprotected left turn at an intersection while yielding to an oncoming car when the oncoming car veer erratically.'''
Town = 'Town05'
param map = localPath(f'../maps/{Town}.xodr') 
param carla_map = Town
model scenic.simulators.carla.model
EGO_MODEL = "vehicle.lincoln.mkz_2017"

param OPT_ADV_SPEED = Range(5, 10)
param OPT_LANE_SWAP_FREQUENCY = Range(20, 50)  # Frequency of lane swaps
param OPT_STEER = Range(-1, 1)  # Steering range for lane changes
param OPT_WAIT_LEFT = Range(5, 20)  # Wait time after steering left

behavior AdvBehavior():
    do FollowLaneBehavior(target_speed=globalParameters.OPT_ADV_SPEED)
    
    while True:
        take SetSteerAction(Range(-1, 0))  # Steering left
        for _ in range(globalParameters.OPT_WAIT_LEFT):  # Wait for a random duration in Range(5,20)
            wait

        take SetSteerAction(Range(0, 1))  # Steering right
        for _ in range(globalParameters.OPT_LANE_SWAP_FREQUENCY):  # Wait for a random duration in Range(20,50)
            wait

## MONITORS
monitor TrafficLights:
    freezeTrafficLights()
    while True:
        if withinDistanceToTrafficLight(ego, 100):
            setClosestTrafficLightStatus(ego, "green")
        if withinDistanceToTrafficLight(AdvAgent, 100):
            setClosestTrafficLightStatus(AdvAgent, "green")
        wait

intersection = Uniform(*filter(lambda i: i.is4Way and i.isSignalized, network.intersections))
egoInitLane = Uniform(*intersection.incomingLanes)
egoManeuver = Uniform(*filter(lambda m: m.type is ManeuverType.LEFT_TURN, egoInitLane.maneuvers))
egoTrajectory = [egoInitLane, egoManeuver.connectingLane, egoManeuver.endLane]
egoSpawnPt = OrientedPoint in egoManeuver.startLane.centerline

# Setting up the ego vehicle at the initial position
ego = Car at egoSpawnPt,
    with regionContainedIn None,
    with blueprint EGO_MODEL

require 10 <= (distance to intersection) <= 40
# Parameters for scenario elements
param OPT_GEO_BLOCKER_Y_DISTANCE = Range(0, 40)
param OPT_GEO_X_DISTANCE = Range(-8, 0)  # Offset for the agent in the opposite lane
param OPT_GEO_Y_DISTANCE = Range(10, 30)

# Setting up the parked car that blocks the ego's path
laneSec = network.laneSectionAt(ego)  # Assuming network.laneSectionAt(ego) is predefined in the geometry part
IntSpawnPt = OrientedPoint following roadDirection from egoSpawnPt for globalParameters.OPT_GEO_BLOCKER_Y_DISTANCE
Blocker = Car at IntSpawnPt,
    with heading IntSpawnPt.heading,
    with regionContainedIn None

# Setup for the motorcyclist who unexpectedly enters the scene
SHIFT = globalParameters.OPT_GEO_X_DISTANCE @ globalParameters.OPT_GEO_Y_DISTANCE
AdvAgent = Car at Blocker offset along IntSpawnPt.heading by SHIFT,
    with heading IntSpawnPt.heading + 180 deg,  # The agent is facing the opposite direction, indicating oncoming
    with regionContainedIn laneSec._laneToLeft,  # Positioned in the left lane, assuming it's the oncoming traffic lane
    with behavior AdvBehavior()