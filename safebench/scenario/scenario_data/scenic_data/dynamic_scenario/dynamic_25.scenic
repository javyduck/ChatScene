'''The ego starts an unprotected left turn at an intersection while yielding to an oncoming car when the oncoming car's throttle malfunctions, leading to an unexpected acceleration and forcing the ego to quickly modify its turning path to avoid a collision.'''
Town = 'Town05'
param map = localPath(f'../maps/{Town}.xodr') 
param carla_map = Town
model scenic.simulators.carla.model
EGO_MODEL = "vehicle.lincoln.mkz_2017"

behavior AdvBehavior():
    while (distance to self) > 60:
        wait  # The adversarial car waits until it is within 60 meters of the ego vehicle.

    do FollowLaneBehavior(globalParameters.OPT_ADV_SPEED) until (
        distance to self < globalParameters.OPT_ADV_DISTANCE)

    while True:
        take SetThrottleAction(globalParameters.OPT_ADV_THROTTLE)  # Aggressively adjusts acceleration.
        
        # Wait for a dynamically determined duration
        for _ in range(globalParameters.OPT_WAIT_THROTTLE):
            wait

        take SetBrakeAction(globalParameters.OPT_ADV_BREAK)  # Aggressively adjusts braking.

        # Wait for a dynamically determined duration
        for _ in range(globalParameters.OPT_WAIT_BRAKE):
            wait

param OPT_ADV_SPEED = Range(0, 20)  # Controls the initial speed of the adversarial car.
param OPT_ADV_DISTANCE = Range(0, 20)  # Specifies the distance at which the car begins its aggressive maneuver.
param OPT_ADV_THROTTLE = Range(0, 1)  # Aggressive throttle range.
param OPT_ADV_BREAK = Range(0, 1)  # Determines the intensity of the braking.
param OPT_WAIT_THROTTLE = Range(5, 20)  # Variable wait time between throttle changes.
param OPT_WAIT_BRAKE = Range(5, 20)  # Variable wait time between brake applications.

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
# Setup the leading vehicle's spawn point directly in front of the ego to simulate a slow-moving vehicle
param OPT_LEADING_DISTANCE = Range(0, 30)
param OPT_LEADING_SPEED = Range(1, 5)
LeadingSpawnPt = OrientedPoint following roadDirection from egoSpawnPt for globalParameters.OPT_LEADING_DISTANCE
LeadingAgent = Car at LeadingSpawnPt,
    with behavior FollowLaneBehavior(target_speed=globalParameters.OPT_LEADING_SPEED)

# Identifying the adjacent lane to the left for the Adversarial Agent and setting its spawn point further in front
param OPT_GEO_Y_DISTANCE = Range(0, 30)
advLane = network.laneSectionAt(ego)._laneToLeft.lane
IntSpawnPt = OrientedPoint following roadDirection from egoSpawnPt for globalParameters.OPT_GEO_Y_DISTANCE
projectPt = Vector(*advLane.centerline.project(IntSpawnPt.position).coords[0])
advHeading = advLane.orientation[projectPt]

# Spawn the Adversarial Agent
AdvAgent = Car at projectPt,
    with heading advHeading,
    with regionContainedIn None,
    with behavior AdvBehavior()