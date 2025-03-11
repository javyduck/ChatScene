'''The ego vehicle is turning right; the adversarial car (positioned behind on the right) suddenly accelerates and then decelerates.'''
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
intersection = Uniform(*filter(lambda i: i.is4Way or i.is3Way, network.intersections))
egoManeuver = Uniform(*filter(lambda m: m.type is ManeuverType.RIGHT_TURN, intersection.maneuvers))
egoInitLane = egoManeuver.startLane
egoTrajectory = [egoInitLane, egoManeuver.connectingLane, egoManeuver.endLane]
egoSpawnPt = OrientedPoint in egoInitLane.centerline

ego = Car at egoSpawnPt,
    with regionContainedIn None,
    with blueprint EGO_MODEL
# Setup the leading vehicle's spawn point directly in front of the ego to simulate a slow-moving vehicle
param OPT_LEADING_DISTANCE = Range(0, 30)
param OPT_LEADING_SPEED = Range(1, 5)
LeadingSpawnPt = OrientedPoint following roadDirection from egoSpawnPt for globalParameters.OPT_LEADING_DISTANCE
LeadingAgent = Car at LeadingSpawnPt,
    with behavior FollowLaneBehavior(target_speed=globalParameters.OPT_LEADING_SPEED)

# Identifying the adjacent lane for the Adversarial Agent and setting its spawn point further in front
param OPT_GEO_Y_DISTANCE = Range(0, 30)
advLane = network.laneSectionAt(ego)._laneToRight.lane
IntSpawnPt = OrientedPoint following roadDirection from egoSpawnPt for globalParameters.OPT_GEO_Y_DISTANCE
projectPt = Vector(*advLane.centerline.project(IntSpawnPt.position).coords[0])
advHeading = advLane.orientation[projectPt]

# Spawn the Adversarial Agent
AdvAgent = Car at projectPt,
    with heading advHeading,
    with regionContainedIn None,
    with behavior AdvBehavior()