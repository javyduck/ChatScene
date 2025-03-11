'''The ego vehicle is approaching the intersection; the adversarial car (on the right) suddenly accelerates and enters the intersection first and suddenly stop.'''
Town = 'Town05'
param map = localPath(f'../maps/{Town}.xodr') 
param carla_map = Town
model scenic.simulators.carla.model
EGO_MODEL = "vehicle.lincoln.mkz_2017"

behavior AdvBehavior():
    while (distance to self) > 60:
        wait  # The adversarial vehicle waits until it is close enough to impact the ego's maneuver.
    do FollowTrajectoryBehavior(globalParameters.OPT_ADV_SPEED, advTrajectory) until (distance to self) < globalParameters.OPT_ADV_DISTANCE or (distance from self to egoTrajectory) < globalParameters.OPT_BRAKE_DISTANCE
    # Accelerates towards the ego vehicle.
    while (distance from self to egoTrajectory) > globalParameters.OPT_BRAKE_DISTANCE:
        take SetThrottleAction(globalParameters.OPT_THROTTLE)  # Applies throttle to increase speed.
    # Once close enough, it decelerates abruptly.
    while True:
        take SetBrakeAction(globalParameters.OPT_BRAKE)  # Initiates a sudden brake to reduce speed sharply.

param OPT_ADV_SPEED = Range(5, 15)  # The speed at which the adversarial vehicle approaches.
param OPT_ADV_DISTANCE = Range(0, 25)  # The proximity within which the acceleration is triggered.
param OPT_THROTTLE = Range(0.5, 1.0)  # The intensity of the throttle during the acceleration.
param OPT_BRAKE = Range(0, 1)  # The intensity of the braking action.
param OPT_BRAKE_DISTANCE = Range(0, 4)  # The critical distance at which the adversarial vehicle begins its deceleration.
intersection = Uniform(*filter(lambda i: i.is4Way, network.intersections))
egoManeuver = Uniform(*filter(lambda m: m.type is ManeuverType.LEFT_TURN, intersection.maneuvers))
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