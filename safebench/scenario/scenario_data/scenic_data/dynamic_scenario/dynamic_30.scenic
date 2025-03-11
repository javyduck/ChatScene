'''The ego is performing a right turn at an intersection when the crossing car suddenly speeds up, entering the intersection and causing the ego to brake abruptly to avoid a collision.'''
Town = 'Town05'
param map = localPath(f'../maps/{Town}.xodr') 
param carla_map = Town
model scenic.simulators.carla.model
EGO_MODEL = "vehicle.lincoln.mkz_2017"


behavior AdvBehavior():
    while (distance to self) > 60:
        wait
    do FollowTrajectoryBehavior(globalParameters.OPT_ADV_SPEED, advTrajectory) until (distance to self) < globalParameters.OPT_ADV_DISTANCE
    while True:
        take SetThrottleAction(globalParameters.OPT_THROTTLE)

param OPT_ADV_SPEED = Range(5, 15)  # The speed at which the adversarial vehicle approaches the intersection.
param OPT_ADV_DISTANCE = Range(5, 20)  # The critical distance at which the adversarial begins to accelerate.
param OPT_THROTTLE = Range(0.5, 1.0)  # The intensity of the accelerate action, ranging from a slight decrease in speed to a full stop.
intersection = Uniform(*filter(lambda i: i.is4Way, network.intersections))
egoManeuver = Uniform(*filter(lambda m: m.type is ManeuverType.RIGHT_TURN, intersection.maneuvers))
egoInitLane = egoManeuver.startLane
egoTrajectory = [egoInitLane, egoManeuver.connectingLane, egoManeuver.endLane]
egoSpawnPt = OrientedPoint in egoInitLane.centerline

ego = Car at egoSpawnPt,
    with regionContainedIn None,
    with blueprint EGO_MODEL
# Defining adversarial maneuvers as those conflicting with the ego's straight path
advManeuvers = filter(lambda i: i.type == ManeuverType.LEFT_TURN, egoManeuver.conflictingManeuvers)
advManeuver = Uniform(*advManeuvers)
advTrajectory = [advManeuver.startLane, advManeuver.connectingLane, advManeuver.endLane]
advSpawnPt = advManeuver.connectingLane.centerline[0]  # Initial point on the connecting lane's centerline
IntSpawnPt = advManeuver.connectingLane.centerline.start  # Start of the connecting lane centerline

param OPT_GEO_Y_DISTANCE = Range(-10, 10)
# Setting up the adversarial agent
AdvAgent = Car following roadDirection from IntSpawnPt for globalParameters.OPT_GEO_Y_DISTANCE,
    with heading IntSpawnPt.heading,
    with regionContainedIn None,
    with behavior AdvBehavior()

# Requirements to ensure the adversarial agent's relative position and trajectory are correctly aligned with the scenario's needs
require 160 deg <= abs(RelativeHeading(AdvAgent)) <= 180 deg
require any([AdvAgent.position in traj for traj in [advManeuver.startLane, advManeuver.connectingLane, advManeuver.endLane]])