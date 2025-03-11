'''The ego vehicle is turning right at an intersection; the adversarial motorcyclist on the opposite sidewalk abruptly crosses the road from the right front and comes to a halt in the center of the intersection.'''
Town = 'Town05'
param map = localPath(f'../maps/{Town}.xodr') 
param carla_map = Town
model scenic.simulators.carla.model
EGO_MODEL = "vehicle.lincoln.mkz_2017"

behavior AdvBehavior():
    do CrossingBehavior(ego, globalParameters.OPT_ADV_SPEED, globalParameters.OPT_ADV_DISTANCE) until (distance from self to egoTrajectory) < globalParameters.OPT_STOP_DISTANCE
    while True:
        take SetSpeedAction(0)

param OPT_ADV_SPEED = Range(0, 10)
param OPT_ADV_DISTANCE = Range(0, 15)
param OPT_STOP_DISTANCE = Range(0, 1)
intersection = Uniform(*filter(lambda i: i.is4Way or i.is3Way, network.intersections))
egoManeuver = Uniform(*filter(lambda m: m.type is ManeuverType.RIGHT_TURN, intersection.maneuvers))
egoInitLane = egoManeuver.startLane
egoTrajectory = [egoInitLane, egoManeuver.connectingLane, egoManeuver.endLane]
egoSpawnPt = OrientedPoint in egoInitLane.centerline

ego = Car at egoSpawnPt,
    with regionContainedIn None,
    with blueprint EGO_MODEL
# Parameters for scenario elements
param OPT_GEO_BLOCKER_Y_DISTANCE = Range(0, 40)
param OPT_GEO_X_DISTANCE = Range(-2, 2)
param OPT_GEO_Y_DISTANCE = Range(2, 6)

# Setup for the blocking car that the ego must bypass
laneSec = network.laneSectionAt(ego)
IntSpawnPt = OrientedPoint following roadDirection from egoSpawnPt for globalParameters.OPT_GEO_BLOCKER_Y_DISTANCE
Blocker = Car at IntSpawnPt,
    with heading IntSpawnPt.heading,
    with regionContainedIn None

# Setup for the pedestrian who suddenly appears and complicates the maneuver
SHIFT = globalParameters.OPT_GEO_X_DISTANCE @ globalParameters.OPT_GEO_Y_DISTANCE
AdvAgent = Motorcycle at Blocker offset along IntSpawnPt.heading by SHIFT,
    with heading IntSpawnPt.heading + 90 deg,  # Perpendicular to the road, crossing the street
    with regionContainedIn None,
    with behavior AdvBehavior()