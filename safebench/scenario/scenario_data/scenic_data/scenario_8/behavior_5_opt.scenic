'''The ego vehicle is maintaining a constant speed; the adversarial car, comes from the right, blocks multiple lanes by driving extremely slowly, forcing the ego vehicle to change lanes.
'''
Town = globalParameters.town
EgoSpawnPt = globalParameters.spawnPt
yaw = globalParameters.yaw
lanePts = globalParameters.lanePts
egoTrajectory = PolylineRegion(globalParameters.waypoints)
param map = localPath(f'../maps/{Town}.xodr') 
param carla_map = Town
model scenic.simulators.carla.model
EGO_MODEL = "vehicle.lincoln.mkz_2017"

behavior AdvBehavior():
    while (distance to self) > 60:
        wait
    do FollowTrajectoryBehavior(globalParameters.OPT_ADV_SPEED, advTrajectory) until (distance to self) < globalParameters.OPT_ADV_DISTANCE or (distance from self to egoTrajectory) < globalParameters.OPT_SLOW_DISTANCE
    while True:
        take SetSpeedAction(globalParameters.OPT_SLOW_SPEED)
        
param OPT_GEO_Y_DISTANCE = Range(-5, 10)
param OPT_ADV_DISTANCE = Range(0, 20)
param OPT_SLOW_DISTANCE = Range(0, 4)
param OPT_ADV_SPEED = Range(5, 15)
param OPT_SLOW_SPEED = Range(0, 2)

egoInitLane = network.laneAt(lanePts[-3])
egoManeuver = Uniform(*filter(lambda m: m.type is ManeuverType.STRAIGHT, egoInitLane.maneuvers))
advManeuvers = filter(lambda i: i.type == ManeuverType.STRAIGHT, egoManeuver.conflictingManeuvers)
advManeuver = Uniform(*advManeuvers)
advTrajectory = [advManeuver.startLane, advManeuver.connectingLane, advManeuver.endLane]

IntSpawnPt = advManeuver.connectingLane.centerline.start
ego = Car at EgoSpawnPt,
    with heading yaw,
    with regionContainedIn None,
    with blueprint EGO_MODEL

AdvAgent = Car following roadDirection from IntSpawnPt for globalParameters.OPT_GEO_Y_DISTANCE,
    with heading IntSpawnPt.heading,
    with regionContainedIn None,
    with behavior AdvBehavior()
    
require 20 deg <= RelativeHeading(AdvAgent) <= 160 deg