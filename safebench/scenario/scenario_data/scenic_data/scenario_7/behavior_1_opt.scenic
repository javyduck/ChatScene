'''The ego is performing a right turn at an intersection when the crossing car suddenly speeds up, entering the intersection and causing the ego to brake abruptly to avoid a collision.
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
    do FollowTrajectoryBehavior(globalParameters.OPT_ADV_SPEED, advTrajectory) until (distance to self) < globalParameters.OPT_ADV_DISTANCE or (distance from self to egoTrajectory) < globalParameters.OPT_STOP_DISTANCE
    while (distance from self to egoTrajectory) > globalParameters.OPT_STOP_DISTANCE:
        take SetThrottleAction(globalParameters.OPT_THROTTLE)
    while True:
        take SetSpeedAction(0)
      
param OPT_GEO_Y_DISTANCE = Range(0, 15)
param OPT_ADV_SPEED = Range(5, 15)
param OPT_ADV_DISTANCE = Range(0, 25)
param OPT_THROTTLE = Range(0.5, 1.0)
param OPT_STOP_DISTANCE = Range(0, 2)

egoInitLane = network.laneAt(lanePts[-3])
egoManeuver = Uniform(*filter(lambda m: m.type is ManeuverType.RIGHT_TURN, egoInitLane.maneuvers))
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
    
require -110 deg <= RelativeHeading(AdvAgent) <= -70 deg
require any([AdvAgent.position in traj for traj in  [advManeuver.startLane, advManeuver.connectingLane, advManeuver.endLane]])