'''The ego approaches a parked car obstructing its lane and must use the opposite lane to go around when an oncoming car suddenly turns into the ego's path without signaling, requiring the ego to react quickly and take evasive action to prevent a collision.
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
    do FollowLaneBehavior(globalParameters.OPT_ADV_SPEED)

param OPT_GEO_BLOCKER_Y_DISTANCE = Range(0, 20)
param OPT_GEO_X_DISTANCE = Range(-8, 0)
param OPT_GEO_Y_DISTANCE = Range(10, 30)
param OPT_ADV_SPEED = Range(0, 10)

ego = Car at EgoSpawnPt,
    with heading yaw,
    with regionContainedIn None,
    with blueprint EGO_MODEL

NewSpawnPt = egoTrajectory[40]
laneSec = network.laneSectionAt(NewSpawnPt)
IntSpawnPt = OrientedPoint following roadDirection from NewSpawnPt for globalParameters.OPT_GEO_BLOCKER_Y_DISTANCE
Blocker = Car at IntSpawnPt,
    with heading IntSpawnPt.heading,
    with regionContainedIn None
    
SHIFT = globalParameters.OPT_GEO_X_DISTANCE @ globalParameters.OPT_GEO_Y_DISTANCE
AdvAgent = Car at Blocker offset along IntSpawnPt.heading by SHIFT,
    with heading IntSpawnPt.heading + 180 deg,
    with regionContainedIn laneSec._laneToLeft,
    with behavior AdvBehavior()
