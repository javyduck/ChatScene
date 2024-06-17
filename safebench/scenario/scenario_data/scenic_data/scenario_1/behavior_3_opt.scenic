""" The ego vehicle is driving on a straight road; the adversarial pedestrian appears from a driveway on the left and suddenly stop and walk diagonally.
"""
Town = globalParameters.town
EgoSpawnPt = globalParameters.spawnPt
yaw = globalParameters.yaw
egoTrajectory = PolylineRegion(globalParameters.waypoints)
param map = localPath(f'../maps/{Town}.xodr') 
param carla_map = Town
model scenic.simulators.carla.model
EGO_MODEL = "vehicle.lincoln.mkz_2017"

behavior AdvBehavior():
    direction = self.heading + globalParameters.OPT_ADV_DEGREE deg
    while (distance to self) > globalParameters.OPT_ADV_DISTANCE:
        wait
    while True:
        take SetWalkingDirectionAction(direction)
        take SetWalkingSpeedAction(globalParameters.OPT_ADV_SPEED)

param OPT_GEO_X_DISTANCE = Range(2, 8)
param OPT_GEO_Y_DISTANCE = Range(15, 50)
param OPT_ADV_SPEED = Range(0, 5)
param OPT_ADV_DISTANCE = Range(0, 15)
param OPT_ADV_DEGREE = Range(-90, 90)

ego = Car at EgoSpawnPt,
    with heading yaw,
    with regionContainedIn None,
    with blueprint EGO_MODEL
    
IntSpawnPt = OrientedPoint following roadDirection from EgoSpawnPt for globalParameters.OPT_GEO_Y_DISTANCE
AdvAgent = Pedestrian left of IntSpawnPt by globalParameters.OPT_GEO_X_DISTANCE,
    with heading IntSpawnPt.heading - 90 deg,
    with regionContainedIn None,
    with behavior AdvBehavior()

require (distance from AdvAgent to intersection) > 10