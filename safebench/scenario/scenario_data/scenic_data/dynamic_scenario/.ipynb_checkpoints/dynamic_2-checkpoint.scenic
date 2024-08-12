Town = 'Town05'
param map = localPath(f'../maps/{Town}.xodr') 
param carla_map = Town
model scenic.simulators.carla.model
EGO_MODEL = "vehicle.lincoln.mkz_2017
behavior CrossingFromLeftBehavior(speed):
    while True:
        take SetWalkingDirectionAction(heading right by 90 deg)
        take SetWalkingSpeedAction(speed)

behavior AdvBehavior():
    # Wait until the ego vehicle is within a certain distance
    while (distance to ego) > globalParameters.OPT_ADV_DISTANCE:
        wait
    # Cross the road from the left front
    do CrossingFromLeftBehavior(globalParameters.OPT_ADV_SPEED) until (distance from self to egoTrajectory) < globalParameters.OPT_STOP_DISTANCE
    # Stop as the ego vehicle approaches
    take SetWalkingSpeedAction(0)

param OPT_ADV_SPEED = Range(0, 5)
param OPT_ADV_DISTANCE = Range(0, 15)
param OPT_STOP_DISTANCE = Range(0, 1)
lane = Uniform(*network.lanes)
egoTrajectory = lane.centerline
EgoSpawnPt = OrientedPoint on lane.centerline

ego = Car at EgoSpawnPt,
    with regionContainedIn None,
    with blueprint EGO_MODEL
param OPT_GEO_BLOCKER_X_DISTANCE = Range(-8, -2)  # Negative range for left side
param OPT_GEO_BLOCKER_Y_DISTANCE = Range(15, 50)
param OPT_GEO_X_DISTANCE = Range(-2, 2)
param OPT_GEO_Y_DISTANCE = Range(2, 6)

LeftFrontSpawnPt = OrientedPoint following roadDirection from EgoSpawnPt for globalParameters.OPT_GEO_BLOCKER_Y_DISTANCE
Blocker = Bench left of LeftFrontSpawnPt by globalParameters.OPT_GEO_BLOCKER_X_DISTANCE,
    with heading LeftFrontSpawnPt.heading,
    with regionContainedIn None

SHIFT = globalParameters.OPT_GEO_X_DISTANCE @ globalParameters.OPT_GEO_Y_DISTANCE
AdvAgent = Pedestrian at Blocker offset along LeftFrontSpawnPt.heading by SHIFT,
    with heading LeftFrontSpawnPt.heading - 90 deg,  # Adjusted for sprinting out from the left
    with regionContainedIn None,
    with behavior AdvBehavior()