'''The ego vehicle is driving on a straight road when a pedestrian suddenly crosses from the right front and stops as the ego vehicle approaches.'''
Town = 'Town05'
param map = localPath(f'../maps/{Town}.xodr') 
param carla_map = Town
model scenic.simulators.carla.model
EGO_MODEL = "vehicle.lincoln.mkz_2017"

behavior AdvBehavior():
    while (distance to self) > globalParameters.OPT_ADV_TRIGGER_DISTANCE:
        wait

param OPT_ADV_SPEED = Range(0, 5)
param OPT_ADV_TRIGGER_DISTANCE = Range(0, 10)
param OPT_ADV_CROSSING_DIRECTION = Range(-45, 45)  # Approximate range for crossing from the right front
param OPT_CROSSING_TIME = 5  # Time in seconds the pedestrian actively crosses before stopping
lane = Uniform(*network.lanes)
EgoTrajectory = lane.centerline
EgoSpawnPt = OrientedPoint on lane.centerline

ego = Car at EgoSpawnPt,
    with regionContainedIn None,
    with blueprint EGO_MODEL
param OPT_GEO_X_DISTANCE = Range(2, 8)
param OPT_GEO_Y_DISTANCE = Range(15, 50)

IntSpawnPt = OrientedPoint following roadDirection from EgoSpawnPt for globalParameters.OPT_GEO_Y_DISTANCE
AdvAgent = Pedestrian right of IntSpawnPt by globalParameters.OPT_GEO_X_DISTANCE,
    with heading LeftFrontSpawnPt.heading - 90 deg,  # Heading perpendicular to the road, adjusted for crossing from right
    with regionContainedIn None,
    with behavior AdvBehavior()