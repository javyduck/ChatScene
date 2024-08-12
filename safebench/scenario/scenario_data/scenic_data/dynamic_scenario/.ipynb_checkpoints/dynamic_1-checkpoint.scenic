Town = 'Town05'
param map = localPath(f'../maps/{Town}.xodr') 
param carla_map = Town
model scenic.simulators.carla.model
EGO_MODEL = "vehicle.lincoln.mkz_2017
behavior CrossingFromRightFrontBehavior(ego, speed, distance, stopDistance):
    do CrossingBehavior(ego, speed, distance) until (distance from self to egoTrajectory) < stopDistance
    take SetWalkingSpeedAction(0)

behavior AdvBehavior():
    initialDirection = self.heading
    while (distance to self) > globalParameters.OPT_ADV_APPROACH_DISTANCE:
        wait
    do CrossingFromRightFrontBehavior(ego, globalParameters.OPT_ADV_SPEED, globalParameters.OPT_ADV_DISTANCE, globalParameters.OPT_STOP_DISTANCE) until True

param OPT_ADV_SPEED = Range(0, 5)
param OPT_ADV_DISTANCE = Range(0, 15)
param OPT_STOP_DISTANCE = Range(0, 1)
param OPT_ADV_APPROACH_DISTANCE = Range(0, 20)
intersection = Uniform(*filter(lambda i: i.is4Way, network.intersections))
egoInitLane = Uniform(*intersection.incomingLanes)
egoManeuver = Uniform(*filter(lambda m: m.type is ManeuverType.STRAIGHT, egoInitLane.maneuvers))
egoTrajectory = [egoInitLane, egoManeuver.connectingLane, egoManeuver.endLane]
egoSpawnPt = OrientedPoint in egoManeuver.startLane.centerline

# Setting up the ego vehicle at the initial position
ego = Car at egoSpawnPt,
    with regionContainedIn None,
    with blueprint EGO_MODEL

require 10 <= (distance to intersection) <= 40
param OPT_GEO_X_DISTANCE = Range(2, 8)
param OPT_GEO_Y_DISTANCE = Range(15, 50)

RightFrontIntersectionPt = OrientedPoint right of EgoSpawnPt by globalParameters.OPT_GEO_X_DISTANCE,
    following roadDirection from EgoSpawnPt for globalParameters.OPT_GEO_Y_DISTANCE

AdvAgent = Pedestrian at RightFrontIntersectionPt,
    with heading RightFrontIntersectionPt.heading + 90 deg,  # Heading perpendicular to the road, adjusted for right crossing
    with regionContainedIn None,
    with behavior AdvBehavior()