'''The ego vehicle is moving straight through the intersection; the adversarial agent, initially on the left front, runs the red light and makes an abrupt left turn, forcing the ego vehicle to perform a collision avoidance maneuver.
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

## MONITORS
monitor TrafficLights:
    freezeTrafficLights()
    while True:
        if withinDistanceToTrafficLight(ego, 100):
            setClosestTrafficLightStatus(ego, "green")
        if withinDistanceToTrafficLight(AdvAgent, 100):
            setClosestTrafficLightStatus(AdvAgent, "red")
        wait
        
behavior AdvBehavior():
    while (distance to self) > 60:
        wait
    do FollowTrajectoryBehavior(globalParameters.OPT_ADV_SPEED, advTrajectory) until (distance from self to egoTrajectory) < globalParameters.OPT_STEER_DISTANCE
    while True:
        take SetSpeedAction(self.speed), SetSteerAction(-globalParameters.OPT_STEER)
  
param OPT_GEO_Y_DISTANCE = Range(-5, 10)
param OPT_ADV_SPEED = Range(5, 15)
param OPT_STEER_DISTANCE = Range(0, 4)
param OPT_STEER = Range(0.5, 1.0)

egoInitLane = network.laneAt(lanePts[0])
egoManeuver = Uniform(*filter(lambda m: m.type is ManeuverType.STRAIGHT, egoInitLane.maneuvers))
advManeuvers = filter(lambda i: i.type == ManeuverType.STRAIGHT, egoManeuver.conflictingManeuvers)
if len(advManeuvers.sample()):
    advManeuver = Uniform(*advManeuvers)
else:
    # route 6 has some problems
    advManeuver = egoManeuver.conflictingManeuvers[0]
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
        
require -160 deg <= RelativeHeading(AdvAgent) <= -20 deg
require any([AdvAgent.position in traj for traj in  [advManeuver.startLane, advManeuver.connectingLane, advManeuver.endLane]])