'''The ego vehicle is turning right; the adversarial car (positioned ahead on the right) reverses abruptly.'''
Town = 'Town05'
param map = localPath(f'../maps/{Town}.xodr') 
param carla_map = Town
model scenic.simulators.carla.model
EGO_MODEL = "vehicle.lincoln.mkz_2017"

behavior AdvBehavior():
    # Follow the lane at the speed defined by OPT_ADV_SPEED until a condition is met
    do FollowLaneBehavior(target_speed=globalParameters.OPT_ADV_SPEED) until (
        distance to some_target < globalParameters.OPT_ADV_DISTANCE)
    
    # Perform a lane change to an adjacent lane
    targetLaneSec = network.laneSectionAt(self).adjacentLanes[0]
    do LaneChangeBehavior(laneSectionToSwitch=targetLaneSec, target_speed=globalParameters.OPT_ADV_SPEED)

    # Wait for a number of steps directly specified by OPT_WAIT_STEPS
    for _ in range(globalParameters.OPT_WAIT_STEPS):
        wait

    # Stop the vehicle after waiting
    do SetSpeedAction(0)

param OPT_ADV_SPEED = Range(5, 10)  # Speed range for the vehicle
param OPT_ADV_DISTANCE = Range(0, 20)  # Distance threshold for stopping the follow behavior
param OPT_WAIT_STEPS = Range(0, 20)  # Directly used range for wait steps before stopping
intersection = Uniform(*filter(lambda i: i.is4Way or i.is3Way, network.intersections))
egoManeuver = Uniform(*filter(lambda m: m.type is ManeuverType.RIGHT_TURN, intersection.maneuvers))
egoInitLane = egoManeuver.startLane
egoTrajectory = [egoInitLane, egoManeuver.connectingLane, egoManeuver.endLane]
egoSpawnPt = OrientedPoint in egoInitLane.centerline

ego = Car at egoSpawnPt,
    with regionContainedIn None,
    with blueprint EGO_MODEL
# Setup the leading vehicle's spawn point directly in front of the ego to simulate a slow-moving vehicle
param OPT_LEADING_DISTANCE = Range(0, 30)
param OPT_LEADING_SPEED = Range(1, 5)
LeadingSpawnPt = OrientedPoint following roadDirection from egoSpawnPt for globalParameters.OPT_LEADING_DISTANCE
LeadingAgent = Car at LeadingSpawnPt,
    with behavior FollowLaneBehavior(target_speed=globalParameters.OPT_LEADING_SPEED)

# Identifying the adjacent lane for the Adversarial Agent and setting its spawn point further in front
param OPT_GEO_Y_DISTANCE = Range(0, 30)
advLane = network.laneSectionAt(ego)._laneToRight.lane
IntSpawnPt = OrientedPoint following roadDirection from egoSpawnPt for globalParameters.OPT_GEO_Y_DISTANCE
projectPt = Vector(*advLane.centerline.project(IntSpawnPt.position).coords[0])
advHeading = advLane.orientation[projectPt]

# Spawn the Adversarial Agent
AdvAgent = Car at projectPt,
    with heading advHeading,
    with regionContainedIn None,
    with behavior AdvBehavior()