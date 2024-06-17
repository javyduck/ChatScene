'''The ego vehicle is attempting to change lanes to avoid a slow-moving leading vehicle; the adversarial car in the target lane suddenly slows down, matching the speed of the leading vehicle, and effectively blocking the ego vehicle from completing the lane change.
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
    do FollowLaneBehavior(target_speed=globalParameters.OPT_ADV_SPEED) until (distance to self) < globalParameters.OPT_ADV_DISTANCE
    while True:
        take SetVelocityAction(*LeadingAgent.velocity)
                
param OPT_LEADING_DISTANCE = Range(0, 30)
param OPT_LEADING_SPEED = Range(1, 5)
param OPT_GEO_Y_DISTANCE = Range(0, 30)
param OPT_ADV_SPEED = Range(5, 10)
param OPT_ADV_DISTANCE = Range(0, 20)

ego = Car at EgoSpawnPt,
    with heading yaw,
    with regionContainedIn None,
    with blueprint EGO_MODEL
    
NewSpawnPt = egoTrajectory[20]
LeadingAgent = Car following roadDirection from NewSpawnPt for globalParameters.OPT_LEADING_DISTANCE,
    with regionContainedIn None,
    with behavior FollowLaneBehavior(target_speed=globalParameters.OPT_LEADING_SPEED)
   
laneSec = network.laneSectionAt(NewSpawnPt)
advLane = laneSec._laneToRight.lane
IntSpawnPt = OrientedPoint following roadDirection from NewSpawnPt for globalParameters.OPT_GEO_Y_DISTANCE
projectPt = Vector(*advLane.centerline.project(IntSpawnPt.position).coords[0])
advHeading = advLane.orientation[projectPt]
AdvAgent = Car at projectPt,
    with heading advHeading,
    with regionContainedIn None,
    with behavior AdvBehavior()
