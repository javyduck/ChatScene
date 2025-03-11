'''The ego vehicle is preparing to change lanes to evade a slow-moving leading vehicle; the adversarial car in the target lane starts weaving between lanes, making it difficult for the ego vehicle to predict its position and safely execute the lane change.'''
Town = 'Town03'
param map = localPath(f'../maps/{Town}.xodr') 
param carla_map = Town
model scenic.simulators.carla.model
EGO_MODEL = "vehicle.lincoln.mkz_2017"

behavior AdvBehavior():
    while True:
        # Follow the lane at the set speed until a certain proximity condition is met
        do FollowLaneBehavior(target_speed=globalParameters.OPT_ADV_SPEED) until distance to some_target < globalParameters.OPT_ADV_DISTANCE  # This condition should refer to a valid object or marker.
        
        # Perform two quick lane changes with a 1-second wait in between
        for _ in range(2):
            if len(network.laneSectionAt(self).adjacentLanes) > 0:
                targetLaneSec = network.laneSectionAt(self).adjacentLanes[0]
                do LaneChangeBehavior(laneSectionToSwitch=targetLaneSec, target_speed=globalParameters.OPT_ADV_SPEED)
                
            # Wait for a fixed interval of 1 second before potentially changing the lane again
            for _ in range(globalParameters.OPT_WAIT_STEPS):
                wait

param OPT_ADV_SPEED = Range(5, 15)  # Range of speeds for lane following
param OPT_ADV_DISTANCE = Range(0, 20)  # Distance threshold for lane change trigger
param OPT_WAIT_STEPS = Range(0, 20)
# Identifying lane sections with a right lane moving in the same forward direction
laneSecsWithRightLane = []
for lane in network.lanes:
    for laneSec in lane.sections:
        if laneSec._laneToRight is not None and laneSec._laneToRight.isForward == laneSec.isForward:
            laneSecsWithRightLane.append(laneSec)

# Selecting a random lane section from identified sections for the ego vehicle
egoLaneSec = Uniform(*laneSecsWithRightLane)
egoSpawnPt = OrientedPoint in egoLaneSec.centerline

# Ego vehicle setup
ego = Car at egoSpawnPt,
    with regionContainedIn None,
    with blueprint EGO_MODEL
# Setup the leading vehicle's spawn point directly in front of the ego to simulate a slow-moving vehicle
param OPT_LEADING_DISTANCE = Range(0, 30)
param OPT_LEADING_SPEED = Range(1, 5)
LeadingSpawnPt = OrientedPoint following roadDirection from egoSpawnPt for globalParameters.OPT_LEADING_DISTANCE
LeadingAgent = Car at LeadingSpawnPt,
    with behavior FollowLaneBehavior(target_speed=globalParameters.OPT_LEADING_SPEED)

# Identifying the adjacent lane to the left for the Adversarial Agent and setting its spawn point further in front
param OPT_GEO_Y_DISTANCE = Range(0, 30)
advLane = network.laneSectionAt(ego)._laneToLeft.lane
IntSpawnPt = OrientedPoint following roadDirection from egoSpawnPt for globalParameters.OPT_GEO_Y_DISTANCE
projectPt = Vector(*advLane.centerline.project(IntSpawnPt.position).coords[0])
advHeading = advLane.orientation[projectPt]

# Spawn the Adversarial Agent
AdvAgent = Car at projectPt,
    with heading advHeading,
    with regionContainedIn None,
    with behavior AdvBehavior()