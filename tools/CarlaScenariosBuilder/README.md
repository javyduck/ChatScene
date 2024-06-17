# Creating and Editing Routes
To create or edit route waypoints, use the following command.
```
python create_routes.py

optional arguments:
  --map                                   Map name
  --scenario                              Scenario
  --route                                 Route index
  --road {auto, intersection, straight}   Create routes based on a intersection or a straight road.
```

# Creating and Editing Scenarios
To create the trigger point and spawn points for other actors in the scenario, execute the following command.
```
python create_scenario.py

optional arguments:
  --map                                   Map name
  --scenario                              Scenario
  --scenario_idx                          Scenario index
  --road {auto, intersection, straight}   Create routes based on a intersection or a straight road.
```

# Export Routes and Scenarios to Safebench
After creating routes and scenarios, please execute the following command to export them to Safebench.
```
python export.py

optional arguments:
  --map                                   Map name
  --save_dir SAVE_DIR                     Output directory
  --scenario SCENARIO                     Scenario index. If the value is negative, the program will automatically 
                                          export all scenarios. Default value is -1.
```

# Visualize Routes
To view the routes visually, use the following command.
```
python visualize_routes.py

optional arguments:
  --save_dir                              Output directory
  --scenario                              Scenario index
```

# Visualize Scenarios
To view the trigger point and spawn points for other actors in a scenario, use the following command.
```
python visualize_scenarios.py

optional arguments:
  --save_dir                              Output directory
  --scenario                              Scenario index
```

# Export Map Waypoint
The program requires map waypoints to create routes and scenarios. If the waypoints have not been exported, please execute the following command to export them. 
Remember to rerun this command whenever the map waypoints have been modified.
```
python get_map_data.py

optional arguments:
  --map                                   Map name
```