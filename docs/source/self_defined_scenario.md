# How to Create Scenario

We provide step-by-step instructions on how to create your own (perception / planning) scenarios to be used in Safebench.

## Create Perception Scenario

There are two steps to create a new scenario with static objects and dynamic pedestrians. 
**Notice that the scenarios in perception tracks only differ in the texture, so the route and planning parts will be fixed.**
<!-- The first step is to define the route file, scenario type file, and scenario config file. -->
The first step is to implement the scenario class, which create actors, define initial behavior of actors, and create static objects.
The second step is to implement the scenario policy, which controls the behavior of the actors if necessary.


### Step 1: Implement Scenario Class
All scenarios need to inherit the `BasicScenario` defined in `safebench/scenario/scenario_definition/base_scenario.py`.
One scenario should create actors in the scenario and update the behavior of the actors at each time step.
To create a new scenario policy, you need to inherit `BasePolicy` and implement the following 6 functions.

**1. Init policy with parameters.**
```python
    def __init__(self, config, logger):
```
You can use the config to initialize the scenario. The logger is used to print the information of the scenario.

**2. Create initial behavior of actors.**
```python
def create_behavior(self, scenario_init_action):
```
Scenarios usually contains a initial action to create the initial behavior of the actors. You can use this function to create the initial behavior of the actors.

**3. Update behavior of actors.**
```python
def update_behavior(self, scenario_action):
```
This function updates the behavior of the actors. The scenario_action is the action from the scenario policy.

**4. Initialize actors in the scenario.**
```python
def initialize_actors(self):
```
This function creates the actors in the scenario with the behavior (e.g., positions) created in function `create_behavior`.

**5. Define conditions to stop scenario.**
```python
def check_stop_condition(self):
```
You can define your own stop condition to stop the scenario. If this is left blank, the scenario will be stopped after some pre-defined conditions, e.g., the number of time steps and collision dection.


### step 2 (Optional): Implement Texture Attacker (Scenario)

Your scenarios may contain a policy to cotnrol the behavior of the actors. If so, you need to implement a scenario policy. If not, you can just use `DummyPolicy`.
We provide a `BasePolicy` as a template for implementing policy, which is stored in file `safebench/scenario/scenario_policy/base_policy.py`.
To create a new scenario policy, you need to inherit `BasePolicy` and implement the following 6 functions.

**1. Init policy with parameters.**
```python
    def __init__(self, config, logger):
```
You can use the config to initialize the scenario. The logger is used to print the information of the scenario.

**2. Train models in policy.**
```python
    def train(self, replay_buffer):
```
This function takes the replay_buffer as input and train the models. To get samples from the replay_buffer, you should call replay_buffer.sample() to get the dictionary of all data.

**3. Set train or eval mode.**
```python
    def set_mode(self, mode):
```
To switch the mode of the models (e.g., neural networks), you should implement this function. You can leave this function blank if your models do not have a mode.

**4. Get the generated textures, via the action from scenario policy.**
```python
    def get_action(self, state, infos, deterministic):
```
This function takes the prediction of the object detection as input and return the adversarial textures that you generated. 
The information of state can be found in the code environment.

**5. Load model from file.**
```python
    def load_model(self):
```
This function loads the model from file. You can leave this function blank if your models do not need to be loaded from file.

**6. Save model to file.**
```python
    def save_model(self):
```
This function saves the model to file. You can leave this function blank if your models do not need to be saved to file.



## Create Planning Scenario
There are three steps to create a new scenario with static objects and dynamic actors.
The first step is to define the route file, scenario type file, and scenario config file.
The second step is to implement the scenario class, which create actors, define initial behavior of actors, and create static objects.
The third step is to implement the scenario policy, which controls the behavior of the actors if necessary.


### Step 1: Define Route and Scenario Config
Firt, you should define the routes of the scenario in `safebench/scenario/scenario_data/route`. 
Then, you should define the combination of scenarios and route by writing a json file in `safebench/scenario/config/scenario_type`. Each item in this file represent a specific scenario.
Finally, a configuration file of your scenario should be written in `safebench/scenario/config`. This file contains the configuration of the scenario, e.g., the number of actors, the number of time steps, the number of static objects, etc.


### Step 2: Implement Scenario Class
All scenarios need to inherit the `BasicScenario` defined in `safebench/scenario/scenario_definition/base_scenario.py`.
One scenario should create actors in the scenario and update the behavior of the actors at each time step.
To create a new scenario policy, you need to inherit `BasePolicy` and implement the following 6 functions.

**1. Init policy with parameters.**
```python
    def __init__(self, config, logger):
```
You can use the config to initialize the scenario. The logger is used to print the information of the scenario.

**2. Create initial behavior of actors.**
```python
def create_behavior(self, scenario_init_action):
```
Scenarios usually contains a initial action to create the initial behavior of the actors. You can use this function to create the initial behavior of the actors.

**3. Update behavior of actors.**
```python
def update_behavior(self, scenario_action):
```
This function updates the behavior of the actors. The scenario_action is the action from the scenario policy.

**4. Initialize actors in the scenario.**
```python
def initialize_actors(self):
```
This function creates the actors in the scenario with the behavior (e.g., positions) created in function `create_behavior`.

**5. Define conditions to stop scenario.**
```python
def check_stop_condition(self):
```
You can define your own stop condition to stop the scenario. If this is left blank, the scenario will be stopped after some pre-defined conditions, e.g., the number of time steps and collision dection.


### step 3 (Optional): Implement Scenario Policy 

Your scenarios may contain a policy to cotnrol the behavior of the actors. If so, you need to implement a scenario policy. If not, you can just use `DummyPolicy`.
We provide a `BasePolicy` as a template for implementing policy, which is stored in file `safebench/scenario/scenario_policy/base_policy.py`.
To create a new scenario policy, you need to inherit `BasePolicy` and implement the following 6 functions.

**1. Init policy with parameters.**
```python
    def __init__(self, config, logger):
```
You can use the config to initialize the scenario. The logger is used to print the information of the scenario.

**2. Train models in policy.**
```python
    def train(self, replay_buffer):
```
This function takes the replay_buffer as input and train the models. To get samples from the replay_buffer, you should call replay_buffer.sample() to get the dictionary of all data.

**3. Set train or eval mode.**
```python
    def set_mode(self, mode):
```
To switch the mode of the models (e.g., neural networks), you should implement this function. You can leave this function blank if your models do not have a mode.

**4. Get action from policy.**
```python
    def get_action(self, state, infos, deterministic):
```
This function takes the state as input and return the action. If the deterministic is True, the action should be deterministic.
The information of state can be found in the code environment.

**5. Load model from file.**
```python
    def load_model(self):
```
This function loads the model from file. You can leave this function blank if your models do not need to be loaded from file.

**6. Save model to file.**
```python
    def save_model(self):
```
This function saves the model to file. You can leave this function blank if your models do not need to be saved to file.

