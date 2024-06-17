# How to Create Agent

We provide step-by-step instructions on how to create your own (perception / planning) agent policy to be trained and evaluated in Safebench.

## Create Perception Agent


**1. Init object detection model with configuration arguments.**
```python
    def __init__(self, config, logger):
```
This function processes the configuration and initialize models in the object detection.

**2. Train perception models with collected data.**
```python
    def train(self, replay_buffer):
```
This function takes the collected data as input and train the models. To get samples from the collected data from simulator, you should call replay_buffer.sample() to get the dictionary of all data.

**3. Set train or eval mode.**
```python
    def set_mode(self, mode):
```
To switch the mode of the models (e.g., neural networks), you should implement this function. You can leave this function blank if your models do not have a mode.

**4. Get action from policy.**
```python
    def get_action(self, state, infos, deterministic):
```
This function takes the state as input and return the predicted bounding box, class, and confidence level at the current observation. 
The information of state can be found in the code environment.

**5. Load model from file.**
```python
    def load_model(self):
```
This function loads the model from file. You can leave this function blank if your models do not need to be loaded from file.

**7. Save model to file.**
```python
    def save_model(self):
```

## Create Planning Agent

We provide a `BasePolicy` as a template for implementing policy, which is stored in file `safebench/agent/base_policy.py`.
To create a new agent policy, you need to inherit `BasePolicy` and implement the following 6 functions.

**1. Init policy with parameters.**
```python
    def __init__(self, config, logger):
```
This function processes the configuration and initialize models in policy.

**2. Train models in policy.**
```python
    def train(self, replay_buffer):
```
This function takes the replay_buffer as input and train the models. To get samples from the replay_buffer, you should call replay_buffer.sample() to get the dictionary of all data.

**3. Set the ego vehile object and global route.**
```python
    def set_ego_and_routes(self, ego_vehicles, routes):
```
The object of ego vehicle is created in gym environment. We allow the users access this object in policy. 
The gloabl routes that the ego vehicle is asked to complete is also provided.
Users can properly uses these two things to design the policy.

**4. Set train or eval mode.**
```python
    def set_mode(self, mode):
```
To switch the mode of the models (e.g., neural networks), you should implement this function. You can leave this function blank if your models do not have a mode.

**5. Get action from policy.**
```python
    def get_action(self, state, infos, deterministic):
```
This function takes the state as input and return the action. 
If the deterministic is True, the action should be deterministic.
The information of state can be found in the code environment.

**6. Load model from file.**
```python
    def load_model(self):
```
This function loads the model from file. You can leave this function blank if your models do not need to be loaded from file.

**7. Save model to file.**
```python
    def save_model(self):
```
This function saves the model to file. You can leave this function blank if your models do not need to be saved to file.
