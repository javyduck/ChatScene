<h2 align="center"><a href="https://arxiv.org/abs/2405.14062" style="color:#903168">
ChatScene: Knowledge-Enabled Safety-Critical Scenario Generation for Autonomous Vehicles</a></h3>

## Installation

This repository utilizes [Safebench](https://github.com/trust-ai/SafeBench) and [Scenic](https://github.com/BerkeleyLearnVerify/Scenic), we acknowledge and thank these projects for their contributions!

**Recommended system: Ubuntu 20.04 or 22.04**

### 1. Local Installation

Step 1: Setup conda environment

```
conda create -n chatscene python=3.8
conda activate chatscene
```

Step 2: Clone this git repo in an appropriate folder

```
git clone https://github.com/javyduck/ChatScene.git
```

Step 3: Enter the repo root folder and install the packages:

```
cd ChatScene
pip install -r requirements.txt
pip install -e .
```

Step 4: Install the Scenic package:

```
cd Scenic
python -m pip install -e .
```

Step 5: Download our [CARLA_0.9.13](https://drive.google.com/file/d/139vLRgXP90Zk6Q_du9cRdOLx7GJIw_0v/view?usp=sharing) and extract it to your folder.

Step 6: Run `sudo apt install libomp5` as per this [git issue](https://github.com/carla-simulator/carla/issues/4498).

Step 7: Add the python API of CARLA to the ```PYTHONPATH``` environment variable. You can add the following commands to your `~/.bashrc`:

```
export CARLA_ROOT={path/to/your/carla}
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.13-py3.8-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/agents
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
```

Then, do `source ~/.bashrc` to update the environment variable.

## CARLA Setup

### 1. Desktop Users

Enter the CARLA root folder, launch the CARLA server and run our platform with

```
# Launch CARLA
./CarlaUE4.sh -prefernvidia -windowed -carla-port=2000
```

### 2. Remote Server Users

Enter the CARLA root folder, launch the CARLA server with headless mode, and run our platform with

```
# Launch CARLA
./CarlaUE4.sh -prefernvidia -RenderOffScreen -carla-port=2000
```

(Optional) You can also visualize the pygame window using [TurboVNC](https://sourceforge.net/projects/turbovnc/files/).
First, launch CARLA with headless mode, and run our platform on a virtual display.

```
# Launch CARLA
./CarlaUE4.sh -prefernvidia -RenderOffScreen -carla-port=2000

# Run a remote VNC-Xserver. This will create a virtual display "8".
/opt/TurboVNC/bin/vncserver :8 -noxstartup
```

You can use the TurboVNC client on your local machine to connect to the virtual display.

```
# Use the built-in SSH client of TurboVNC Viewer
/opt/TurboVNC/bin/vncviewer -via user@host localhost:n

# Or you can manually forward connections to the remote server by
ssh -L fp:localhost:5900+n user@host
# Open another terminal on local machine
/opt/TurboVNC/bin/vncviewer localhost::fp
```

where `user@host` is your remote server, `fp` is a free TCP port on the local machine, and `n` is the display port specified when you started the VNC server on the remote server ("8" in our example).

## ChatScene

In ChatScene, we ensure a fair comparison with the baselines by using the same eight scenarios, sampling five behaviors for each scenario from the database. The corresponding generated comeplete Scenic files, with some modifications, have been provided in `safebench/scenario/scenario_data/scenic_data` (with some manual modifications to use the same fixed 10 routes for the ego agent to ensure fair comparison with the baselines).

The ego agent is controlled by a default RL model, while the surrounding adversarial agent is controlled by Scenic.

The agent configuration is provided in `safebench/agent/config/adv_scenic.yaml`. By default, it loads a pretrained RL model from Safebench-v1.

### Modes in ChatScene:

1. **train_scenario**: Select the most challenging scenes for the same behavior under the same scenario.

   Configuration can be found in `safebench/scenario/config/train_agent_scenic.yaml`.

   The `sample_num = 50, opt_step = 10, select_num = 2` settings in the file mean we sample 50 scenes and select the 2 most challenging ones for evaluation. The default setting is to choose scenes that lead to a collision of the ego agent and provide the lowest overall score. We optimize the range of parameters, like speed, every 10 steps based on collision statistics from previously sampled scenes.

   Example command for optimizing the scene:

   ```
   python scripts/run_train.py --agent_cfg=adv_scenic.yaml --scenario_cfg=train_scenario_scenic.yaml --mode train_scenario --scenario_id 1
   ```

   Use the following command if you are using a TurboVNC client on your local machine to connect to the virtual display:

   ```
   DISPLAY=:8 python scripts/run_train.py --agent_cfg=adv_scenic.yaml --scenario_cfg=train_scenario_scenic.yaml --mode train_scenario --scenario_id 1
   ```

   The IDs for the final selected scenes will be stored in `safebench/scenario/scenario_data/scenic_data/scenario_1/scenario_1.json`.

2. **train_agent**: Train the agent based on the selected challenging scenes:

   ```
   python scripts/run_train.py --agent_cfg=adv_scenic.yaml --scenario_cfg=train_agent_scenic.yaml --mode train_agent --scenario_id 1
   ```

   We have a total of 10 routes for each scenario. We use the first 8 for training and the last 2 for testing (route IDs: `[0,1,2,3,4,5,6,7]`). The configuration, including `scenario_1.json`, will train the agent based on the most challenging scenes (the ones leading to a collision of the ego agent).

3. **eval**: Evaluate the trained agent on the last 2 routes (route IDs: `[8,9]`), the `test_epoch` is for loading a finetuned model after a specific training epoch:

   ```
   python scripts/run_eval.py --agent_cfg=adv_scenic.yaml --scenario_cfg=eval_scenic.yaml --mode eval --scenario_id 1 --test_epoch -1
   ```

The `-1` here is for loading our provided fine-tuned agent in each scenario based on our Scenic scenarios in `safebench/agent/model_ckpt/adv_train/sac/scenic/scenario_1/model.sac.-001.torch`.

## Coming Soon: Dynamic Mode

The above part ensures using the same scenario and routes for fair comparison with baselines. However, ChatScene can generate scenarios and scenes freely without any constraints. Simply provide a text description, such as "*The ego vehicle is driving on a straight road; the adversarial pedestrian suddenly crosses the road from the right front and suddenly stops in front of the ego.*" is enough for the training. We are currently integrating our database with GPT-4o for generating more diverse scenarios based on our pre-built retrieval database, and will upload both soonly.

- [ ] Integrate GPT-4o with our retrieval database and commit the dynamic mode. ETA: within one week. :-)
- [ ] Finetune an LLM for generating snippets end-to-end based on the data constructed from our database.

If you have any questions, please open an issue or email [jiaweiz7@illinois.edu](mailto:jiaweiz7@illinois.edu). We aim to resolve your issues as soon as possible!
