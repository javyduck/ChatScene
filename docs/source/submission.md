<!--
 * @Date: 2023-02-20 20:13:41
 * @LastEditTime: 2023-03-27 12:15:34
 * @Description: 
-->

<!-- # Hosting a Challenge via EvalAI

This document provides an overview on how to host a code-upload based challenge on EvalAI. A code-upload based challenge is usually a reinforcement learning challenge in which participants upload their trained model in the form of a Docker image. The environment is also a docker image.

Info below extracted from the [EvalAI Documentation](https://evalai.readthedocs.io/en/latest/).

<br>

## Step 1: Set Up Main Repo
Clone this [EvalAI-Starter Repo](https://github.com/Cloud-CV/EvalAI-Starters) as a template. For info on how to use a repo as a template, see [this](https://docs.github.com/en/free-pro-team@latest/github/creating-cloning-and-archiving-repositories/creating-a-repository-from-a-template).

<br>

## Step 2: Set Up Challenge Configuration
Open the "challenge_config.yml" in the repo. Update the values of the features in the file based on the characteristics of the challenge. More info about the features can be found [here](https://evalai.readthedocs.io/en/latest/configuration.html).

Note that the following two features have to have the following values:

1) remote_evaluation: True
2) is_docker_based: True


For evaluation to be possible, an [AWS Elastic Kubernetes Service (EKS)](https://aws.amazon.com/eks/) cluster might need to be created. The following info is needed:

1) aws_account_id
2) aws_access_key_id
3) aws_secret_access_key
4) aws_region

This info needs to emailed to team@cloudcv.org, who will set up the infrastructure in the AWS account.

<br>

## Step 3: Define Evaluation Code
A evaluation file needs to be created to determine which metrics will be determined at which phase. This will also evalute the participants' submissions and post a score to the leaderboard. The environment image should be created by the host and the agent image should be pushed by the participants.

The overall structure of the evaluation code is fixed for architectural reasons.

To define the evaluation code:

1) Open the environment.py file located in EvalAI-Starters/code_upload_challenge_evaluation/environment/.
2) Edit the evaluator_environment class.

    ```
    class evaluator_environment:
        def __init__(self, environment="CartPole-v0"):
            self.score = 0
            self.feedback = None
            self.env = gym.make(environment)
            self.env.reset()

        def get_action_space(self):
            return list(range(self.env.action_space.n))

        def next_score(self):
            self.score += 1
    ```

    There are three methods:

    a) \_\_init\_\_: initialization method<br>
    b) get_action_space: returns the action space of the agent in the environment<br>
    c) next_score: returns or updates the reward achieved<br>

Additional methods can be added as need be.

3) Edit the Environment class in environment.py.
    ```
    class Environment(evaluation_pb2_grpc.EnvironmentServicer):
        def __init__(self, challenge_pk, phase_pk, submission_pk, server):
            self.challenge_pk = challenge_pk
            self.phase_pk = phase_pk
            self.submission_pk = submission_pk
            self.server = server

        def get_action_space(self, request, context):
            message = pack_for_grpc(env.get_action_space())
            return evaluation_pb2.Package(SerializedEntity=message)

        def act_on_environment(self, request, context):
            global EVALUATION_COMPLETED
            if not env.feedback or not env.feedback[2]:
                action = unpack_for_grpc(request.SerializedEntity)
                env.next_score()
                env.feedback = env.env.step(action)
            if env.feedback[2]:
                if not LOCAL_EVALUATION:
                    update_submission_result(
                        env, self.challenge_pk, self.phase_pk, self.submission_pk
                    )
                else:
                    print("Final Score: {0}".format(env.score))
                    print("Stopping Evaluation!")
                    EVALUATION_COMPLETED = True
            return evaluation_pb2.Package(
                SerializedEntity=pack_for_grpc(
                    {"feedback": env.feedback, "current_score": env.score,}
                )
            )
     ```
 
     [gRPC](https://grpc.io/) servers are used to get actions in the form of messages from the agent container. This class can be edited to fit the needs of the current challenge. Seriailzation and deserialization of the messages to be sent across gRPC is needed. The following two methods may be helpful for this:

     a) unpack_for_gprc: this method deserializes entities from request/response sent over gRPC. This is useful for receiving messages (for example, actions from the agent).<br>
     b) pack_for_gprc: this method serializes entities to be sent over a request over gRPC. This is useful for sending messages (for example, feedback from the environment).<br>
 
4) Edit the requirements file based on the required packages for the environment.
5) Edit environment Dockerfile located in EvalAI-Starters/code_upload_challenge_evaluation/docker/environment/ if need be.
6) Fill in the docker enviroment variables in docker.env located in EvalAI-Starters/code_upload_challenge_evaluation/docker/environment/:

    ```
    AUTH_TOKEN=x
    EVALAI_API_SERVER=https://eval.ai
    LOCAL_EVALUATION = True
    QUEUE_NAME=x
    ```
7) Create a docker image on upload on Amazon Elastic Container Registry (ECR). More info on pushing a docker image to ECR can be found [here](https://docs.aws.amazon.com/AmazonECR/latest/userguide/docker-push-ecr-image.html).

    ```
    docker build -f <file_path_to_Dockerfile>

    aws ecr get-login-password --region <region> | docker login --username AWS --password-stdin <aws_account_id>.dkr.ecr.<region>.amazonaws.com
    docker tag <image_id> <aws_account_id>.dkr.ecr.<region>.amazonaws.com/<my-repository>:<tag>
    docker push <aws_account_id>.dkr.ecr.<region>.amazonaws.com/<my-repository>:<tag>
    ```

8) Add environment image to challenge configuration for challenge phase. For each challenge phase, add the link to the environment image.
    ```
    ...
    challenge_phases:
        - id: 1
        ...
        - environment_image: <docker image uri>
    ...
    ```


9) Create a starter example for creating the agent: the participants are expected to create a docker image with the policy and methods to interact with the environment. To create the agent environment:

    a) Create the starter script. A template, agent.py, is provided in EvalAI-Starters/code_upload_challenge_evaluation/agent/.
    ```
    import evaluation_pb2
    import evaluation_pb2_grpc
    import grpc
    import os
    import pickle
    import time

    time.sleep(30)

    LOCAL_EVALUATION = os.environ.get("LOCAL_EVALUATION")

    if LOCAL_EVALUATION:
        channel = grpc.insecure_channel("environment:8085")
    else:
        channel = grpc.insecure_channel("localhost:8085")

    stub = evaluation_pb2_grpc.EnvironmentStub(channel)

    def pack_for_grpc(entity):
        return pickle.dumps(entity)

    def unpack_for_grpc(entity):
        return pickle.loads(entity)

    flag = None

    while not flag:
        base = unpack_for_grpc(
            stub.act_on_environment(
                evaluation_pb2.Package(SerializedEntity=pack_for_grpc(1))
            ).SerializedEntity
        )
        flag = base["feedback"][2]
        print("Agent Feedback", base["feedback"])
        print("*"* 100)
    ```
    
    b) Edit requirements.txt located in EvalAI-Starters/code_upload_challenge_evaluation/requirements based on package requirements.<br>
    c) Edit the Dockerfile (if need be) located in EvalAI-Starters/code_upload_challenge_evaluation/docker/agent/ which will interact run agent.py to interact with the environment.<br>
    d) Edit docker.env located in EvalAI-Starters/code_upload_challenge_evaluation/docker/agent/ to be:<br>
    
    ```
    LOCAL_EVALUATION = True
    ```

<br>

## Step 4: Edit Challenge HTML Templates
Update the HTML templates in EvalAI-Starters/templates. The submission-guidelines.html should be detailed to ensure participants can upload their submissions. The participants are expected to submit links to their docker images using evalai-cli (more info [here](https://cli.eval.ai/)). The command is:
```
evalai push <image>:<tag> --phase <phase_name>
```
<br>
At this point, the challenge configuration has been submitted for review and the EvalAI team has been notified. They will review and approve the challenge.
<br> -->

# How to Submit Results

## General Submission via Docker
TBD

## Perception Track
- In perception attack tracks, the participants are required to submit a file named after `stopsign.jpg` and `car.jpg` in `safebench/scenario/scenario_data/submission/`.
- In perception defense tracks, the participants are required to submit an object detection script in `safebench/agent/object_detection/submission.py`, where we've already provided a template and some example detection model like YOLO-v5 and Faster-RCNN. You should include any utility function and checkpoints you need in the `agent/object_detection/` folder.

During the evaluation, only two folders (`safebench/agent/object_detection/` and `safebench/scenario/scenario_data/submission`) will be kept, while the rest of the code will be replaced by our evaluation template. So please make sure to **include all the codes and files** you need in these two folders.

After the evaluation finishes, you will get a feedback as the mean Average Precision

## Planning Track
TBD