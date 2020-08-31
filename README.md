# How To: Run multiple Workers

## Step 1: Set the number of NRP Backends
Go into the *custom_nrp_installer.sh* script and edit **line 416** to a number greater than zero.
This will set-up as many NRP-Backends as specified.
## Step 2: Install NRP with the custom installer script
> ./custom_nrp_installer.sh install

This will pull the docker images for frontend and backend and copy experiments and model files to the NRP-backends. 

The frontend will be installed on 172.19.0.2:9000 and the backends will start with the IP 172.19.0.3:8080 (nrp0) and the IP increases with each added container.
## Step 3: Edit the frontend service discovery
One manual step is required: If the install succeded, in the end you will be asked to edit the service discovery configuration of the frontend container.
To do this, go into the folder of the installer script and find the *custom_nrp_config.json*. Copy the content. 
Then, connect to the frontend container, e.g. with VS Code, or on command line with 
> docker exec -it frontend bash 

Find the service-discovery config under *home/bbpnrsoa/nrp/src/nrpBackendProxy/config.json* and replace the *"server"* object with the content of *custom_nrp_config.json*. 

It should then look similar to this (for 3 backends):

```sh
{
  "refreshInterval": 5000,
  "auth": {
    "renewInternal": 600000,
    "clientId": "0dcfb392-32c7-480c-ae18-cbaf29e8a6b1",
    "clientSecret": "<client_oidc_secret>",
    "url": "https://services.humanbrainproject.eu/oidc",
    "deactivate": true
  },
  "port": 8443,
  "modelsPath": "$HBP/Models",
  "experimentsPath": "$HBP/Experiments",
  "servers": { 
    "172.19.0.3": { "gzweb": { "assets": "http://172.19.0.3:8080/assets", "nrp-services": "http://172.19.0.3:8080", "videoStreaming": "http://172.19.0.3:8080/webstream/", "websocket": "ws://172.19.0.3:8080/gzbridge" }, "rosbridge": { "websocket": "ws://172.19.0.3:8080/rosbridge" }, "serverJobLocation": "local" }, 
    "172.19.0.4": { "gzweb": { "assets": "http://172.19.0.4:8080/assets", "nrp-services": "http://172.19.0.4:8080", "videoStreaming": "http://172.19.0.4:8080/webstream/", "websocket": "ws://172.19.0.4:8080/gzbridge" }, "rosbridge": { "websocket": "ws://172.19.0.4:8080/rosbridge" }, "serverJobLocation": "local" }, 
    "172.19.0.5": { "gzweb": { "assets": "http://172.19.0.5:8080/assets", "nrp-services": "http://172.19.0.5:8080", "videoStreaming": "http://172.19.0.5:8080/webstream/", "websocket": "ws://172.19.0.5:8080/gzbridge" }, "rosbridge": { "websocket": "ws://172.19.0.5:8080/rosbridge" }, "serverJobLocation": "local" }},
  "storage": "FS",
  "authentication": "FS",
  "backendScripts": {
    "restart-backend":
      "$HBP/user-scripts/config_files/nrpBackendProxy/restart-backend.sh"
  },
  "activity-logs": {
    "localfile": "/tmp/nrp_activity.log"
  }
}
```
Restart all containers to apply the changes made with 
> ./custom_nrp_installer.sh restart


Now, the container should be active and running.

You can check if all services are discovered correctly by entering one of the backend containers
```sh
$ docker exec -it nrp0 bash
$ cle-virtual-coach python
$ from hbp_nrp_virtual_coach.virtual_coach import VirtualCoach
$ vc = VirtualCoach(environment='local', storage_username='nrpuser', storage_password='password')
$ vc.print_available_servers()
```
## Step 4: Install the Distributed Reinforcement Learning (DRL) files
The last setup-step is to copy the custom experiment and worker files to the respective backends. To do this, use the command

> ./custom_nrp_installer.sh install_drl

This may take a while, since it also has to individually install all python requirments (e.g. pytorch).

It will also install the requirements for the local learner.

## Step 5: Start the DRL-Experiment
Start all robots with

> ./custom_nrp_installer.sh start_experiment

This will open an individual terminal for the central learner and for each nrp-backend.
You can then monitor the experiment in the terminals or viewing the simulation in the standard NRP-website on [172.19.0.2:9000/#/esv-private](172.19.0.2:9000/#/esv-private).

Close any experiment/window with CTRL-C. Restart / stop everything with 
> ./custom_nrp_installer.sh restart

or
> ./custom_nrp_installer.sh stop

to make sure the IPs and NRP-Backendservers are not occupied by any containers not closed properly.

# Distributed Reinforcement Learning Agent

## Installing packages with pip

After initializing your backend container, enter the container, copy the repository and install the required packages:

```bash
docker cp . nrp:/home/bbpnrsoa/distributed-reinforcement-learning 
docker exec -it nrp bash
cd /home/bbpnrsoa/distributed-reinforcement-learning
pip install -r requirements.txt --no-cache-dir
```
**Note**: If you run the learner locally, you also have to install the packages in your local environment. (Check if you run Python 2.7. on your local machine, not Python3!) 

## Socket Inits:
Configure your IPs! 
The IP in `SERdemo1Learner/Learner_v1` is the Server-IP and must be the same as IP in `SERdemo1/Worker_v1`. 

Either work inside the Docker-Container and use localhost 127.0.0.1
Or have Worker inside the container and attach the Learner to the Docker-Host. Use 
```bash
docker network inspect nrpnet
```
 to find out the IP of your Docker-Host (e.g 172.19.0.1).


## Import experiment files
To load the experiment in your installation of the NRP (either local or source install), open the
NRP frontend in your browser at location http://host.docker.internal:9000/#/esv-private, and then navigate to the 'My experiments' tab. There, click on the 'Import folder'
button, and select the `SERdemo1/experiment/ser_rl_ss20` folder to upload. After that, the experiment should
appear under 'My experiments', and you should be able to launch it.


## Start Reinforcement Learning:
1. Start Learner either in the container:
```bash
docker exec -it nrp bash
cd /home/bbpnrsoa/distributed-reinforcement-learning
python SERdemo1Learner/Learner_v1.py
```
or start the learner locally:
```bash
python SERdemo1Learner/Learner_v1.py
```
You should see
```bash
('listening on', ('your IP address', 65432))
```
in your console now.

2. Launch the experiment in the NRP frontend. Once the simulation has started, click the play button at the top left to start the robot.

3. Start another bash window, access the container and start the Worker in the container:
```bash
docker exec -it nrp bash
cd /home/bbpnrsoa/distributed-reinforcement-learning
python SERdemo1/Worker_v1.py
```
After the robot inital commands you should see
```bash
('starting connection to', ('your IP address', 65432))
```
in your worker terminal and 
```bash
('accepted connection from', ('your IP address', 34928))
```
in your learner terminal.
## Experiment Setup
The environment consists of one Hollie Robot arm with six Degrees of Freedom, sitting on a table. A
Schunk hand is mounted on the arm, but it is of little relevance to the task to be solved. There is
also one blue cylinder on the table.

<img src="SERdemo1/experiment/ser_rl_ss20/ExDDemoManipulation.png" width="400" />

The task is for the arm to move towards the cylinder and knock it off the table. The observations at
each time step are: 
* The current joint positions (six dimensional)
* The current object pose (position and orientation) (seven dimensional)

and the actions to be taken are:
* The six joint positions

A position controller takes care of moving the joints to the desired positions.

# Besides reinforcement learning 

## Interacting with the Simulation
After launching the experiment and clicking on the play button, you can interact with the simulation
from a python shell though the Agent class in 'experiment_api.py'. It is better to do this within
the docker container, because you might need to install additional dependencies if you want to run 
it on your system. Below are the steps for interacting with the simulation from within the docker 
container:

0. Move to the experiment folder
```
$ cd SERdemo1/experiment/ser_rl_ss20
```

1. Copy the experiment_api.py file to the backend container:
```
$ docker cp experiment_api.py nrp:/home/bbpnrsoa/
```

2. Access the backend docker container:
```
$ docker exec -it nrp bash
```

3. Open a python shell inside the backend container and import the experiment api:
```
$ cd ~

$ python

>>> import experiment_api
```

4. Instantiate the agent class and explore the available functions:
```
>>> agent = experiment_api.Agent()
>>> agent.get_current_state()
>>> agent.act(1, 1, 1, 1, 1, 1)
>>> agent.reset()
```

Feel free to extend the experiment_api.py with functions that you see necessary.
