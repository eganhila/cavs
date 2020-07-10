# FLow imports
from flow.core.params import VehicleParams
from flow.controllers import IDMController
from flow.core.params import SumoCarFollowingParams
from flow.core.params import InFlows
from flow.core.params import NetParams
from flow.core.params import SumoParams, EnvParams, InitialConfig
from flow.core.experiment import Experiment

# custom imports
from curbside_scenario import curbsideNetworkPZones as curbsideNetwork, ADDITIONAL_NET_PARAMS
from curbside_controllers import curbsideAccelController, curbsideLaneChangeController, curbsideRouter
from curbside_env import CurbsideEnv, ADDITIONAL_ENV_PARAMS

from flow.controllers import RLController

import json

import ray
from ray.tune import run_experiments
from ray.tune.registry import register_env

from flow.utils.registry import make_create_env
from flow.utils.rllib import FlowParamsEncoder
from ray.rllib.agents.registry import get_agent_class

def main():
    # Setup vehicles and inflow
    vehicles = VehicleParams()

    vehicles.add(veh_id="rl",
                 acceleration_controller=(RLController, {}),
                 routing_controller=(curbsideRouter, {}),
                 lane_change_controller=(curbsideLaneChangeController, {}),
                 num_vehicles=1)

    # other parameters

    additional_net_params = ADDITIONAL_NET_PARAMS.copy()
    additional_net_params["length_inflow"] = 50
    additional_net_params['number_parking_zones'] = 10

    net_params = NetParams(#inflows=inflow,  # our inflows
                           additional_params=additional_net_params)
    sim_params = SumoParams(render=False,
                             sim_step=0.2,
                             save_render=False,
                            restart_instance=True,
                            seed=10)

    env_params = EnvParams(additional_params=ADDITIONAL_ENV_PARAMS)

    initial_config = InitialConfig(edges_distribution=["inflow"])

    flow_params = dict(
        exp_tag='curbside',
        env_name=CurbsideEnv,
        network=curbsideNetwork,
        simulator='custom_traci',
        sim=sim_params,
        env=env_params,
        net=net_params,
        veh=vehicles,
        initial=initial_config,
    )

    # number of time steps
    HORIZON = 1000
    flow_params['env'].horizon = HORIZON

    # number of parallel workers
    N_CPUS = 4
    # number of rollouts per training iteration
    N_ROLLOUTS = 50

    ray.init(num_cpus=N_CPUS)

    # The algorithm or model to train. This may refer to "
    #      "the name of a built-on algorithm (e.g. RLLib's DQN "
    #      "or PPO), or a user-defined trainable function or "
    #      "class registered in the tune registry.")
    alg_run = "PPO"

    agent_cls = get_agent_class(alg_run)
    config = agent_cls._default_config.copy()
    config["num_workers"] = N_CPUS - 1  # number of parallel workers
    config["train_batch_size"] = HORIZON * N_ROLLOUTS  # batch size
    config["gamma"] = 0.999  # discount rate
    config["model"].update({"fcnet_hiddens": [16, 16]})  # size of hidden layers in network
    config["use_gae"] = True  # using generalized advantage estimation
    config["lambda"] = 0.97
    config["sgd_minibatch_size"] = min(16 * 1024, config["train_batch_size"])  # stochastic gradient descent
    config["kl_target"] = 0.02  # target KL divergence
    config["num_sgd_iter"] = 50  # number of SGD iterations
    config["horizon"] = HORIZON  # rollout horizon

    # save the flow params for replay
    flow_json = json.dumps(flow_params, cls=FlowParamsEncoder, sort_keys=True,
                           indent=4)  # generating a string version of flow_params
    config['env_config']['flow_params'] = flow_json  # adding the flow_params to config dict
    config['env_config']['run'] = alg_run

    # Call the utility function make_create_env to be able to
    # register the Flow env for this experiment
    create_env, gym_name = make_create_env(params=flow_params, version=0)

    # Register as rllib env with Gym
    register_env(gym_name, create_env)

    trials = run_experiments({
    flow_params["exp_tag"]: {
            "run": alg_run,
            "env": gym_name,
            "config": {
                **config
            },
            "checkpoint_freq": 5,  # number of iterations between checkpoints
            "checkpoint_at_end": True,  # generate a checkpoint at the end
            "max_failures": 999,
            "stop": {  # stopping conditions
                "training_iteration": 250,  # number of iterations to stop after
            },
        },
    })

if __name__ == "__main__":
    main()
