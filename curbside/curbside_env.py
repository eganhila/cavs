from flow.envs.base import Env
from flow.core import rewards
from gym.spaces.box import Box

import numpy as np
import collections

ADDITIONAL_ENV_PARAMS = {
    # maximum acceleration for autonomous vehicles, in m/s^2
    "max_accel": 3,
    # maximum deceleration for autonomous vehicles, in m/s^2
    "max_decel": 3,
    # maximum number of controllable vehicles in the network
    "num_rl": 5,
}
state_translation = {"inflow":0,"slowing":1,"parking":2,"parked":3, "outflow":4}

class CurbsideEnv(Env):

	def __init__(self, env_params, sim_params, network, simulator='traci'):
	    for p in ADDITIONAL_ENV_PARAMS.keys():
	        if p not in env_params.additional_params:
	            raise KeyError(
	                'Environment parameter "{}" not supplied'.format(p))

	    # maximum number of controlled vehicles
	    self.num_rl = env_params.additional_params["num_rl"]


	    super().__init__(env_params, sim_params, network, simulator)

	@property
	def action_space(self):
	    """See class definition."""
	    return Box(
	        low=-abs(self.env_params.additional_params["max_decel"]),
	        high=self.env_params.additional_params["max_accel"],
	        shape=(self.num_rl, ),
	        dtype=np.float32)

	@property
	def observation_space(self):
	    """See class definition."""
	    return Box(low=-1, high=1, shape=(4 * self.num_rl, ), dtype=np.float32)

	def _apply_rl_actions(self, rl_actions):
		"""See class definition."""
		for i,rl_id in enumerate(self.k.vehicle.get_rl_ids()):
			self.k.vehicle.apply_acceleration(rl_id, rl_actions[i])
			target_lane = self.k.vehicle.get_lane_changing_controller(
							rl_id).get_action(self)
			self.k.vehicle.apply_lane_change(rl_id,direction=target_lane)

	def get_state(self, rl_id=None, **kwargs):
		max_speed = self.k.network.max_speed()*3
		max_length = self.k.network.length()
		L_p = self.network.net_params.additional_params["length_parking"] 
		L_i = self.network.net_params.additional_params["length_inflow"] 
		L_o = self.network.net_params.additional_params["length_outflow"] 
		N_p = self.network.net_params.additional_params["number_parking_zones"]
		L = L_p + L_i + L_o

		observation = [0 for _ in range(4 * self.num_rl)]

		for i, rl_id in enumerate(self.k.vehicle.get_rl_ids()):

			# OBSERVATION 1
			# get distance to parking spot normalized by length of parking area

			l = self.k.vehicle.get_distance_to_pzone(rl_id, self)
			N = self.k.vehicle.get_pzone(rl_id, self)
			x = self.k.vehicle.get_global_position(rl_id, self) 

			# OBSERVATION 2
			# get current status (0 = not yet parked, 1=parking completed)
			state = state_translation[self.k.vehicle.get_state(rl_id)]
			state = state in [3,4]

			# OBSERVATION 3
			# get current speed normalized by max speed
			this_speed = self.k.vehicle.get_speed(rl_id)

			observation[5 * i + 0] = x / L
			observation[5 * i + 1] = l / L
			observation[5 * i + 2] = N / N_p
			observation[5 * i + 3] = (this_speed) / max_speed
			observation[5 * i + 4] = state / 4 # number of states


		return observation

	def compute_reward(self, rl_actions, **kwargs):
		"""See class definition."""
		L_p = self.network.net_params.additional_params["length_parking"] 

		if rl_actions is None:
			return 0
		
		if kwargs["fail"]:
			return 0

		reward = 0
		cost = 0

		for rl_id in self.k.vehicle.get_rl_ids():

			edge = self.k.vehicle.get_edge(rl_id)
			pzone = self.k.vehicle.get_pzone(rl_id,self)
			state = self.k.vehicle.get_state(rl_id)

			if state in ['parked', "outflow"]:
				reward += self.k.vehicle.get_speed(rl_id)				
			else:
				if edge == f"parking_{pzone}": 
					reward += 1

			cost += 1


		# weights for cost1, cost2, and cost3, respectively
		eta_cost, eta_reward = 1e-4, 1e-2

		total = eta_reward*reward - eta_cost*cost #eta1 * cost1 + eta2 * cost2 #+ eta3*cost3
		return total

