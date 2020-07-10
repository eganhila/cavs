from flow.controllers import IDMController, GippsController
from flow.controllers.base_lane_changing_controller import BaseLaneChangeController
from flow.controllers.base_routing_controller import BaseRouter
import numpy as np





class curbsideLaneChangeController(BaseLaneChangeController):
    """A lane-changing model used to move vehicles into lane 0."""

    def get_lane_change_action(self, env):

        self.update_state(env)
        current_state = env.k.vehicle.get_state(self.veh_id)

        if current_state == "inflow": lc = 0
        elif current_state == "slowing": lc = -1
        elif current_state == "parking": lc = -1
        elif current_state == "parked": lc = 0
        elif current_state == "outflow": lc = 1

        return lc

    def update_state(self, env):
        last_state = env.k.vehicle.get_state(self.veh_id)
        current_edge = env.k.vehicle.get_edge(self.veh_id)
        current_lane = env.k.vehicle.get_lane(self.veh_id)

        v = env.k.vehicle.get_speed(self.veh_id)
        pzone = env.k.vehicle.get_pzone(self.veh_id, env)

        env.k.vehicle.update_tpark_elapsed(self.veh_id)

        if last_state == "inflow":

            if pzone == 0 or current_edge == f"parking_{pzone-1}":
                current_state = "parking"
            else:
                current_state = "inflow"

        elif last_state == "parking":
            if current_edge == f"parking_{pzone}" and v <1e-3:# and current_lane == 0:
                current_state = "parked"
            #elif current_edge != f"parking_{pzone}":
            #    current_state = 'slowing'
            else:
                current_state = "parking"

        elif last_state == "parked":
            t_elapsed = env.k.vehicle.get_tparking_elapsed(self.veh_id, env)
            t_total = env.k.vehicle.get_tparking_total(self.veh_id, env) / 10 # just for now!!! change this!!!!!!!!!

            if t_elapsed >= t_total:
                current_state = "outflow"
            if v > 1e-3:
                current_state = "parking"
            else:
                current_state = "parked"
        elif  last_state == "outflow": 
            current_state = "outflow"

        env.k.vehicle.set_state(self.veh_id, current_state)


class curbsideRouter(BaseRouter):

    def choose_route(self, env):
        # Not actually route chosing, just looking for parking spaces
        pzone = env.k.vehicle.get_pzone(self.veh_id, env)
        edge = env.k.vehicle.get_edge(self.veh_id)

        Nzones = env.network.net_params.additional_params["number_parking_zones"]
        other_vehicles = list(env.k.vehicle._TraCIVehicle__vehicles.keys())
        other_vehicles.remove(self.veh_id)
        Nlook = 5

        pzone_orig = pzone
        # if we are in the inflow or outflow, just proceed as normal
        if edge in ["inflow", "outflow"] or ":" in edge: return
        # parking edge number we are on 
        Nedge = int(edge.strip("parking_"))

        # if we are nowhere close just proceed
        if pzone > Nedge+5: return

        # if we accidentally passed our parking spot we have to update it
        if pzone < Nedge: pzone = Nedge

        # If our desired spot is not open then we look around +/-
        if self.check_parking_occupied(env, pzone, self.veh_id):

            for i in [-1,1,-2,2,-3,3,-4,4]:

                if pzone + i > Nedge+5: continue
                if pzone + i < Nedge: continue
                if pzone + i >= Nzones-1: continue
                if pzone < 1: continue

                if not self.check_parking_occupied(env, pzone+i, self.veh_id):
                    pzone = pzone + i
                    break

        #env.k.vehicle.set_pzone(self.veh_id, pzone)

    def check_parking_occupied(self, env, Nedge, veh_id):
        edge = f"parking_{Nedge}"
        ids = env.k.vehicle.get_ids_by_edge(edge)
        ids = [i for i in ids if i!=veh_id]
        occupied = sum([l ==0 for l in env.k.vehicle.get_lane(ids)])


        return occupied

 


class curbsideAccelController(IDMController):
    ##### Below this is new code #####


   
        #print(self.veh_id, current_state,  current_edge, pzone)
        

    def get_accel(self,env):

        #env.k.vehicle.choose_route(env)
        self.update_state(env)
        
        current_state = env.k.vehicle.get_state(self.veh_id)

        if current_state in ["inflow", "slowing", "parking"]: a = self.park(env)
        #elif current_state == "slowing": a = self.slow_down(env)
        #elif current_state == "parking": a = self.park(env)
        elif current_state == "parked": a = 0
        elif current_state == "outflow": a = self.drive_normal(env)


        return a

    def park(self,env):
        N_pz = env.network.net_params.additional_params["number_parking_zones"]
        L_total = env.network.net_params.additional_params["length_parking"] 
        L_pz = L_total/N_pz
        L_slow = 5*L_pz


        edge = env.k.vehicle.get_edge(self.veh_id)
        if edge in ['inflow', 'outflow']: return self.drive_normal(env)


        Nedge = int(edge.strip("parking_"))
        x = env.k.vehicle.get_position(self.veh_id)+L_pz*(Nedge-1)
        xpzone = env.k.vehicle.get_pzone(self.veh_id, env)*L_pz
        l = xpzone - x

        a_IDM = self.drive_normal(env)
        a_slow = self.slow_down(env, l)


        if l > L_slow: return a_IDM
        #elif l < L_slow: return a_slow
        else: 
            a = (l*a_IDM+(L_slow-l)*a_slow)/L_slow
            return a

            
    def drive_normal(self,env):
        return super().get_accel(env)

    def slow_down(self,env,h):
        v = env.k.vehicle.get_speed(self.veh_id)
        N_pz = env.network.net_params.additional_params["number_parking_zones"]
        L_pz = env.network.net_params.additional_params["length_parking"]

        lead_vel = 0
        s_star = self.s0 + max(
                0, v * self.T + v * (v - lead_vel) /
                (2 * np.sqrt(self.a * self.b)))

        return self.a * (1 - (v / self.v0)**self.delta - (s_star / h)**2)
        