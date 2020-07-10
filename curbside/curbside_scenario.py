from flow.networks.base import Network
from flow.core.params import InitialConfig
from flow.core.params import TrafficLightParams
from numpy import pi, sin, cos

INFLOW_EDGE_LEN = 200  # length of the inflow edges (needed for resets)
VEHICLE_LENGTH = 5

ADDITIONAL_NET_PARAMS = {
    "lanes": 2,
    "speed_limit": 10,
    "number_parking_zones": 5,
    "length_inflow": 200,
    "length_outflow":100,
    "length_parking": 200,
}


class curbsideNetworkPZones(Network):

    def __init__(self,
                 name,
                 vehicles,
                 net_params,
                 initial_config=InitialConfig(),
                 traffic_lights=TrafficLightParams()):
        """Initialize a merge scenario."""
        for p in ADDITIONAL_NET_PARAMS.keys():
            if p not in net_params.additional_params:
                raise KeyError('Network parameter "{}" not supplied'.format(p))

        super().__init__(name, vehicles, net_params, initial_config,
                         traffic_lights)

    def specify_nodes(self, net_params):
        """See parent class."""
        L_p = net_params.additional_params["length_parking"]
        L_i = net_params.additional_params["length_inflow"]
        L_o = net_params.additional_params["length_outflow"]
        N_p = net_params.additional_params["number_parking_zones"]

        nodes = [
            {
                "id": "inflow",
                "x": 0,
                "y": 0
            }] + [{
                "id":f"parking_{i}",
                "x": L_i + i*L_p/N_p,
                "y": 0

            } for i in range(N_p+1)] + [{
                "id": "outflow",
                "x":L_p+L_i+L_o,
                "y":0
            }]

        return nodes

    def specify_edges(self, net_params):
        """See parent class."""
        L_p = net_params.additional_params["length_parking"]
        L_i = net_params.additional_params["length_inflow"]
        L_o = net_params.additional_params["length_outflow"]
        N_p = net_params.additional_params["number_parking_zones"]

        edges = [{
            "id": "inflow",
            "type": "circulatorType",
            "from": "inflow",
            "to": "parking_0",
            "length": L_i
        }] + [{
            "id": f"parking_{i}",
            "type": "curbType",
            "from": f"parking_{i}",
            "to": f"parking_{i+1}",
            "length": L_p/N_p
        } for i in range(N_p)] + [{
            "id": "outflow",
            "type": "circulatorType",
            "from": f"parking_{N_p}",
            "to": "outflow",
            "length": L_o
        }]

        return edges

    def specify_types(self, net_params):
        """See parent class."""
        lanes = net_params.additional_params["lanes"]
        speed = net_params.additional_params["speed_limit"]

        types = [{
            "id": "curbType",
            "numLanes": lanes,
            "speed": speed
        },
        {
            "id": "circulatorType",
            "numLanes": lanes -1,
            "speed": speed,
        }]

        return types

    def specify_routes(self, net_params):
        """See parent class."""
        N_p = net_params.additional_params["number_parking_zones"]

        rts = {
            "inflow": ["inflow"]+[f"parking_{i}" for i in range(N_p)]+["outflow"],
            "outflow":["outflow"]
        }
        for i in range(N_p):
            rts[f"parking_{i}"] = [f"parking_{j}" for j in range(i,N_p)]+["outflow"]

        return rts

    def specify_edge_starts(self,):
        """See parent class."""
        L_p = self.net_params.additional_params["length_parking"]
        L_i = self.net_params.additional_params["length_inflow"]
        L_o = self.net_params.additional_params["length_outflow"]
        N_p = self.net_params.additional_params["number_parking_zones"]

        edgestarts = [("inflow", 0)] +[(f"parking_{i}", L_i+i*L_p/N_p) for i in range(N_p)]+[("outflow",L_p+L_i)]

        return edgestarts

    def specify_connections(self, net_params):
        """See parent class."""
        N_p = self.net_params.additional_params["number_parking_zones"]
        conn_dic = {"inflow":[{
                        "from": "inflow",
                        "to": "parking_0",
                        "fromLane": 0,
                        "toLane": 1}],
                    f"parking_{N_p-1}":[{
                        "from":f"parking_{N_p-1}",
                        "to":"outflow",
                        "fromLane":1,
                        "toLane":0
                    }]

            }

        return conn_dic
