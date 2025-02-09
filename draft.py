
import numpy as np


class AgentBase():

    def __init__(self, nb_channels = 8, mlp_units = [16], final_unit="logistic", nb_class=2):

        self.nb_channels = nb_channels
        self.model = "TODO model def sous tensorflow"





class ActuatorAgent(AgentBase):

    def __init__(self, assigned_actuator):
        super(ActuatorAgent, self).__init__()

        self.assigned_actuator = assigned_actuator



class SensorAgent(AgentBase):

    def __init__(self, assigned_env):
        super(SensorAgent, self).__init__()
        self.assign_framework = assigned_env



class HiddenAgent(AgentBase):

    def __init__(self, agent_layer, time_unit = 1):
        super(HiddenAgent, self).__init__()

        self.agent_layer = agent_layer


class MultiAgentBrain():

    def __init__(self,
                 init_agents_composition: dict = {
                                           "sensor:clear-eye":2,
                                           "sensor:blur-eye": 2,
                                           "actuator:rotor-horizontal-eye":1,
                                           "actuator:rotor-vertical-eye":1,
                                           "actuator:discriminator":1,
                                           "hidden:1":12,
                                           "hidden:2":4},
                time_factor: int = 16):

        self.init_agents_composition = init_agents_composition
        self.time_factor = time_factor

        # instantiate all agents
        self.agents = {}

        for key, value in init_agents_composition.items():

            if key.startswith("sensor"):
                assigned_env = key.split(":")[1]
                self.agents[key] = [SensorAgent(assigned_env=assigned_env) for _ in range(value)]

            elif key.startswith("actuator"):
                assigned_actuator = key.split(":")[1]
                self.agents[key] = [ActuatorAgent(assigned_actuator=assigned_actuator) for _ in range(value)]

            elif key.startswith("hidden"):
                agent_layer = int(key.split(":")[1])
                time__unit = min(1, time_factor * (agent_layer - 1))
                self.agents[key] = [HiddenAgent(agent_layer=agent_layer, time_unit=time__unit) for _ in range(value)]



