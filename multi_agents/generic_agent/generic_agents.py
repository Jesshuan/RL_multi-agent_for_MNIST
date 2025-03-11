
from multi_agents.generic_agent.base_agent import AgentBase


class ActuatorAgent(AgentBase):

    def __init__(self, assigned_actuator, *args, **kwargs):
        super(ActuatorAgent, self).__init__(*args, **kwargs)

        self.assigned_actuator = assigned_actuator



class SensorAgent(AgentBase):

    def __init__(self, assigned_env, *args, **kwargs):
        super(SensorAgent, self).__init__(*args, **kwargs)
        self.assign_framework = assigned_env



class HiddenAgent(AgentBase):

    def __init__(self, agent_layer, time_unit = 1, *args, **kwargs):
        super(HiddenAgent, self).__init__(*args, **kwargs)

        self.agent_layer = agent_layer
        self.time_unit = time_unit