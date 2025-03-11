
from copy import deepcopy

from multi_agents.generic_agent.generic_agents import SensorAgent, ActuatorAgent, HiddenAgent

import random

import numpy as np

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
                time_factor: int = 16,
                actuator_channels: int = 8,
                hid_agent_channels: int = 8,
                connexion_rule : str = "40 | 30 | 40",
                reward_retention_coeff: float = 0.5):

        self.init_agents_composition = init_agents_composition
        self.time_factor = time_factor
        self.connexion_rule = connexion_rule
        self.actuator_channels = actuator_channels
        self.hid_agent_channels = hid_agent_channels
        self.reward_retention_coeff = reward_retention_coeff

        # instantiate all agents
        self.agents_registry = {}
        self.agents_connexion = {}

        self.parsed_conn_rules = self.parse_connexion_rule()
        self.nb_actuator = self.actuator_count()
        self.nb_sensor = self.sensor_count()
        self.nb_hidden_by_layer_count_list = self.hidden_by_layer_count_list()
        self.max_hidden_layer = self.compute_max_hidden_layer()

        self.nb_agents_by_cat = {
            "sensor" : self.nb_sensor,
            "actuator" : self.nb_actuator,
            "hidden" : self.nb_hidden_by_layer_count_list
        }

        # others
        self.transformed_reward_mapping = {
            "actuator" : 1,
            "sensor" : 10,
            "hidden" : 10
        }
    
    def sensor_count(self):
        return sum([value for key, value in self.init_agents_composition.items() if key.startswith("sensor:")])

    def actuator_count(self):
        return sum([value for key, value in self.init_agents_composition.items() if key.startswith("actuator:")])
    
    def hidden_by_layer_count_list(self):
        return [value for key, value in self.init_agents_composition.items() if key.startswith("hidden:")]

    def compute_max_hidden_layer(self):
        return max([int(key.split(":")[1]) for key in self.init_agents_composition.keys() if key.startswith("hidden:")])

    def parse_connexion_rule(self):
        return [int(expr) for expr in deepcopy(self.connexion_rule).replace(" ", "").split("|")]

    def build_agents_registry(self):
        for key, value in self.init_agents_composition.items():
            if key.startswith("sensor"):
                assigned_env = key.split(":")[1]
                for i in range(value):
                    agent_id = key + "_" + str(i)
                    self.agents_registry[agent_id] = SensorAgent(agent_id = agent_id, assigned_env=assigned_env)
            elif key.startswith("actuator"):
                assigned_actuator = key.split(":")[1]
                for i in range(value):
                    agent_id = key + "_" + str(i)
                    self.agents_registry[agent_id] = ActuatorAgent(agent_id = agent_id, assigned_actuator=assigned_actuator)
            elif key.startswith("hidden"):
                agent_layer = int(key.split(":")[1])
                time__unit = max(1, self.time_factor * (agent_layer - 1))
                for i in range(value):
                    agent_id = key + "_" + str(i)
                    self.agents_registry[agent_id] = HiddenAgent(agent_id = agent_id, agent_layer=agent_layer, time_unit=time__unit)


    def build_agents_connexions(self):
        for key, value in self.init_agents_composition.items():

            if key.startswith("sensor"):
                for i in range(value):
                    agent_id = key + "_" + str(i)
                    self.agents_connexion[agent_id] = []

            if key.startswith("actuator"):
                nb_conn_intern = min(int(((self.parsed_conn_rules[1]) / 100) * self.actuator_channels), self.nb_actuator - 1)
                nb_conn_hidd_1 = self.actuator_channels - nb_conn_intern
                for i in range(value):
                    conn_list = []
                    agent_id = key + "_" + str(i)
                    index_intern_list = [ag_id for ag_id in self.agents_registry.keys() if agent_id!=ag_id and (ag_id.startswith("actuator:") or ag_id.startswith("sensor:"))]
                    conn_list.extend(random.sample(population = index_intern_list, k = nb_conn_intern))
                    index_extern_list = [ag_id for ag_id in self.agents_registry.keys() if ag_id.startswith("hidden:1")]
                    conn_list.extend(random.sample(population = index_extern_list, k = nb_conn_hidd_1))
                    self.agents_connexion[agent_id] = conn_list

            elif key.startswith("hidden"):

                agent_layer = int(key.split(":")[1])

                if agent_layer == self.max_hidden_layer:

                    nb_conn_intern = min(int(((self.parsed_conn_rules[1]) / 100) * self.hid_agent_channels), self.nb_hidden_by_layer_count_list[-1] - 1)
                    nb_conn_hidd_down_layer = self.hid_agent_channels - nb_conn_intern
                    for i in range(value):
                        conn_list = []
                        agent_id = key + "_" + str(i)
                        index_intern_list = [ag_id for ag_id in self.agents_registry.keys() if agent_id!=ag_id and ag_id.startswith(f"hidden:{self.max_hidden_layer}")]
                        conn_list.extend(random.sample(population = index_intern_list, k = nb_conn_intern))
                        if self.max_hidden_layer - 1 >=1 :
                            index_extern_list = [ag_id for ag_id in self.agents_registry.keys() if ag_id.startswith(f"hidden:{self.max_hidden_layer - 1}")]
                        else:
                            index_extern_list = [ag_id for ag_id in self.agents_registry.keys() if (ag_id.startswith(f"actuator:") or ag_id.startswith('sensor:'))]
                        conn_list.extend(random.sample(population = index_extern_list, k = nb_conn_hidd_down_layer))
                        self.agents_connexion[agent_id] = conn_list
                else:
                    
                    nb_conn_intern = min(int(((self.parsed_conn_rules[1]) / 100) * self.hid_agent_channels), self.nb_hidden_by_layer_count_list[agent_layer - 1] - 1)
                    nb_conn_hidd_down_layer = min(int(((self.parsed_conn_rules[2]) / 100) * self.hid_agent_channels), self.nb_hidden_by_layer_count_list[agent_layer])
                    nb_conn_hidd_up_layer = self.hid_agent_channels - nb_conn_intern - nb_conn_hidd_down_layer
                    for i in range(value):
                        conn_list = []
                        agent_id = key + "_" + str(i)
                        index_intern_list = [ag_id for ag_id in self.agents_registry.keys() if agent_id!=ag_id and ag_id.startswith(f"hidden:{agent_layer}")]
                        conn_list.extend(random.sample(population = index_intern_list, k = nb_conn_intern))
                        if agent_layer >=2 :
                            index_down_list = [ag_id for ag_id in self.agents_registry.keys() if ag_id.startswith(f"hidden:{agent_layer - 1}")]
                        else:
                            index_down_list = [ag_id for ag_id in self.agents_registry.keys() if ag_id.startswith(f"actuator:") or ag_id.startswith(f"sensor:") ]
                        conn_list.extend(random.sample(population = index_down_list, k = nb_conn_hidd_down_layer))
                        index_up_list = [ag_id for ag_id in self.agents_registry.keys() if ag_id.startswith(f"hidden:{agent_layer + 1}")]
                        conn_list.extend(random.sample(population = index_up_list, k = nb_conn_hidd_up_layer))
                        self.agents_connexion[agent_id] = conn_list

    def return_category_and_deep_pos_layer_from_agent_id(self, agent_id):

        agent_category = agent_id.split(":")[0]
        if agent_category == "hidden":
            pos_layer = int(agent_id.split("hidden:")[1].split("_")[0])
        else:
            pos_layer = 1

        return agent_category, pos_layer

    def adapt_reward_to_agent(self, reward, agent_category, pos_layer):

        base_transfo = self.transformed_reward_mapping.get(agent_category)
        if agent_category == "hidden":
            nb_consecutive_agents_in_layer = self.nb_agents_by_cat.get(agent_category)[pos_layer - 1]
        else:
            nb_consecutive_agents_in_layer = self.nb_agents_by_cat.get(agent_category)

        return np.round((reward * base_transfo**(pos_layer) * nb_consecutive_agents_in_layer)/ 100, 4)
    
    
    def reward_distrib_at_group(self, reward, agent_list):

        len_group = len(agent_list)

        if int(self.reward_retention_coeff*reward) <= len_group:
            reward_per_agent = 1
        else:
            reward_per_agent = np.round(self.reward_retention_coeff*reward/len_group, 4)


        for agent_id in agent_list:
            
            agent_category, pos_layer = self.return_category_and_deep_pos_layer_from_agent_id(agent_id=agent_id)

            transformed_reward = self.adapt_reward_to_agent(reward_per_agent, agent_category, pos_layer)

            if transformed_reward >=1 :
                self.agents_registry[agent_id].receive_reward(transformed_reward)
                reward -= reward_per_agent
            else:
                pass

        return reward
    
    def iterative_reward_distrib(self, reward, agent_list):

        if reward > len(agent_list):

            reward = self.reward_distrib_at_group(reward, agent_list)

            downstream_conn_agent_list = [conn_ag_id for ag_id in agent_list for conn_ag_id in self.agents_connexion[ag_id]]

            self.iterative_reward_distrib(reward, downstream_conn_agent_list)

        return None

    def distrib_global_reward(self, reward, direct_agent_group):

        init_agent_list = [ag_id for ag_id in self.agents_registry if ag_id.startswith(direct_agent_group)]

        self.iterative_reward_distrib(reward, init_agent_list)