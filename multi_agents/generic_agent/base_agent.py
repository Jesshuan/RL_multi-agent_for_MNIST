from copy import copy
import numpy as np


class AgentBase():

    def __init__(self, agent_id, nb_channels = 8, mlp_units = [16], final_unit="logistic", nb_class=2, memory_size=1000, freq_random_1 = 0.25):

        self.init_life_point = 100
        self.life_point = copy(self.init_life_point)

        self.agent_id = agent_id
        self.nb_channels = nb_channels
        self.model = "TODO model def sous tensorflow"

        self.memory_size = memory_size
        self.reward_history = []
        self.freq_random_1 = freq_random_1

    def receive_reward(self, reward):

        self.reward_history.append(reward)

    def action(self, state):

        # totally random
        base_rand_int = np.random.random(1)

        if base_rand_int <= self.freq_random_1:
            return 1
        else:
            return 0