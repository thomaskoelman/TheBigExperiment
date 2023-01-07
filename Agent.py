import numpy as np
import math
from Particles import Particles
from Game import Game
from Lookup import Lookup

class Agent:
    def __init__(self, table: Lookup, reciprocation = 0.1, f_ab = 0.1, f_nash = 0.1):
        self.particles = Particles(nb=300, shape=(16,16))
        self.reciprocation = reciprocation
        self.f_ab = f_ab
        self.f_nash = f_nash
        self.__rewards = []
        self.table = table
        self.distr = np.array([1/12] * 12)

    def get_cooperation_value(self):
        # assess how cooperative your opponent is
        belief = self.particles.get_average_attitude()
        opp_belief = self.particles.get_average_belief()
        return (belief + opp_belief) / (math.sqrt(belief ** 2 + 1) * math.sqrt(opp_belief ** 2 + 1))

    def pick_move(self, game: Game):
        belief = np.clip(self.particles.get_average_attitude(), -1, 1)
        #print("belief: ",belief)
        pivot = self.particles.most_frequent_pivot()
        attitude = np.clip(belief + self.reciprocation, -1, 1)
        g = game.modify(attitude, belief)
        strategy, _ = g.lemke_howson(pivot)
        move = np.random.choice(len(strategy), p=strategy)
        return move, strategy[move]

    def estimate_prob_of_move(self, game: Game, move: int):
        belief = np.clip(self.particles.get_average_attitude(), -1, 1)
        opp_bel = np.clip(self.particles.get_average_belief(), -1 , 1)
        pivot = self.particles.most_frequent_pivot()
        g = game.modify(opp_bel, belief)
        _, strategy = g.lemke_howson(pivot)
        return strategy[move]

    def observe_move(self, game: Game, move: int):
        table = self.table
        error = self.estimate_error(table, game, move)
        self.particles.resample(game, move)
        self.particles.perturb_all(error, self.f_ab, self.f_nash)

    def estimate_error(self, table: Lookup, game: Game, move: int):
        j = self.estimate_prob_of_move(game, move)
        k = self.get_cooperation_value()
        j_id = np.digitize(j, table.j_bins) - 1
        k_id = np.digitize(k, table.k_bins) - 1
        error_lvls = table.err_lvls
        for err_id, prob in enumerate(self.distr):
            self.distr[err_id] *= table.t(j_id, k_id, err_id)
        if sum(self.distr):
            self.distr = self.distr / sum(self.distr)
        return np.average(error_lvls, weights=self.distr)


    def accept_reward(self, reward: float):
        self.__rewards.append(reward)

    def average_reward(self):
        return np.average(self.__rewards)

    def get_belief(self):
        return np.clip(self.particles.get_average_attitude(), -1, 1)

    def get_estimated_belief(self):
        return np.clip(self.particles.get_average_belief(), -1, 1)

    def get_attitude(self):
        return np.clip(self.particles.get_average_attitude(), -1, 1) + self.reciprocation

    def get_true_error(self, agent):
        true_attitude = agent.get_attitude()
        true_belief = agent.get_belief()
        est_attitude = self.get_belief()
        est_belief = self.get_estimated_belief()
        return math.sqrt((true_attitude - est_attitude) ** 2 + (true_belief - est_attitude) ** 2)


# g = Game(np.array([[-2,-10],[0,-5]]), np.array([[-2,-10],[0,-5]]))
# a = Agent()
# for i in range(100):
#     print(a.pick_move(g))