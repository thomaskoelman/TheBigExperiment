import numpy as np
from Game import Game
import math

class Lookup:
    def __init__(self, shape):
        self.j_bins = np.arange(0, 1., step=0.1)
        self.k_bins = np.arange(-1, 1, 0.1)
        self.err_lvls = np.array([0.0, 0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128, 0.256, 0.512, 1.0])
        x = len(self.j_bins)
        y = len(self.k_bins)
        z = len(self.err_lvls)
        self.tbl = np.zeros((z, y, x))
        def add_record(true_belief, true_attitude, est_belief, est_attitude, error):
            p_1, p_2 = np.random.standard_normal(size=shape), np.random.standard_normal(size=shape)
            g = Game(p_1, p_2)

            true_pivot = np.random.randint(sum(shape))
            _, true_strategy = g.modify(true_belief, true_attitude).lemke_howson(true_pivot)
            true_move = np.random.choice(len(true_strategy), p=true_strategy)

            est_pivot = np.random.randint(sum(shape))
            _, est_strategy = g.modify(est_belief, est_attitude).lemke_howson(est_pivot)
            est_prob_move = est_strategy[true_move]

            k = (est_attitude + est_belief) / (math.sqrt(est_attitude ** 2 + 1) * math.sqrt(est_belief ** 2 + 1))

            j_id = np.digitize(est_prob_move, self.j_bins) - 1
            k_id = np.digitize(k, self.k_bins) - 1
            err_id = np.digitize(error, self.err_lvls) - 1

            self.tbl[err_id, k_id, j_id] += 1

        for true_belief, true_attitude, est_belief, est_attitude in [(true_bel, true_att, est_bel, est_att) for true_bel in np.arange(-1, 1, 0.256) for est_bel in np.arange(-1, 1, 0.256) for est_att in np.arange(-1, 1, 0.256) for true_att in np.arange(-1, 1, 0.256)]:
            print(true_belief, true_attitude, est_belief, est_attitude)
            error = math.sqrt((true_attitude-est_attitude) ** 2 + (true_belief - est_belief) ** 2)
            add_record(true_belief, true_attitude, est_belief, est_attitude, error)
            if not error:
                for true_belief, true_attitude, est_belief, est_attitude in [(true_bel, true_att, est_bel, est_att) for
                                                                             true_bel in np.arange(true_belief-0.128, true_belief+0.128, 0.064) for
                                                                             est_bel in np.arange(true_attitude-0.128, true_attitude+0.128, 0.064) for
                                                                             est_att in np.arange(est_belief-0.128, est_belief+0.128, 0.064) for
                                                                             true_att in np.arange(est_attitude-0.128, est_attitude+0.128, 0.64)]:
                    error = math.sqrt((true_attitude - est_attitude) ** 2 + (true_belief - est_belief) ** 2)
                    add_record(true_belief, true_attitude, est_belief, est_attitude, error)
                    if not error:
                        for true_belief, true_attitude, est_belief, est_attitude in [
                            (true_bel, true_att, est_bel, est_att) for
                            true_bel in np.arange(true_belief - 0.032, true_belief + 0.032, 0.016) for
                            est_bel in np.arange(true_attitude - 0.032, true_attitude + 0.032, 0.016) for
                            est_att in np.arange(est_belief - 0.032, est_belief + 0.032, 0.016) for
                            true_att in np.arange(est_attitude - 0.032, est_attitude + 0.032, 0.016)]:
                            error = math.sqrt((true_attitude - est_attitude) ** 2 + (true_belief - est_belief) ** 2)
                            add_record(true_belief, true_attitude, est_belief, est_attitude, error)
                            if not error:
                                for true_belief, true_attitude, est_belief, est_attitude in [
                                    (true_bel, true_att, est_bel, est_att) for
                                    true_bel in np.arange(true_belief - 0.008, true_belief + 0.008, 0.004) for
                                    est_bel in np.arange(true_attitude - 0.008, true_attitude + 0.008, 0.004) for
                                    est_att in np.arange(est_belief - 0.008, est_belief + 0.008, 0.004) for
                                    true_att in np.arange(est_attitude - 0.008, est_attitude + 0.008, 0.004)]:
                                    error = math.sqrt(
                                        (true_attitude - est_attitude) ** 2 + (true_belief - est_belief) ** 2)
                                    add_record(true_belief, true_attitude, est_belief, est_attitude, error)
                                    if not error:
                                        for true_belief, true_attitude, est_belief, est_attitude in [
                                            (true_bel, true_att, est_bel, est_att) for
                                            true_bel in np.arange(true_belief - 0.002, true_belief + 0.002, 0.001) for
                                            est_bel in np.arange(true_attitude - 0.002, true_attitude + 0.002, 0.001)
                                            for
                                            est_att in np.arange(est_belief - 0.002, est_belief + 0.002, 0.001) for
                                            true_att in np.arange(est_attitude - 0.002, est_attitude + 0.002, 0.001)]:
                                            error = math.sqrt(
                                                (true_attitude - est_attitude) ** 2 + (true_belief - est_belief) ** 2)
                                            add_record(true_belief, true_attitude, est_belief, est_attitude, error)
                                            if not error:
                                                for true_belief, true_attitude, est_belief, est_attitude in [
                                                    (true_bel, true_att, est_bel, est_att) for
                                                    true_bel in
                                                    np.arange(true_belief - 0.0005, true_belief + 0.0005, 0.00025) for
                                                    est_bel in
                                                    np.arange(true_attitude - 0.0005, true_attitude + 0.0005, 0.00025)
                                                    for
                                                    est_att in np.arange(est_belief - 0.0005, est_belief + 0.0005, 0.00025)
                                                    for
                                                    true_att in
                                                    np.arange(est_attitude - 0.0005, est_attitude + 0.0005, 0.00025)]:
                                                    error = math.sqrt(
                                                        (true_attitude - est_attitude) ** 2 + (
                                                                    true_belief - est_belief) ** 2)
                                                    add_record(true_belief, true_attitude, est_belief, est_attitude,
                                                               error)
        for err_id, k_id in [(a, b) for a in range(z) for b in range(y)]:
            j_values = self.tbl[err_id, k_id]
            if sum(j_values):
                self.tbl[err_id, k_id] = j_values / sum(j_values)
            else:
                self.tbl[err_id, k_id] = np.array([1/len(j_values)] * len(j_values))

    def get_table(self):
        return self.tbl

    def t(self, j, k, err_lvl):
        return self.tbl[err_lvl, k, j]

# tabel = Lookup(shape=(2,2))
# print(tabel.get_table())
# for err_id, k_id in [(z, y) for y in range(len(tabel.k_bins)) for z in range(len(tabel.err_lvls))]:
#     j_values = tabel.tbl[err_id, k_id]
#     print(j_values)