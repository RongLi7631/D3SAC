import numpy as np
import gym
from gym import spaces
import random
import para
import math
from scipy.special import lambertw

N = para.N
M = para.M
F_BS = para.F_BS
w_t = para.w_t
w_e = para.w_e


class EdgeEnv:
    def __init__(self):
        low = np.tile(np.zeros(N + 6), (M, 1))
        high = np.tile(np.ones(N + 6), (M, 1))
        self.observation_space = spaces.Box(low=low, high=high, shape=(M, N + 6))
        self.action_space = gym.spaces.Discrete(N + 1)
        self.action_space2 = gym.spaces.Box(low=np.array([0.0]), high=np.array([1.0]), shape=(1,))
        self.state = None

    def step(self, m, b, f, tcn, bs_list, md_list):
        T = 0
        T1 = 0
        E1 = 0
        T1_pmax = 0
        E1_pmax = 0
        p = 0
        R = md_list[m].Priority
        if R == 1:
            tcn[3] = tcn[3] + 1
        elif R == 2:
            tcn[4] = tcn[4] + 1
        else:
            tcn[5] = tcn[5] + 1
        if b == 0:
            if f != 0:
                T, E = md_list[m].local_com(md_list[m].B, md_list[m].C, f)
                reward = R - (w_t * T + w_e * E)
            else:
                reward = -1
        else:
            f1 = int(1e9 + 2e9 * f)
            if bs_list[b - 1].res_F < f1:
                reward = -1
            else:
                p_max = 0.5
                T2, E2 = bs_list[b - 1].computing(md_list[m].B, md_list[m].C, f1)
                if b == md_list[m].connect_BS + 1:
                    p, g = self.get_p(md_list[m], T2, 0)
                    T1, E1 = md_list[m].offloading(md_list[m].B, p, g)
                    T = T1 + T2
                    E = E1 + E2
                else:
                    pareto_value = np.random.pareto(1.5) + 1
                    pareto_ratio = min(pareto_value * 0.01, 0.9)
                    T3 = md_list[m].B / (5e9 * (1 - pareto_ratio))
                    p, g = self.get_p(md_list[m], T2, T3)
                    T1, E1 = md_list[m].offloading(md_list[m].B, p, g)

                    T = T1 + T2 + T3
                    E = E1 + E2
                T1_pmax, E1_pmax = md_list[m].offloading(md_list[m].B, p_max, g)
                bs_list[b - 1].res_F = bs_list[b - 1].res_F - f1
                reward = R - (w_t * T + w_e * E)
        if T > md_list[m].Gamma or T == 0:
            reward = reward - 1
        else:
            if R == 1:
                tcn[0] = tcn[0] + 1
            elif R == 2:
                tcn[1] = tcn[1] + 1
            else:
                tcn[2] = tcn[2] + 1

        if b != 0:
            for i in range(m + 1, M):
                self.state[i][b - 1] = bs_list[b - 1].res_F / max(F_BS)

        md_list[m].B, md_list[m].C, md_list[m].Gamma, md_list[m].Priority = self.task()
        md_list[m].move()
        md_list[m].connect_BS = md_list[m].connect_choice()
        self.state[m][N] = (md_list[m].B - 0.8e6) / (2.4e6 - 0.8e6)
        self.state[m][N + 1] = (md_list[m].C - 100) / 900
        self.state[m][N + 2] = (md_list[m].Gamma - 0.6) / 0.1
        self.state[m][N + 3] = (md_list[m].Priority - 1) / 2
        self.state[m][N + 4] = md_list[m].distance(bs_list[md_list[m].connect_BS]) / 1600
        self.state[m][N + 5] = md_list[m].connect_BS / N

        phi_p = w_t * T1 + w_e * E1
        phi_pmax = w_t * T1_pmax + w_e * E1_pmax
        return np.array(self.state), reward, False, tcn, T1, E1, T1_pmax, E1_pmax, phi_p, phi_pmax, p

    def step_real(self, m, b, f, tcn, bs_list, md_list, step):
        T = 0
        T1 = 0
        E1 = 0
        T1_pmax = 0
        E1_pmax = 0
        p = 0
        R = md_list[m].Priority
        if R == 1:
            tcn[3] = tcn[3] + 1
        elif R == 2:
            tcn[4] = tcn[4] + 1
        else:
            tcn[5] = tcn[5] + 1
        if b == 0:
            if f != 0:
                T, E = md_list[m].local_com(md_list[m].B, md_list[m].C, f)
                reward = R - (w_t * T + w_e * E)
            else:
                reward = -1
        else:
            f1 = int(1e9 + 2e9 * f)
            if bs_list[b - 1].res_F < f1:
                reward = -1
            else:
                p_max = 0.5
                T2, E2 = bs_list[b - 1].computing(md_list[m].B, md_list[m].C, f1)
                if b == md_list[m].connect_BS + 1:
                    p, g = self.get_p(md_list[m], T2, 0)
                    T1, E1 = md_list[m].offloading(md_list[m].B, p, g)
                    T = T1 + T2
                    E = E1 + E2
                else:
                    pareto_value = np.random.pareto(1.5) + 1
                    pareto_ratio = min(pareto_value * 0.01, 0.9)
                    T3 = md_list[m].B / (5e9 * (1 - pareto_ratio))
                    p, g = self.get_p(md_list[m], T2, T3)
                    T1, E1 = md_list[m].offloading(md_list[m].B, p, g)
                    T = T1 + T2 + T3
                    E = E1 + E2
                T1_pmax, E1_pmax = md_list[m].offloading(md_list[m].B, p_max, g)
                bs_list[b - 1].res_F = bs_list[b - 1].res_F - f1
                reward = R - (w_t * T + w_e * E)
        if T > md_list[m].Gamma or T == 0:
            reward = reward - 1
        else:
            if R == 1:
                tcn[0] = tcn[0] + 1
            elif R == 2:
                tcn[1] = tcn[1] + 1
            else:
                tcn[2] = tcn[2] + 1

        if b != 0:
            for i in range(m + 1, M):
                self.state[i][b - 1] = bs_list[b - 1].res_F / max(F_BS)

        md_list[m].B, md_list[m].C, md_list[m].Gamma, md_list[m].Priority = self.task()
        md_list[m].move_real(step)
        md_list[m].connect_BS = md_list[m].connect_choice()
        self.state[m][N] = (md_list[m].B - 0.8e6) / (2.4e6 - 0.8e6)
        self.state[m][N + 1] = (md_list[m].C - 100) / 900
        self.state[m][N + 2] = (md_list[m].Gamma - 0.6) / 0.1
        self.state[m][N + 3] = (md_list[m].Priority - 1) / 2
        self.state[m][N + 4] = md_list[m].distance(bs_list[md_list[m].connect_BS]) / 1600
        self.state[m][N + 5] = md_list[m].connect_BS / N

        phi_p = w_t * T1 + w_e * E1
        phi_pmax = w_t * T1_pmax + w_e * E1_pmax
        return np.array(self.state), reward, False, tcn, T1, E1, T1_pmax, E1_pmax, phi_p, phi_pmax, p

    def reset(self, md_list, bs_list):
        for m in range(0, M):
            md_list[m].reset()
        for n in range(0, N):
            bs_list[n].reset()
        self.state = np.zeros((M, N + 6))
        for i in range(0, M):
            for n in range(0, N):
                self.state[i][n] = bs_list[n].res_F / max(F_BS)
        for m in range(0, M):
            md_list[m].connect_BS = md_list[m].connect_choice()
            self.state[m][N] = (md_list[m].B - 0.8e6) / (2.4e6 - 0.8e6)
            self.state[m][N + 1] = (md_list[m].C - 100) / 900
            self.state[m][N + 2] = (md_list[m].Gamma - 0.6) / 0.1
            self.state[m][N + 3] = (md_list[m].Priority - 1) / 2
            self.state[m][N + 4] = md_list[m].distance(bs_list[md_list[m].connect_BS]) / 1600
            self.state[m][N + 5] = md_list[m].connect_BS / N
        return np.array(self.state)

    def reset_state(self):
        for m in range(0, M):
            for n in range(0, N):
                self.state[m][n] = F_BS[n] / max(F_BS)

    def task(self):
        B = random.randint(0.8e6, 2.4e6)
        C = random.randint(100, 1000)
        Gamma = random.uniform(0.6, 0.7)
        Priority = random.randint(1, 3)
        return B, C, Gamma, Priority

    def get_p(self, md, T2, T3):
        B = md.B
        Gamma = md.Gamma
        g = md.gain()
        k = g / 3.981e-18
        P_max = 0.5
        if Gamma - T2 - T3 > 0:
            Gamma_bar = max(Gamma - T2 - T3, 1e-3)
            P_min = min((2 ** (B / (1e7 * Gamma_bar)) - 1) / k, 0.5)
            p = (math.exp(lambertw((w_t * k - w_e) / (w_e * math.e)) + 1) - 1) / k
            if p <= P_min:
                p = P_min
            elif p >= P_max:
                p = P_max
        else:
            p = P_max
        return p, g

    def get_state(self):
        return np.array(self.state)
