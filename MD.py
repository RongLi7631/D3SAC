import random
import math
import numpy as np
import para
import os

N = para.N
zeta = para.zeta
A_max = para.A_max
ALPHA = para.alpha
steps = para.steps


class MD:
    def __init__(self, id, F_MD, env, bs_list):
        self.id = id
        self.v = random.randint(0, 3)
        self.x = random.randint(0, 1600)
        self.y = random.randint(0, 1200)
        self.theta = random.uniform(0, 2 * math.pi)
        self.F_MD = F_MD
        self.a = 2.7
        self.beta = 0.05
        self.env = env
        self.B, self.C, self.Gamma, self.Priority = env.task()
        self.bs_list = bs_list
        self.connect_BS = self.connect_choice()
        self.dir = "dataset/MDT/{}.txt".format(self.id)
        self.data = self.load_data()

    def load_data(self):
        if not os.path.isfile(self.dir):
            raise FileNotFoundError(f"The user data file does not exist：{self.dir}")
        user_positions = {}
        lines_read = 0
        with open(self.dir, 'r', encoding='utf-8') as f:
            for line_number, line in enumerate(f):
                if lines_read >= steps:
                    break
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 3:
                    print(f"[Warning] Format error on line {line_number + 1}. It should contain at least 3 columns. Skipping: {line}")
                    continue
                try:
                    x = round(float(parts[1]), 2)
                    y = round(float(parts[2]), 2)
                except ValueError:
                    print(f"[Warning] The x or y in line {line_number + 1} is not a valid number. Skipping.: {line}")
                    continue
                user_positions[line_number] = (x, y)
                lines_read += 1
        return user_positions

    def move(self):
        Delta_v = random.uniform(-A_max * zeta, A_max * zeta)
        Delta_theta = random.uniform(-0.5 * pow(zeta, 2) * ALPHA, 0.5 * pow(zeta, 2) * ALPHA)
        self.x = min(max(self.x + self.v * math.cos(self.theta), 0), 1600)
        self.y = min(max(self.y + self.v * math.sin(self.theta), 0), 1200)
        self.v = min(max(self.v + Delta_v, 0), 10)
        self.theta = self.theta + Delta_theta

    def move_real(self, step):
        self.x, self.y = self.data[step]

    def distance(self, bs):
        d = np.sqrt((self.x - bs.X) ** 2 + (self.y - bs.Y) ** 2 + bs.H ** 2)
        return d

    def gain(self):
        d = self.distance(self.bs_list[self.connect_BS])
        G = self.beta * (d ** -self.a)
        g_real = np.random.randn() * np.sqrt(0.5)
        g_imag = np.random.randn() * np.sqrt(0.5)
        g_rayleigh = g_real + 1j * g_imag
        g_complex = np.sqrt(G) * g_rayleigh
        g = np.abs(g_complex) ** 2
        return g

    def data_trans(self, p_MD, g):
        s = g * p_MD / 3.981e-18
        R = 1e7 * math.log2(1 + s)
        return max(R, 0.001)

    def offloading(self, B, p_MD, g):
        R = self.data_trans(p_MD, g)
        T = B / R
        E = p_MD * B / R
        return T, E

    def local_com(self, B, C, f):
        T = B * C / (f * self.F_MD)
        E = 1e-27 * pow((f * self.F_MD), 2) * B * C
        return T, E

    def reset(self):
        self.B, self.C, self.Gamma, self.Priority = self.env.task()
        self.v = random.randint(0, 3)
        self.x = random.randint(0, 1600)
        self.y = random.randint(0, 1200)
        self.theta = random.uniform(0, 2 * math.pi)

    def connect_choice(self):
        D = []
        for n in range(0, N):
            d = np.sqrt((self.x - self.bs_list[n].X) ** 2 + (self.y - self.bs_list[n].Y) ** 2 + self.bs_list[n].H ** 2)
            D.append(d)
        return np.argmin(D)
