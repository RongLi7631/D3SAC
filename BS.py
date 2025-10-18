class BS:
    def __init__(self, id, F_BS, X, Y, H):
        self.id = id
        self.F_BS = F_BS
        self.res_F = self.F_BS
        self.X = X
        self.Y = Y
        self.H = H

    def computing(self, B, C, f_BS):
        T = B * C / f_BS
        E = 1e-27 * pow(f_BS, 2) * B * C
        return T, E

    def reset(self):
        self.res_F = self.F_BS
