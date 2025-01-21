


class params:
    def __init__(self, dt, L=4, surface_slope=0.0, acc=lambda x: 0.0, ρ=9.138e-19, g=9.7692e15):
        self.dt = dt
        self.L = L
        self.slope = surface_slope
        self.acc = acc
        self.ρ = ρ
        self.g = g
