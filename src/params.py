


class params:
    def __init__(self, dt, L=4, surface_slope=0.0, acc=0.0):
        self.dt = dt
        self.spherical_harmonics_order = L
        self.slope = surface_slope
        self.acc = acc
