import math
import numpy as np

class JointTrajectoryInterpolate:
    def __init__(self, shape=None):
        COEFF_LENGTH = 4
        if shape is None:
            self.data_shape = (COEFF_LENGTH,)
        elif type(shape) == int:
            self.data_shape = (COEFF_LENGTH, shape)
        else:
            self.data_shape = (COEFF_LENGTH, *shape)
        self.a=np.zeros(self.data_shape)
        self.t=0

    def interpolate(self, dt):
        self.t = self.t + dt
        target = self.a[3]*(self.t**3) + self.a[2]*(self.t**2) + self.a[1]*(self.t) + self.a[0]

        return target

    def cubic_interpolation_setup(self, q0, dq0, qf, dqf, tf):
        self.a[0] = q0
        self.a[1] = dq0
        self.a[2] = 3.0 * (qf - q0) / tf ** 2 - 2 * dq0 / tf - dqf / tf
        self.a[3] = -2 * (qf - q0) / tf ** 3 + (dqf + dq0) / tf ** 2
        self.t = 0 #reset the timing
