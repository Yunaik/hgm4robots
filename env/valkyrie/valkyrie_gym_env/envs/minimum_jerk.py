import numpy as np


def mjtg(currents, setpoints, frequency, move_time):
    trajectories = []
    for idx in range(len(currents)):
        current = currents[idx]
        setpoint = setpoints[idx]

        trajectory = []
        timefreq = int(move_time * frequency)
        for time in range(1, timefreq+1):
            trajectory.append(
                current + (setpoint - current) *
                (10.0 * (time/timefreq)**3
                 - 15.0 * (time/timefreq)**4
                 + 6.0 * (time/timefreq)**5))

        trajectories.append(trajectory)

    return np.array(trajectories)
