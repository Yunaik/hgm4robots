import numpy as np

import matplotlib
matplotlib.rc('figure', max_open_warning = 0)

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
matplotlib.use('Agg')

def multipage(filename, figs=None):
    pp = PdfPages(filename)
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
        # print("All fig names")
        # print(plt.get_fignums())
    for fig in figs:
        fig.savefig(pp, format='pdf',bbox_inches='tight')
    pp.close()


def mjtg(currents, setpoints, frequency, move_time):
    trajectories = []
    trajectories_vel = []
    for idx in range(len(currents)):
        current = currents[idx]
        setpoint = setpoints[idx]

        trajectory = []
        trajectory_derivative = []
        timefreq = int(move_time * frequency)
        # print("Time freq: %d" % timefreq)
        for time in range(1, timefreq+1):
            trajectory.append(
                current + (setpoint - current) *
                (10.0 * (time/timefreq)**3
                - 15.0 * (time/timefreq)**4
                + 6.0 * (time/timefreq)**5))

            trajectory_derivative.append(
                frequency * (1.0/timefreq) * (setpoint - current) *
                (30.0 * (time/timefreq)**2.0
                - 60.0 * (time/timefreq)**3.0
                + 30.0 * (time/timefreq)**4.0))
        # trajectory.append(trajectory[-1]) # append twice for convenience
        # trajectory.append(trajectory[-1])
        trajectories.append(trajectory)
        trajectories_vel.append(trajectory_derivative)
    
    return np.array(trajectories), np.array(trajectories_vel)

# Set up and calculate trajectory.
# average_velocity = 20.0
# current = 0.0
# setpoint = 180.0
# frequency = 25
# time = (setpoint - current) / average_velocity
freq = 1000
traj, traj_vel = mjtg([0], [1], freq, 1)
traj_vel /= max(traj_vel[0,:])

traj_acc = np.diff(traj_vel, axis=1)*freq
traj_acc /= max(traj_acc[0,:])

traj_jerk = np.diff(traj_acc, axis=1)*freq
traj_jerk /= max(traj_jerk[0,:])

t = np.arange(0,freq,1)/freq
# print(traj_acc.shape)
fig = plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(t, traj[0,:], label="Position")
plt.title("Position")
plt.xlabel("Time [s]")
plt.ylabel("Position [m]")
plt.subplot(1,2,2)
plt.plot(t, traj_vel[0,:], label="Velocity")
plt.plot(t[1:], traj_acc[0,:], label="Acceleration")
plt.plot(t[2:], traj_jerk[0,:], label="Jerk")
plt.legend()
plt.title("Normalised timederivatives of position")
plt.xlabel("Time [s]")
plt.ylabel("Derivatives")
multipage("minimum_jerk.pdf")
