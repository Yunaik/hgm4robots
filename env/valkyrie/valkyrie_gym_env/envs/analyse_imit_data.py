import numpy as np
import matplotlib.pyplot as plt
import copy

lfoot_pos = np.genfromtxt("imit_data/lfoot_pos.csv")
rfoot_pos = np.genfromtxt("imit_data/rfoot_pos.csv")
lfoot_vel = np.genfromtxt("imit_data/lfoot_vel.csv")
rfoot_vel = np.genfromtxt("imit_data/rfoot_vel.csv")
lfoot_orientation = np.genfromtxt("imit_data/lfoot_orientation.csv")
rfoot_orientation = np.genfromtxt("imit_data/rfoot_orientation.csv")
index = np.genfromtxt("imit_data/indx.csv")

counter = 0

for count, idx in enumerate(index):
    if idx == 0:
        counter += 1
    if counter == 2 and idx == 0:
        start_idx = count

lfoot_pos = lfoot_pos[start_idx:, :]
rfoot_pos = rfoot_pos[start_idx:, :]
index = index[start_idx:]

lfoot_vel = np.diff(lfoot_pos[:, 0])
rfoot_vel = np.diff(rfoot_pos[:, 0])

lfoot_vel = np.hstack((lfoot_vel, 0))
rfoot_vel = np.hstack((rfoot_vel, 0))
lfoot_abs = []
rfoot_abs = []

for idx, pos_x in enumerate(lfoot_pos[:, 0]):
    if len(lfoot_abs) == 0:
        lfoot_abs.append(0)
    else:
        if lfoot_vel[idx] > 0:
            lfoot_abs.append(lfoot_abs[-1]+lfoot_vel[idx])
        else:
            lfoot_abs.append(lfoot_abs[-1])

for idx, pos_x in enumerate(rfoot_pos[:, 0]):
    if len(rfoot_abs) == 0:
        rfoot_abs.append(0)
    else:
        if rfoot_vel[idx] > 0:
            rfoot_abs.append(rfoot_abs[-1]+rfoot_vel[idx])
        else:
            rfoot_abs.append(rfoot_abs[-1])


t = np.arange(len(index))/25

print("X vel: %.2f(left), %.2f(right). Average: %.2f" % (
    lfoot_abs[-1]/t[-1], rfoot_abs[-1]/t[-1], (lfoot_abs[-1]/t[-1]+rfoot_abs[-1]/t[-1])/2))
dist = max(lfoot_pos[:, 0]) - min(lfoot_pos[:, 0])
_t = 41/25
print("Left Distance: %.2f, vel: %.2f, time: %.2f" % (dist, dist/_t, _t))
dist = max(rfoot_pos[:, 0]) - min(rfoot_pos[:, 0])
print("Right Distance: %.2f, vel: %.2f, time: %.2f" % (dist, dist/_t, _t))
plt.figure()
plt.plot(t, lfoot_pos[:, 0], label="left x")
plt.plot(t, rfoot_pos[:, 0], label="right x")
plt.legend()
plt.title("X Pos relative")
plt.figure()
plt.plot(t, lfoot_abs, label="left x")
plt.plot(t, rfoot_abs, label="right x")
plt.title("X Pos absolute")
plt.legend()

plt.show()
