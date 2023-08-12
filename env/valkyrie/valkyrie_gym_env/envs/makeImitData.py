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
lfoot_vel = lfoot_vel[start_idx:, :]
rfoot_vel = rfoot_vel[start_idx:, :]
lfoot_orientation = lfoot_orientation[start_idx:, :]
rfoot_orientation = rfoot_orientation[start_idx:, :]
index = index[start_idx:]

t = np.arange(len(index))/25


lfoot_pos = lfoot_pos[:41, :]
rfoot_pos = rfoot_pos[:41, :]
lfoot_vel = lfoot_vel[:41, :]
rfoot_vel = rfoot_vel[:41, :]
lfoot_orientation = lfoot_orientation[:41, :]
rfoot_orientation = rfoot_orientation[:41, :]
index = index[:41]


lfoot_pos_offset = copy.copy(lfoot_pos)
rfoot_pos_offset = copy.copy(rfoot_pos)

lfoot_pos_offset[:, 2] -= max(lfoot_pos[:, 2])
rfoot_pos_offset[:, 2] -= max(rfoot_pos[:, 2])


pitch_threshold = 5*3.14/180.
l_contact_orientation = [idx for idx, val in enumerate(
    lfoot_orientation[:, 1]) if val < pitch_threshold]
r_contact_orientation = [idx for idx, val in enumerate(
    rfoot_orientation[:, 1]) if val < pitch_threshold]

pos_z_threshold = 0.02
lfoot_pos_z_contact = [idx for idx, val in enumerate(
    lfoot_pos[:, 2]) if abs(val) < pos_z_threshold]
rfoot_pos_z_contact = [idx for idx, val in enumerate(
    rfoot_pos[:, 2]) if abs(val) < pos_z_threshold]

vel_z_threshold = 0.1
lfoot_vel_z_contact = [idx for idx, val in enumerate(
    lfoot_vel[:, 2]) if abs(val) < vel_z_threshold]
rfoot_vel_z_contact = [idx for idx, val in enumerate(
    rfoot_vel[:, 2]) if abs(val) < vel_z_threshold]
plt.figure()
plt.subplot(3, 1, 1)
plt.plot(index, lfoot_orientation[:, 1]*180/3.14, label="left p")
plt.plot(index, rfoot_orientation[:, 1]*180/3.14, label="right p")
plt.title("Orientation")
plt.legend()

"""Manually add one to left and remove one from right to ensure same amount of contacts"""
idx = r_contact_orientation.pop(-3)
print("Popped %d from right" % idx)
l_contact_orientation.insert(1, 1)

idx_in_both = [idx for idx in index if (
    idx in l_contact_orientation) and (idx in r_contact_orientation)]
print("L: %d/%d. R: %d/%d. DSP: %.2f, SSP: %.2f" % (len(l_contact_orientation),
                                                    41, len(r_contact_orientation), 41, len(idx_in_both)/41, 1-len(idx_in_both)/41))

l_contact = np.array([0.]*len(index))
r_contact = np.array([0.]*len(index))

for idx in l_contact_orientation:
    l_contact[idx] = 1
for idx in r_contact_orientation:
    r_contact[idx] = 1


plt.subplot(3, 1, 2)
plt.scatter(index, (l_contact)*1.0, label="L contact Orientation")
plt.scatter(index, (r_contact)*1.1, label="R contact Orientation")
plt.legend()
plt.ylim([0., 1.5])

plt.subplot(3, 1, 3)
plt.plot(index, lfoot_pos_offset[:, 2], label="left z")
plt.plot(index, rfoot_pos_offset[:, 2], label="right z")
plt.title("Z Pos")
plt.legend()

plt.show()

np.savetxt("imit_data/imit_lfoot_pos.csv", lfoot_pos)
np.savetxt("imit_data/imit_rfoot_pos.csv", rfoot_pos)
np.savetxt("imit_data/imit_lfoot_orientation.csv", lfoot_orientation)
np.savetxt("imit_data/imit_rfoot_orientation.csv", rfoot_orientation)
np.savetxt("imit_data/imit_lfoot_contact.csv", l_contact)
np.savetxt("imit_data/imit_rfoot_contact.csv", r_contact)
np.savetxt("imit_data/imit_index.csv", index)
