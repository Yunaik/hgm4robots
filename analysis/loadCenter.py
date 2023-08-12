from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import copy
import pickle

from analysis.zmp_generator import generateCOM

import matplotlib
matplotlib.rc('figure', max_open_warning=0)

matplotlib.use('Agg')


def multipage(filename, figs=None):
    pp = PdfPages(filename)
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pp, format='pdf', bbox_inches='tight')
    pp.close()


frequency = 200
start_idx = int(frequency*0.1)
end_idx = frequency*6
com_pos = np.genfromtxt('com_pos.csv', delimiter=',')[start_idx:end_idx, :]
com_vel = np.diff(com_pos, axis=0)*frequency

lfoot = np.genfromtxt('lfoot.csv', delimiter=',')[:, start_idx:end_idx]
rfoot = np.genfromtxt('rfoot.csv', delimiter=',')[:, start_idx:end_idx]

foot_force = pickle.load(open("foot_force.p", "rb"))[start_idx:end_idx]

force_threshold = 10  # in [N]
px = 0
px_measured = []
for idx, force in enumerate(foot_force):
    lforce_info = force[0]
    rforce_info = force[1]
    upper_sum = 0
    lower_sum = 0
    is_touching = False
    if (len(lforce_info) + len(rforce_info)) > 0:
        for linfo in lforce_info:
            if linfo[9] > force_threshold:
                upper_sum += linfo[6][0]*linfo[9]
                lower_sum += linfo[9]
                is_touching = True

        for rinfo in rforce_info:
            if rinfo[9] > force_threshold:
                upper_sum += rinfo[6][0]*rinfo[9]
                lower_sum += rinfo[9]
                is_touching = True
        if is_touching:
            px = upper_sum/lower_sum
    px_measured.append(px)

zmp = []
for idx in range(len(lfoot[0, :])):
    if lfoot[0, idx] > rfoot[0, idx]:
        zmp.append(lfoot[0, idx])
    else:
        zmp.append(rfoot[0, idx])

com_vel_raw = copy.copy(com_vel)
window_size = 11
com_vel[:, 0] = savgol_filter(com_vel[:, 0], window_size, 3)
com_vel[:, 1] = savgol_filter(com_vel[:, 1], window_size, 3)
com_vel[:, 2] = savgol_filter(com_vel[:, 2], window_size, 3)

com_acc = np.diff(com_vel, axis=0)*frequency
com_acc_raw = np.diff(com_vel_raw, axis=0)*frequency
window_size = 101
com_acc[:, 0] = savgol_filter(com_acc[:, 0], window_size, 3)
com_acc[:, 1] = savgol_filter(com_acc[:, 1], window_size, 3)
com_acc[:, 2] = savgol_filter(com_acc[:, 2], window_size, 3)

com_pos = com_pos[:-2, :]
com_vel = com_vel[:-1, :]
com_vel_raw = com_vel_raw[:-1, :]
lfoot = lfoot[:, :-2]
rfoot = rfoot[:, :-2]
zmp = zmp[:-2]
max_zmp = 0
zmp_max = []
for point in zmp:
    max_zmp = point if point > max_zmp else max_zmp
    zmp_max.append(max_zmp)

zmp_measured = sorted(zmp)
zmp_measured = savgol_filter(zmp_measured, 51, 3)


px_measured = px_measured[:-2]
px_measured = savgol_filter(px_measured, 101, 3)

t = np.arange(0, len(com_pos))/frequency
zmp_x = com_pos[:, 0] - com_pos[:, 2]/9.81*com_acc[:, 0]
zmp_y = com_pos[:, 1] - com_pos[:, 2]/9.81*com_acc[:, 1]

omeg = np.sqrt(9.81/com_pos[:, 2])
cp_x = com_pos[:, 0]+com_vel_raw[:, 0]/omeg
cp_y = com_pos[:, 1]+com_vel_raw[:, 1]/omeg
window_size = 301
cp_x = savgol_filter(cp_x, window_size, 3)
cp_y = savgol_filter(cp_y, window_size, 3)

zmp_x = savgol_filter(zmp_x, window_size, 3)
zmp_y = savgol_filter(zmp_y, window_size, 3)

ref_zmp = np.array(zmp_max)-min(zmp_max)
com_zmp_wpg = generateCOM(ref_zmp)
plt.figure()
zmp_offset = 0.00
offset = min(zmp_max)
com_pos_x = np.array(com_pos[:, 0])

zmp_measured = np.array(zmp_measured)
zmp_max = np.array(zmp_max)

com_zmp_wpg = np.array(com_zmp_wpg)

skip = 8
zmp_max_interp = zmp_max[::skip]


plt.plot(t, com_pos_x-min(com_pos_x)+0.06, label="Measured CoM")
plt.plot(t, zmp_measured-min(zmp_measured), label="Measured ZMP")
plt.plot(t[::skip], zmp_max_interp-min(zmp_max_interp),
         label="ZMP reference", alpha=0.8, linestyle="--")
plt.plot(t, com_zmp_wpg-min(com_zmp_wpg),
         label="CoM reference", alpha=0.8, linestyle="--")

plt.xlim([1.0, 5.0])
plt.xlabel("Time [s]")
plt.ylabel("X position [m]")
plt.title("X position over time")
plt.legend()


plt.subplot(2,1,2)
plt.plot(t, com_pos[:,1], label="Y")
plt.plot(t, zmp_y, label="zmp y")
plt.plot(t, cp_y, label="cp y")
plt.plot(t, lfoot[1,:], label="Left y foot")
plt.plot(t, rfoot[1,:], label="Right y foot")
plt.title("Pos")
plt.legend()

plt.figure()
plt.plot(com_pos[:,0], com_pos[:,1], label="COM")
plt.plot(zmp_x, zmp_y, label="zmp")
plt.plot(cp_x, cp_y, label="cp")
plt.title("Pos")
plt.legend()

plt.figure()
plt.subplot(2,1,1)
plt.plot(t, com_vel[:,0], label="X")
plt.plot(t, com_vel_raw[:,0], label="X raw", alpha=0.5)

plt.subplot(2,1,2)
plt.plot(t, com_vel[:,1], label="Y")
plt.plot(t, com_vel_raw[:,1], label="Y raw", alpha=0.5)
plt.title("Vel")

plt.figure()
plt.subplot(2,1,1)
plt.plot(t, com_acc[:,0], label="X")
plt.plot(t, com_acc_raw[:,0], label="Raw X", alpha=0.5)
plt.ylim([min(com_acc[:,0]), max(com_acc[:,0])])

plt.subplot(2,1,2)
plt.plot(t, com_acc[:,1], label="Y")
plt.plot(t, com_acc_raw[:,1], label="Raw Y", alpha=0.5)
plt.title("Acc")
plt.ylim([min(com_acc[:,1]), max(com_acc[:,1])])

multipage("figures.pdf")

print("Saving zmp_x")
offset = min(zmp_max)
np.savetxt("zmp_x.csv", zmp_max-offset, delimiter=",")
