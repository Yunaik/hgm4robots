import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import copy
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

lfoot = np.genfromtxt('lfoot.csv', delimiter='')
rfoot = np.genfromtxt('rfoot.csv', delimiter='')

threshold = 0.11
lfoot_pos = []
for idx,lpos in enumerate(lfoot[0,:]):
    if lfoot[2,idx] < threshold:
        lfoot_pos.append(lpos)
rfoot_pos = []
for idx,rpos in enumerate(rfoot[0,:]):
    if rfoot[2,idx] < threshold:
        rfoot_pos.append(rpos)

zmp = []
for idx in range(len(lfoot[0,:])):
    if lfoot[0,idx] > rfoot[0,idx]:
        zmp.append(lfoot[0,idx])
    else:
        zmp.append(rfoot[0,idx])
        
plt.figure()
plt.subplot(3,1,1)
plt.plot(lfoot[0,:], label="Left x")
plt.plot(rfoot[0,:], label="Right x")
plt.plot(zmp, label="ZMP")
plt.legend()

plt.subplot(3,1,2)
plt.plot(lfoot_pos, label="Left x")
plt.plot(rfoot_pos, label="Right x")
plt.legend()

plt.subplot(3,1,3)
plt.plot(lfoot[2,:], label="Left z")
plt.plot(rfoot[2,:], label="Right z")
plt.title("Pos")
plt.legend()

multipage("figures.pdf")

