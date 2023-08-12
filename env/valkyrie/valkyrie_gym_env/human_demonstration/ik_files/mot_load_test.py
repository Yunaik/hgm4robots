import numpy as np
import pickle
import matplotlib.pyplot as plt
import scipy
from scipy import signal

name = 's7ik3'
file = open(name+'.mot', 'r')
# print(file.read())
# for line in file:
#     for word in line.split():
#         print(word)
lines = []
for line in file:
    lines.append(line.rstrip('\n'))

data_type = []
motion_data = dict()
words = lines[10].split()#returns list

for word in words:
    data_type.append(word)
    motion_data.update({word:[]})
    print(word)

for line in lines[11:-1]:
    datas = line.split()
    for i in range(len(datas)):
        motion_data[data_type[i]].append(float(datas[i]))

data_freq=100
gait_freq=100.0/len(motion_data['time'])
print(gait_freq)
time = motion_data['time']
print("time: ", time)

joint_angle = dict()
for key in data_type[1:-1]:
    data = motion_data[key]
    joint_angle.update({key:data})
    data = joint_angle[key]
    plt.plot(data, label=key)
    plt.legend()
plt.show()

joint_velocity = dict()
for key in data_type[1:-1]:
    value = joint_angle[key]
    temp = []
    length = len(value)
    temp.extend(value)
    temp.extend(value)
    temp.extend(value)
    vel = []
    temp = np.array(temp)
    temp = temp.astype(float)
    for i in range(length, 2*length):
        v = (temp[i+1]-temp[i])*data_freq
        vel.append(v)
    joint_velocity.update({key:vel})

for key in data_type[1:-1]:
    data = joint_velocity[key]
    plt.plot(data, label=key)
    plt.legend()
plt.show()

filter_joint_vel = dict()
for key in data_type[1:-1]:
    value = joint_velocity[key]
    temp = []
    length = len(value)
    temp.extend(value)
    temp.extend(value)
    temp.extend(value)
    vel = []
    b, a = signal.butter(1,0.25)
    y = signal.filtfilt(b,a,temp)
    vel = y[length:2*length]
    filter_joint_vel.update({key:vel})

for key in data_type[1:-1]:
    data = filter_joint_vel[key]
    plt.plot(data, label=key)
    plt.legend()
plt.show()

key = data_type[7]
plt.plot(filter_joint_vel[key])
plt.plot(motion_data[key])

plt.show()

print(motion_data['pelvis_rotation'])
var_dict = dict()
var_dict.update({'data_freq':data_freq})
var_dict.update({'data_type':data_type})
var_dict.update({'gait_freq':gait_freq})
var_dict.update({'motion_data':motion_data})
var_dict.update({'joint_angle':joint_angle})
var_dict.update(({'joint_velocity':filter_joint_vel}))
output = open(name + '_processed.obj', 'wb')
pickle.dump(var_dict, output)