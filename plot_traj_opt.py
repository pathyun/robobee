
#   plot_traj_opt.py

#   Objective : Plot piecewise polynomial objects collected from TrajectoryOptPlot.cc

#   Input : TODO : Make it as a python function and get called from TrajectoryOptPlot.cc

#   Output : plots of state and input torque

#   Author : Nak-seung Patrick Hyun
#   Date : 07/25/2018
#


import math
import numpy as np
import matplotlib.pyplot as plt

from pydrake.all import (
    RotationMatrix,
    Quaternion,
    RollPitchYaw)

# Make numpy printing prettier

np.set_printoptions(precision=3, suppress=True)

gravity = 9.8; # Needed to compute thrust to weight ratio

show_flag_qd = 1;
show_flag_q=1;
show_flag_control =1;

#-[0] Load data (state, input)
Directory = '/home/patrick/Research/drake/examples/robobee'


x_raw = np.loadtxt(Directory+'/state_trajopt.txt');
u_raw = np.loadtxt(Directory+'/input_trajopt.txt');
t_raw = np.loadtxt(Directory+'/time_col_trajopt.txt');
x_col_raw = np.loadtxt(Directory+'/state_col_trajopt.txt');
u_col_raw = np.loadtxt(Directory+'/input_col_trajopt.txt');

#-[1] Separate time, state, and input

t_values = x_raw[:,0];
num_time = np.size(t_values,0);
num_states = np.size(x_raw,1)-1; # First column is time index
num_input = np.size(u_raw,1)-1;  # First column is time index
num_knot = np.size(t_raw,0);

x_values = np.zeros((num_time,num_states))
u_values = np.zeros((num_time,num_input))
x_values = x_raw[:, 1:];
u_values = u_raw[:, 1:];
knot_values = t_raw;
x_col_values = x_col_raw;
u_col_values = u_col_raw;

#-[2] Ploting the result

num_iteration = num_time
times = t_values
rpy = np.zeros((num_iteration,3)) # Convert to Euler Angle
ubar = np.zeros((num_iteration,num_input)) # Convert to Euler Angle
u = np.zeros((num_iteration,num_input)) # Convert to Euler Angle

# Converting into Roll, Pitch and Yaw
for j in range(0,num_iteration):

    # ubar[:,j]=test_Feedback_Linearization_controller_BS(state_out[:,j])
    ubar[j,:]=u_values[j,:]
    q_temp =x_values[j,3:7]
    # print(np.dot(q_temp.T, q_temp))
    q_temp = q_temp/np.sqrt(np.dot(q_temp.T, q_temp))
    quat_temp = Quaternion(q_temp)    # Quaternion
    R = RotationMatrix(quat_temp)
    rpy[j,:]=RollPitchYaw(R).vector()
    u[j,:]=ubar[j,:]
    # u[0,j]=x_values[7,j] # Control
    
if show_flag_q==1:
    plt.clf()
    fig = plt.figure(1).set_size_inches(6, 6)
    for i in range(0,3):
        # print("i:%d" %i)
        plt.subplot(3, 1, i+1)
        # print("test:", num_state)
        plt.plot(times, x_values[:,i])
        plt.plot(knot_values, x_col_values[:,i], 'go')
        plt.grid(True)
        if i==0:
            plt.ylabel("x (m)")
        elif i==1:
            plt.ylabel("y (m)")
        elif i==2:
            plt.ylabel("z (m)")
    plt.xlabel("Time (s)")
        

    ####- Plot Euler angle
    fig = plt.figure(2).set_size_inches(6, 6)
    for i in range(0,3):
        # print("i:%d" %i)
        plt.subplot(3, 1, i+1)
        # print("test:", num_state)
        plt.plot(times, rpy[:,i])
        plt.grid(True)
        j=i+3
        # plt.ylabel("x[%d]" % j)
        if i==0:
            plt.ylabel("roll (rad)")
        elif i==1:
            plt.ylabel("pitch (rad)")
        elif i==2:
            plt.ylabel("yaw (rad)")
    plt.xlabel("Time (s)")


####- Plot Quaternion

# fig = plt.figure(2).set_size_inches(6, 6)
# for i in range(0,4):
#   # print("i:%d" %i)
#     plt.subplot(4, 1, i+1)
#     # print("test:", num_state)
#     plt.plot(state_log.sample_times(), state_log.data()[i+3, :])
#     plt.grid(True)
#     j=i+3
#     # plt.ylabel("x[%d]" % j)
#     if i==0:
#       plt.ylabel("q0")
#     elif i==1:
#       plt.ylabel("q1")
#     elif i==2:
#       plt.ylabel("q2")
#     elif i==3:
#       plt.ylabel("q3")

if show_flag_qd==1:
    fig = plt.figure(3).set_size_inches(6, 6)
    for i in range(0,3):
        # print("i:%d" %i)
        plt.subplot(3, 1, i+1)
        # print("test:", num_state)
        plt.plot(times, x_values[:,i+7])
        plt.plot(knot_values, x_col_values[:,i+7], 'go')
        plt.grid(True)
        j=i+7
        # plt.ylabel("x[%d]" % j)
        if i==0:
            plt.ylabel("vx (m/s)")
        elif i==1:
            plt.ylabel("vy (m/s)")
        elif i==2:
            plt.ylabel("vz (m/s)")
    plt.xlabel("Time (s)")

    fig = plt.figure(4).set_size_inches(6, 6)
    for i in range(0,3):
        # print("i:%d" %i)
        plt.subplot(3, 1, i+1)
        # print("test:", num_state)
        plt.plot(times, x_values[:,i+10])
        plt.plot(knot_values, x_col_values[:,i+10], 'go')
        plt.grid(True)
        j=i+10
        # plt.ylabel("x[%d]" % j)
        if i==0:
            plt.ylabel("wx (rad/s)")
        elif i==1:
            plt.ylabel("wy (rad/s)")
        elif i==2:
            plt.ylabel("wz (rad/s)")
    plt.xlabel("Time (s)")

if show_flag_control==1:
    fig = plt.figure(5).set_size_inches(6, 6)
    for i in range(0,4):
        # print("i:%d" %i)
        plt.subplot(4, 1, i+1)
        # print("test:", num_state)
        if i==0:
            mg_gain=1/gravity; #  1cm = 0.01m
        else:
            mg_gain=1e0; # 1000mg =1g
        plt.plot(times, u[:,i]*mg_gain)
        plt.plot(knot_values, u_col_values[:,i]*mg_gain, 'go')
        plt.grid(True)
        # plt.ylabel("x[%d]" % j)
        if i==0:
            plt.ylabel("T-W ratio")
        elif i==1:
            plt.ylabel("tau_r (mNmm)")
        elif i==2:
            plt.ylabel("tau_p (mNmm)")
        elif i==3:
            plt.ylabel("tau_y (mNmm)")
    plt.xlabel("Time (s)")
# plt.subplot(num_state, 1, num_state)
# plt.plot(input_log.sample_times(), input_log.data()[0, :])
# plt.ylabel("u[0]")
# plt.xlabel("t")

plt.show()