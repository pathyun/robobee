# Testing Python 
import math
import numpy as np
# from sympy import *

from pydrake.all import (
    DiagramBuilder,
    Simulator,
    SignalLogger,
    VectorSystem,
    RotationMatrix,
    Quaternion,
    RollPitchYaw
    )

##################### 
# Quaternion test
#####################

q = np.zeros((4,1)) # quaternion setup
theta = math.pi/4;  # angle of rotation
v=np.array([1.,1.,0])
q[0]=math.cos(theta/2) #q0
# print("q:",q[0])
v_norm=np.sqrt((np.dot(v,v))) # axis of rotation
v_normalized =v.T/v_norm
# print("vnorm:", v_norm)
# print("vnormalized", np.dot(v_normalized,v_normalized.T))
q[1:4,0]=math.sin(theta/2)*v.T/v_norm
print("q: ", q)
# print("q_norm: ", np.dot(q.T,q))

quat = Quaternion(q)

# unit quaternion to Rotation matrix
R = RotationMatrix(quat)
print("R:", R.matrix())
rpy=RollPitchYaw(R)
print("rpy",rpy.vector())
# print(R.matrix().dtype) # data type of Rotation matrix
# print(np.dot(R.matrix().T,R.matrix()))

# R\in SO(3) check
Iden = np.dot(R.matrix().T,R.matrix())
print("R^TR=:",Iden)

# get quaternion from rotationmatrix
quat_from_rot = RotationMatrix.ToQuaternion(R)
# print("quaternion_from_rotmax:", quat_from_rot.wxyz())

# Get E(q) matrix given unit quaternion q

E = np.zeros((3,4))
E[0,:] = np.array([-1*q[1],    q[0], -1*q[3],    q[2]]).T
E[1,:] = np.array([-1*q[2],    q[3],    q[0], -1*q[1]]).T
E[2,:] = np.array([-1*q[3], -1*q[2],    q[1],    q[0]]).T

print("E(q)^T:", E.T)

# compute quaternion derivative
w= np.zeros((3,1))
w[0]=1.
print("w:", w)
quat_dot = np.dot(E.T,w)/2.   # \dot{quat}=1/2*E(q)^Tw
print("quat_dot:", quat_dot)

##################### 
# Euler equation test
#####################

w=np.zeros((3,1))  		# angular velocity
tau =np.zeros((3,1))	# torque
w[0]=1;
w[1]=2;
w[2]=3;
tau[0] =4;
tau[1] = 5;
tau[2] = 6;

I=np.zeros((3,3)) 		# Intertia matrix
Ixx= 1;
Iyy= 2;
Izz= 3;

I[0,0]=Ixx;
I[1,1]=Iyy;
I[2,2]=Izz;
I_inv = np.linalg.inv(I);

print("I:",I)
print("I^-1",I_inv)
Iw = np.dot(I,w)
print("Iw", Iw)
what = np.cross(np.eye(3),w.T) # Caution, hat operation needs to be flipped. eye(3) first in the argument.

print("what:",what)

wIw_w_hat = np.dot(what,Iw)     # w^Iw
wIw_wo_hat = np.cross(w.T,Iw.T) # w x (Iw)

print("wIw_w_hat:",wIw_w_hat)
print("wIw_wo_hat:", wIw_wo_hat)

w_dot = -np.dot(I_inv,wIw_w_hat)+np.dot(I_inv,tau) # \dot{w}=-I^{-1} w x (Iw) + I^{-1] tau}
print("w_dot:",w_dot)

####################################
# Symbolic test
####################################

# xSym = Symbol('x')
# ySym = Symbol('y')


