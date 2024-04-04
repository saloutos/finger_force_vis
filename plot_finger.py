import numpy as np
import scipy
import scipy.io
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from utils import *

import plotly.graph_objects as go

# Load CAD mesh files
print("loading CAD mesh files (this may take a while)")

n = 4  # number of objects to load
objFileNames = ['finger_meshes/MCR_link.obj',
                'finger_meshes/MCP_link.obj',
                'finger_meshes/PIP_link.obj',
                'finger_meshes/DIP_link_sphere.obj']

# Load all the meshes defined
obj = []
for ii in range(n):
    obj.append(readObj(objFileNames[ii]))

print("loaded CAD mesh files")

# Define the values for q_test, qd_test, and F_test
q_test = np.array([-0.3, 0.4, 0.2, 0.1])  # mcr, mcp, pip, dip
qd_test = np.array([1.0, 0.0, 0.0, 0.0])  # mcr, mcp, pip, dip
F_test = np.array([1.0, 1.0, 0.0])  # x, y, z at end effector

# task space jacobian: v = J*qdot, tau = J^T * F

T, J  = finger_kinematics(q_test)

# from joint space to actuator space, i.e. qdot_phi = Jact*qdot_theta, tau_theta = Jact^T * tau_phi
Jact = np.array([[1.0, 0.0, 0.0, 0.0],
                 [-16.38/15.98, 1.0, 0.0, 0.0],
                 [11.48/15.98, 9.98/14.38, 1.0, 0.0],
                 [-8.08/15.98, 5.58/14.38, 9.98/14.38, 1.0]])
# from actuator space to joint space, i.e. qdot_theta = Jjoint*qdot_phi                                                                                                                                                                                                             , tau_phi = Jjoint^T * tau_theta
Jjoint = np.linalg.pinv(Jact)

# Calculate velocities
v_test = np.dot(J, qd_test)

# Calculate torques
tau_test = np.dot(J.T, F_test)

# Calculate actuator torques
tau_act_test = np.dot(Jjoint.T, tau_test)

# # Visualize
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# visualize_finger(ax, T, obj, 0.02)

# # plot ee vel and joint vels
# v_scale = 0.5
# show_ee_vel_joint_vels(ax, T, v_test, qd_test, v_scale)

# # plot force and torques
# # f_scale = 0.5
# # show_ee_force_joint_torques(ax, T, F_test, tau_test, f_scale)

# ax.set_aspect('equal')
# ax.view_init(50, 75)

# plt.show()

# visualize finger using plotly
fig = go.Figure()
visualize_finger_plotly(fig, T, obj, 0.02)
fig.show()