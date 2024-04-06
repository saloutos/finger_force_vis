import numpy as np
import scipy
import scipy.io
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utils import *

# import polytope lib
import pycapacity.robot as capacity
from pycapacity.visual import * # pycapacity visualisation tools

import plotly.graph_objects as go

# from joint space to actuator space, i.e. phidot = Jact*theta_dot
Jact = np.array([[1.0, 0.0, 0.0, 0.0],
                 [-16.38/15.98, 1.0, 0.0, 0.0],
                 [11.48/15.98, 9.98/14.38, 1.0, 0.0],
                 [-8.08/15.98, 5.58/14.38, 9.98/14.38, 1.0]])
# from actuator space to joint space, i.e. thetadot = Jjoint*phi_dot
Jjoint = np.linalg.pinv(Jact)

# torque limits
t_max = 2.5*np.ones(4)  # actuator torque limits max and min
t_min = -2.5*np.ones(4)

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

# from experiment, finger poses and force directions
finger_poses = np.array([[0.15, 0.30, -0.82, -1.13],
                        [0.0, 0.23, -0.44, -1.13],
                        [-0.15, 0.30, -0.82, -1.13],
                        [0.0, 0.6, -1.3, -0.9],
                        [0.0, 0.55, -0.8, -1.27]])
single_finger_pose = -1.0*np.mean(finger_poses, axis=0) # mcr, mcp, pip, dip

force_directions = np.array([[0, 0, 1.0],
                            [-1.0, 0, 0],
                            [0, 0, -1.0],
                            [1.0, 0, 0],
                            [0, -1.0, 0],
                            [0, 1.0, 0]])

T, Jtask  = finger_kinematics(single_finger_pose)
p_ee = T[:3, 3, 4]

# calculate polytope based on actuator torque limits
Jstar = Jtask @ Jjoint
f_poly_a = capacity.force_polytope(Jstar, t_min, t_max)
f_poly_a.find_faces()
f_poly_a.find_halfplanes()

# find boundary points along given force directions
p_force_mags = []
p_forces = []
for i in range(np.shape(force_directions)[0]):
    u = force_directions[i,:].reshape((3,1))

    # solve lin prog for each unit vector to get force magnitude
    p_sol = scipy.optimize.linprog(-1.0, f_poly_a.H @ u, f_poly_a.d)
    if p_sol.success==True:
        p_force_mags.append(p_sol.x[0])
        p_force = p_sol.x[0] * u.T
        p_forces.append(p_force.squeeze())
p_forces = np.array(p_forces)

print(p_force_mags)

# prepare for plotting
poly_scale = 0.0005
poly_a_v = np.array(f_poly_a.vertices).T
poly_a_F = np.array(f_poly_a.face_indices)
poly_a_v = poly_scale*poly_a_v + p_ee

p_f_v = poly_scale*p_forces + p_ee


# visualize finger using plotly
fig = go.Figure()

visualize_finger_plotly(fig, T, obj, alpha=0.5)

fig.add_trace( go.Mesh3d( x=poly_a_v[:,0], y=poly_a_v[:,1], z=poly_a_v[:,2], \
                         i=poly_a_F[:,0], j=poly_a_F[:,1], k=poly_a_F[:,2], \
                            opacity=0.15))

fig.add_trace( go.Scatter3d(x=p_f_v[:,0],y=p_f_v[:,1],z=p_f_v[:,2], mode='markers' ) )

fig.update_scenes(aspectmode='data')

fig.show()
