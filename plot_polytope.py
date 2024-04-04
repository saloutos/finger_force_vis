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
q_test = np.array([-np.pi/2, 0.5, 0.5, 0.5])  # mcr, mcp, pip, dip

T, J  = finger_kinematics(q_test)
p_ee = T[:3, 3, 4]

# from joint space to actuator space, i.e. phidot = Jact*theta_dot
Jact = np.array([[1.0, 0.0, 0.0, 0.0],
                 [-16.38/15.98, 1.0, 0.0, 0.0],
                 [11.48/15.98, 9.98/14.38, 1.0, 0.0],
                 [-8.08/15.98, 5.58/14.38, 9.98/14.38, 1.0]])
# from actuator space to joint space, i.e. thetadot = Jjoint*phi_dot
Jjoint = np.linalg.pinv(Jact)

# torque limits
t_max = 2.0*np.ones(4)  # joint torque limits max and min
t_min = -2.0*np.ones(4)

f_poly = capacity.force_polytope(J, t_min, t_max) # calculate the polytope
f_poly.find_faces()

poly_scale = 0.0005
poly_v = np.array(f_poly.vertices).T
poly_F = np.array(f_poly.face_indices)

poly_v = poly_scale*poly_v + p_ee

# # plotting the polytope

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # draw finger
# visualize_finger(ax, T, obj, 0.02)

# # draw faces and vertices
# plot_polytope(plot=ax, 
#               polytope=f_poly, 
#               label='force', 
#               edge_color='black', 
#               face_color=[0.75,0.0,0.75],
#               alpha = 0.4,
#               center = p_ee,
#               scale = 0.0002,
#               show_vertices = False)

# ax.set_aspect('equal')
# ax.view_init(50, 75)

# plt.legend()
# plt.show()




# visualize finger using plotly
fig = go.Figure()

visualize_finger_plotly(fig, T, obj, 0.02)

fig.add_trace( go.Mesh3d( x=poly_v[:,0], y=poly_v[:,1], z=poly_v[:,2], \
                         i=poly_F[:,0], j=poly_F[:,1], k=poly_F[:,2], \
                            opacity=0.15))

fig.update_scenes(aspectmode='data')

fig.show()
