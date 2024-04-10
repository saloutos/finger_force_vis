import numpy as np
import scipy
import scipy.io
import matplotlib.pyplot as plt
from utils import *

import plotly.graph_objects as go

import pycapacity.robot as capacity




# Load CAD mesh files
print("loading CAD mesh files (this may take a while)")

n = 4  # number of objects to load
objFileNames = ['finger_meshes/MCR_link.obj',
                'finger_meshes/MCP_link.obj',
                'finger_meshes/PIP_link.obj',
                'finger_meshes/DIP_link_sphere.obj']

# Load all the meshes defined
obj_right = []
obj_left = []
for ii in range(n):
    new_obj_right = readObj(objFileNames[ii])
    obj_right.append(new_obj_right)
    new_obj_left = readObj(objFileNames[ii])
    new_obj_left.v[:,1] = -1.0*new_obj_left.v[:,1]
    obj_left.append(new_obj_left)

print("loaded CAD mesh files")


# Left finger
q_left = np.array([0.2, 0.7, -0.7, -0.7]) # mcr, mcp, pip, dip
T0_left = np.eye(4)
T0_left[1,3] = 0.06
T_left, J_left = finger_kinematics(q_left, T0_left)
p_ee_left = T_left[:3,3,4]

# Right finger
q_right = np.array([-0.2, -0.7, 0.7, 0.7]) # mcr, mcp, pip, dip
T0_right = np.eye(4)
T0_right[1,3] = -0.06
T_right, J_right = finger_kinematics(q_right, T0_right)
p_ee_right = T_right[:3,3,4]

# from joint space to actuator space, i.e. phidot = Jact*theta_dot
Jact = np.array([[1.0, 0.0, 0.0, 0.0],
                 [-16.38/15.98, 1.0, 0.0, 0.0],
                 [11.48/15.98, 9.98/14.38, 1.0, 0.0],
                 [-8.08/15.98, 5.58/14.38, 9.98/14.38, 1.0]])
# from actuator space to joint space, i.e. thetadot = Jjoint*phi_dot
Jjoint = np.linalg.pinv(Jact)

# joint torque limits
t_max = 2.5*np.ones(4)  # joint torque limits max and min
t_min = -2.5*np.ones(4)

# polytopes for left finger
f_poly_j_l = capacity.force_polytope(J_left, t_min, t_max) # calculate the polytope
f_poly_j_l.find_faces()

# actuator torque limits
Jstar_l = J_left @ Jjoint
f_poly_a_l = capacity.force_polytope(Jstar_l, t_min, t_max)
f_poly_a_l.find_faces()
f_poly_a_l.find_halfplanes()

# prepare for plotting
poly_scale = 0.0005
poly_j_v_l = np.array(f_poly_j_l.vertices).T
poly_j_F_l = np.array(f_poly_j_l.face_indices)
poly_j_v_l = poly_scale*poly_j_v_l + p_ee_left

poly_a_v_l = np.array(f_poly_a_l.vertices).T
poly_a_F_l = np.array(f_poly_a_l.face_indices)
poly_a_v_l = poly_scale*poly_a_v_l + p_ee_left



# polytopes for right finger
f_poly_j_r = capacity.force_polytope(J_right, t_min, t_max) # calculate the polytope
f_poly_j_r.find_faces()

# actuator torque limits
Jstar_r = J_right @ Jjoint
f_poly_a_r = capacity.force_polytope(Jstar_r, t_min, t_max)
f_poly_a_r.find_faces()
f_poly_a_r.find_halfplanes()

# prepare for plotting
poly_scale = 0.0005
poly_j_v_r = np.array(f_poly_j_r.vertices).T
poly_j_F_r = np.array(f_poly_j_r.face_indices)
poly_j_v_r = poly_scale*poly_j_v_r + p_ee_right

poly_a_v_r = np.array(f_poly_a_r.vertices).T
poly_a_F_r = np.array(f_poly_a_r.face_indices)
poly_a_v_r = poly_scale*poly_a_v_r + p_ee_right


# Minkowski sum of the actuator polytopes
f_poly_a_sum = f_poly_a_l + f_poly_a_r
f_poly_a_sum.find_faces()
f_poly_a_sum.find_halfplanes()
# intersection of the actuator polytopes
f_poly_a_int = f_poly_a_l & f_poly_a_r
f_poly_a_int.find_faces()
f_poly_a_int.find_halfplanes()

# prepare for plotting
poly_scale = 0.0005
sum_poly_a_v = np.array(f_poly_a_sum.vertices).T
sum_poly_a_F = np.array(f_poly_a_sum.face_indices)
sum_poly_a_v = poly_scale*sum_poly_a_v + 0.5*(p_ee_right+p_ee_left)

int_poly_a_v = np.array(f_poly_a_int.vertices).T
int_poly_a_F = np.array(f_poly_a_int.face_indices)
int_poly_a_v = poly_scale*int_poly_a_v + 0.5*(p_ee_right+p_ee_left)




# visualize finger using plotly
fig = go.Figure()

visualize_finger_plotly(fig, T_left, obj_left, alpha=0.35, SE3_scale=0.01)

fig.add_trace( go.Mesh3d( x=poly_j_v_l[:,0], y=poly_j_v_l[:,1], z=poly_j_v_l[:,2], \
                            i=poly_j_F_l[:,0], j=poly_j_F_l[:,1], k=poly_j_F_l[:,2], \
                            opacity=0.15, color="#AB63FA", \
                            name='joint', showlegend=True))

fig.add_trace( go.Mesh3d( x=poly_a_v_l[:,0], y=poly_a_v_l[:,1], z=poly_a_v_l[:,2], \
                            i=poly_a_F_l[:,0], j=poly_a_F_l[:,1], k=poly_a_F_l[:,2], \
                            opacity=0.15, color="#00CC96", \
                            name='actuator', showlegend=True))

visualize_finger_plotly(fig, T_right, obj_right, alpha=0.35, SE3_scale=0.01)

fig.add_trace( go.Mesh3d( x=poly_j_v_r[:,0], y=poly_j_v_r[:,1], z=poly_j_v_r[:,2], \
                            i=poly_j_F_r[:,0], j=poly_j_F_r[:,1], k=poly_j_F_r[:,2], \
                            opacity=0.15, color="#AB63FA", \
                            name='joint', showlegend=False))

fig.add_trace( go.Mesh3d( x=poly_a_v_r[:,0], y=poly_a_v_r[:,1], z=poly_a_v_r[:,2], \
                            i=poly_a_F_r[:,0], j=poly_a_F_r[:,1], k=poly_a_F_r[:,2], \
                            opacity=0.15, color="#00CC96", \
                            name='actuator', showlegend=False))

# draw origin too
draw_SE3_plotly(fig, np.eye(4), 0.01)

# improve aspect ratio
fig.update_scenes(aspectmode='data')

fig.add_trace( go.Mesh3d( x=sum_poly_a_v[:,0], y=sum_poly_a_v[:,1], z=sum_poly_a_v[:,2], \
                            i=sum_poly_a_F[:,0], j=sum_poly_a_F[:,1], k=sum_poly_a_F[:,2], \
                            opacity=0.15, color="red", \
                            name='sum', showlegend=True))

fig.add_trace( go.Mesh3d( x=int_poly_a_v[:,0], y=int_poly_a_v[:,1], z=int_poly_a_v[:,2], \
                            i=int_poly_a_F[:,0], j=int_poly_a_F[:,1], k=int_poly_a_F[:,2], \
                            opacity=0.15, color="blue", \
                            name='int', showlegend=True))

fig.show()