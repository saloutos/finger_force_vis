# algorithm:

# generate list of uniformly distributed vectors on unit sphere

# given joint configuration, calculate task space jacobian
# for each unit vector, find force magnitude, lambda by solving linear program
# note, could this be parallelized?

# problem: max over lambda of just lambda, s.t. tau_act_neg_limit <= Jjoint^T * J^T * lambda* u <= tau_act_pos_limit
# note, might want a constraint on lambda? especially lambda>0?

# if the solution is valid, save vertex lambda*u as a point on the boundary of the feasible force polytope

# finally, use vertices to plot polytope
# might need to find a way to create convex hull for plotting, look into find_faces() method from pycapacity lib

# note: wrap optimization as a function so that we can quickly find maximum force in any direction in any pose?

import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.io
import plotly.graph_objects as go
import pycapacity.robot as capacity
import scipy.optimize
import scipy.optimize
from scipy.spatial import ConvexHull
import time
from utils import *

# constants
# torque limit in actuator space
tau_lim = 2.5
F_lim = 100000
# from joint space to actuator space, i.e. phidot = Jact*theta_dot
Jact = np.array([[1.0, 0.0, 0.0, 0.0],
                 [-16.38/15.98, 1.0, 0.0, 0.0],
                 [11.48/15.98, 9.98/14.38, 1.0, 0.0],
                 [-8.08/15.98, 5.58/14.38, 9.98/14.38, 1.0]])
# from actuator space to joint space, i.e. thetadot = Jjoint*phi_dot
Jjoint = np.linalg.pinv(Jact)




# choose joint angle
q = np.array([-np.pi/2, 0.5, 0.5, 0.5])  # mcr, mcp, pip, dip
# calculate task-space jacobian
T, Jtask = finger_kinematics(q)
# generate some unit vectors
n = 1000
vecs = random_three_vector(n).T

force_mags = []
forces = []
loop_start = time.time()
for i in range(n):
    u = vecs[:,i].reshape((3,1))
    # solve lin prog for each unit vector to get force magnitude
    c =np.array([-1.0])
    A_up = Jjoint.T @ Jtask.T @ u
    A_low = -1.0*A_up
    b_up = np.array([tau_lim, tau_lim, tau_lim, tau_lim])
    b_low = -1.0*-1.0*b_up
    A_ub = np.vstack((A_up,A_low))
    b_ub = np.concatenate((b_up,b_low))
    start = time.time()
    f_sol = scipy.optimize.linprog(c,A_ub,b_ub,bounds=(0,F_lim))
    end = time.time()-start
    if f_sol.success==True:
        force_mags.append(f_sol.x[0])
        force = f_sol.x[0] * u.T
        forces.append(force.squeeze())
forces = np.array(forces)
loop_end = time.time()-loop_start
print(loop_end)

# calculate convex hull for opt results
force_hull = ConvexHull(forces)
force_vertices = force_hull.points
force_faces = force_hull.simplices

# now, calculate vertices based on mapping combinations of torque limits
tau_lims = np.array([-tau_lim,tau_lim])
potential_act_limits = np.array(np.meshgrid(tau_lims,tau_lims,tau_lims,tau_lims)).T.reshape(-1,4).T
potential_joint_limits = Jact.T @ potential_act_limits
Jtinv = np.linalg.pinv(Jtask.T)
potential_force_limits = Jtinv @ potential_joint_limits

potential_hull = ConvexHull(potential_force_limits.T)
potential_vertices = potential_hull.points
potential_faces = potential_hull.simplices

# now, try to use pycapacity library
t_max = tau_lim*np.ones(4)
t_min = -tau_lim*np.ones(4)
Jstar = Jtask @ Jjoint
f_poly = capacity.force_polytope(Jstar, t_min, t_max)
f_poly.find_faces()
f_poly.find_halfplanes()
poly_v = np.array(f_poly.vertices).T
poly_F = np.array(f_poly.face_indices)

# try to find boundary points along sampled vectors
p_force_mags = []
p_forces = []
loop_start = time.time()
for i in range(n):
    u = vecs[:,i].reshape((3,1))
    # solve lin prog for each unit vector to get force magnitude
    p_sol = scipy.optimize.linprog(-1.0, f_poly.H @ u, f_poly.d)
    end = time.time()-start
    if p_sol.success==True:
        p_force_mags.append(p_sol.x[0])
        p_force = p_sol.x[0] * u.T
        p_forces.append(p_force.squeeze())
p_forces = np.array(p_forces)
loop_end = time.time()-loop_start
print(loop_end)

# now, plot the hulls
fig = go.Figure()

fig.add_trace( go.Scatter3d(x=forces[:,0],y=forces[:,1],z=forces[:,2], mode='markers' ) )


fig.add_trace( go.Mesh3d( x=force_vertices[:,0], y=force_vertices[:,1], z=force_vertices[:,2], \
                            i=force_faces[:,0], j=force_faces[:,1], k=force_faces[:,2], \
                            opacity=0.15) )

# fig.add_trace( go.Mesh3d( x=potential_vertices[:,0], y=potential_vertices[:,1], z=potential_vertices[:,2], \
#                             i=potential_faces[:,0], j=potential_faces[:,1], k=potential_faces[:,2], \
#                             opacity=0.15) )

fig.add_trace( go.Mesh3d( x=poly_v[:,0], y=poly_v[:,1], z=poly_v[:,2], \
                         i=poly_F[:,0], j=poly_F[:,1], k=poly_F[:,2], \
                            opacity=0.15))

fig.add_trace( go.Scatter3d(x=p_forces[:,0],y=p_forces[:,1],z=p_forces[:,2], mode='markers' ) )

fig.update_scenes(aspectmode='data')

fig.show()
