import numpy as np
import scipy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import plotly.graph_objects as go

# helper functions
class Obj:
    def __init__(self):
        self.v = None
        self.vt = None
        self.vn = None
        self.f = {'v': [], 'vt': [], 'vn': []}

def readObj(fname):
    obj = Obj()
    v = []
    vt = []
    vn = []

    with open(fname, 'r') as fid:
        for line in fid:
            line = line.strip()
            if not line:
                continue

            tokens = line.split()
            line_type = tokens[0]

            if line_type == 'v':  # mesh vertices
                v.append([float(x) for x in tokens[1:]])
            elif line_type == 'vt':  # texture coordinates
                vt.append([float(x) for x in tokens[1:]])
            elif line_type == 'vn':  # normal coordinates
                vn.append([float(x) for x in tokens[1:]])
            elif line_type == 'f':  # face definition
                fv = []
                fvt = []
                fvn = []

                for token in tokens[1:]:
                    indices = token.split('/')
                    fv.append(int(indices[0])-1) # subtract 1, since this is an index
                    if len(indices) > 1 and indices[1]:
                        fvt.append(int(indices[1])-1) # subtract 1, since this is an index
                    if len(indices) > 2 and indices[2]:
                        fvn.append(int(indices[2])-1) # subtract 1, since this is an index

                obj.f['v'].append(fv)
                obj.f['vt'].append(fvt)
                obj.f['vn'].append(fvn)

    obj.v = np.array(v)
    obj.vt = np.array(vt)
    obj.vn = np.array(vn)

    return obj


def draw_SE3(axs, T, scale=None, string_color=None):
    p = T[:3, 3]
    if scale is None:
        ax = p + T[:3, 0] * 0.12
        ay = p + T[:3, 1] * 0.12
        az = p + T[:3, 2] * 0.12
    else:
        ax = p + T[:3, 0] * scale
        ay = p + T[:3, 1] * scale
        az = p + T[:3, 2] * scale
    
    if string_color is None:
        axs.plot([p[0], ax[0]], [p[1], ax[1]], [p[2], ax[2]], 'black', linewidth=1.5)
        axs.plot([p[0], ay[0]], [p[1], ay[1]], [p[2], ay[2]], 'black', linewidth=1.5)
        axs.plot([p[0], az[0]], [p[1], az[1]], [p[2], az[2]], 'black', linewidth=1.5)
    elif string_color == 'rgb':
        axs.plot([p[0], ax[0]], [p[1], ax[1]], [p[2], ax[2]], 'red', linewidth=1.5)
        axs.plot([p[0], ay[0]], [p[1], ay[1]], [p[2], ay[2]], 'green', linewidth=1.5)
        axs.plot([p[0], az[0]], [p[1], az[1]], [p[2], az[2]], 'blue', linewidth=1.5)
    else:
        axs.plot([p[0], ax[0]], [p[1], ax[1]], [p[2], ax[2]], string_color, linewidth=1.5)
        axs.plot([p[0], ay[0]], [p[1], ay[1]], [p[2], ay[2]], string_color, linewidth=1.5)
        axs.plot([p[0], az[0]], [p[1], az[1]], [p[2], az[2]], string_color, linewidth=1.5)

def Rx(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    R = np.array([[1, 0, 0],
                  [0, c, -s],
                  [0, s, c]]) 
    return R

def Ry(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    R = np.array([[c, 0, s],
                  [0, 1, 0],
                  [-s, 0, c]])
    return R

def Rz(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    R = np.array([[c, -s, 0],
                  [s, c, 0],
                  [0, 0, 1]])
    return R

def random_three_vector(n=1):
    """
    Generates a random 3D unit vector (direction) with a uniform spherical distribution
    Algo from http://stackoverflow.com/questions/5408276/python-uniform-spherical-distribution
    :return:
    """
    vecs = np.zeros((n,3))
    for i in range(n):
        phi = np.random.uniform(0,np.pi*2)
        costheta = np.random.uniform(-1,1)

        theta = np.arccos( costheta )
        vecs[i,0] = np.sin( theta) * np.cos( phi ) # x
        vecs[i,1] = np.sin( theta) * np.sin( phi ) # y
        vecs[i,2] = np.cos( theta ) # z

    return vecs

def visualize_finger(axs, T, obj, SE3_scale):
    
    # Draw origin
    draw_SE3(axs, np.eye(4), SE3_scale)
    
    # Draw each body and its frame
    for ii in range(4):
        draw_SE3(axs, T[:, :, ii], SE3_scale, 'rgb')
        V = np.dot(T[:3, :3, ii], obj[ii].v.T).T + T[:3, 3, ii]
        F = np.array(obj[ii].f['v'])
        axs.plot_trisurf(V[:, 0], V[:, 1], V[:, 2], triangles=F, color=(0, 0, 0, 0.15))
    
    # Draw end-effector frame
    draw_SE3(axs, T[:, :, 4], SE3_scale, 'rgb')

    axs.set_xlabel('X')
    axs.set_ylabel('Y')
    axs.set_zlabel('Z')
    axs.grid(True)

def visualize_finger_plotly(fig, T, obj, alpha=1.0, SE3_scale=None):
    # Draw each body and its frame
    for ii in range(4):
        V = np.dot(T[:3, :3, ii], obj[ii].v.T).T + T[:3, 3, ii]
        F = np.array(obj[ii].f['v'])
        fig.add_trace( go.Mesh3d( x=V[:,0], y=V[:,1], z=V[:,2], \
                                i = F[:,0], j=F[:,1], k=F[:,2], \
                                opacity=alpha, \
                                color="#808080") )
        draw_SE3_plotly(fig, T[:,:,ii], scale=SE3_scale)
    # draw end effector frame too
    draw_SE3_plotly(fig, T[:,:,4], scale=SE3_scale)


def draw_SE3_plotly(fig, T, scale=None):
    p = T[:3, 3]
    if scale is None:
        ax = p + T[:3, 0] * 0.12
        ay = p + T[:3, 1] * 0.12
        az = p + T[:3, 2] * 0.12
    else:
        ax = p + T[:3, 0] * scale
        ay = p + T[:3, 1] * scale
        az = p + T[:3, 2] * scale
    # add x-axis
    fig.add_trace( go.Scatter3d(x=[p[0],ax[0]],y=[p[1],ax[1]],z=[p[2],ax[2]], \
                                    line_color='red', line_width=4, \
                                    mode='lines', showlegend=False ) )
    # add y-axis
    fig.add_trace( go.Scatter3d(x=[p[0],ay[0]],y=[p[1],ay[1]],z=[p[2],ay[2]], \
                                    line_color='green', line_width=4, \
                                    mode='lines', showlegend=False ) )
    # add z-axis
    fig.add_trace( go.Scatter3d(x=[p[0],az[0]],y=[p[1],az[1]],z=[p[2],az[2]], \
                                    line_color='blue', line_width=4, \
                                    mode='lines',showlegend=False ) )



def show_ee_vel_joint_vels(axs, T, v, qd, scale):
    # show velocity vector (and joint vels)
    # End effector velocity
    p_ee = T[:3, 3, 4]
    v_plt = scale * v
    axs.quiver(p_ee[0], p_ee[1], p_ee[2], v_plt[0], v_plt[1], v_plt[2], color=[0.6, 0.0, 0.75])
    # MCR joint velocity
    p_plt = T[:3, 3, 0]
    ax_plt = T[:3, 1, 0]
    qd_plt = 0.1 * scale * qd[0] * ax_plt
    axs.quiver(p_plt[0], p_plt[1], p_plt[2], qd_plt[0], qd_plt[1], qd_plt[2], color=[1.0, 0.65, 0.0])
    # MCP joint velocity
    p_plt = T[:3, 3, 1]
    ax_plt = T[:3, 2, 1]
    qd_plt = 0.1 * scale * qd[1] * ax_plt
    axs.quiver(p_plt[0], p_plt[1], p_plt[2], qd_plt[0], qd_plt[1], qd_plt[2], color=[1.0, 0.65, 0.0])
    # PIP joint velocity
    p_plt = T[:3, 3, 2]
    ax_plt = T[:3, 2, 2]
    qd_plt = 0.1 * scale * qd[2] * ax_plt
    axs.quiver(p_plt[0], p_plt[1], p_plt[2], qd_plt[0], qd_plt[1], qd_plt[2], color=[1.0, 0.65, 0.0])
    # DIP joint velocity
    p_plt = T[:3, 3, 3]
    ax_plt = T[:3, 2, 3]
    qd_plt = 0.1 * scale * qd[3] * ax_plt
    axs.quiver(p_plt[0], p_plt[1], p_plt[2], qd_plt[0], qd_plt[1], qd_plt[2], color=[1.0, 0.65, 0.0])

def show_ee_force_joint_torques(axs, T, F, tau, scale): 
    # show force vector (and joint torques)

    # End effector force
    p_ee = T[:3, 3, 4]
    F_plt = 0.1 * scale * F
    axs.quiver(p_ee[0], p_ee[1], p_ee[2], F_plt[0], F_plt[1], F_plt[2], color=[0.6, 0.0, 0.75])
    # MCR joint torque
    p_plt = T[:3, 3, 0]
    ax_plt = T[:3, 1, 0]
    tau_plt = scale * tau[0] * ax_plt
    axs.quiver(p_plt[0], p_plt[1], p_plt[2], tau_plt[0], tau_plt[1], tau_plt[2], color=[1.0, 0.65, 0.0])
    # MCP joint torque
    p_plt = T[:3, 3, 1]
    ax_plt = T[:3, 2, 1]
    tau_plt = scale * tau[1] * ax_plt
    axs.quiver(p_plt[0], p_plt[1], p_plt[2], tau_plt[0], tau_plt[1], tau_plt[2], color=[1.0, 0.65, 0.0])
    # PIP joint torque
    p_plt = T[:3, 3, 2]
    ax_plt = T[:3, 2, 2]
    tau_plt = scale * tau[2] * ax_plt
    axs.quiver(p_plt[0], p_plt[1], p_plt[2], tau_plt[0], tau_plt[1], tau_plt[2], color=[1.0, 0.65, 0.0])
    # DIP joint torque
    p_plt = T[:3, 3, 3]
    ax_plt = T[:3, 2, 3]
    tau_plt = scale * tau[3] * ax_plt
    axs.quiver(p_plt[0], p_plt[1], p_plt[2], tau_plt[0], tau_plt[1], tau_plt[2], color=[1.0, 0.65, 0.0])


def finger_kinematics(q, T0=np.eye(4)):
    # link lengths and offsets
    l0 = 0.0185
    l1 = 0.050
    l2 = 0.040
    l3 = 0.032 # 0.028

    # link transforms
    Tmcr_l = T0
    Tmcp_l = np.eye(4)
    Tmcp_l[0, 3] = l0
    Tpip_l = np.eye(4)
    Tpip_l[0, 3] = l1
    Tdip_l = np.eye(4)
    Tdip_l[0, 3] = l2
    T_ee = np.eye(4)
    T_ee[0, 3] = l3

    # joint transforms
    Tmcr_j = np.eye(4)
    Tmcr_j[:3, :3] = Ry(q[0])
    Tmcp_j = np.eye(4)
    Tmcp_j[:3, :3] = Rz(q[1])
    Tpip_j = np.eye(4)
    Tpip_j[:3, :3] = Rz(q[2])
    Tdip_j = np.eye(4)
    Tdip_j[:3, :3] = Rz(q[3])

    # return transforms for outputs of each joint
    T = np.zeros((4, 4, 5))
    T[:, :, 0] = np.dot(Tmcr_l, Tmcr_j)
    T[:, :, 1] = np.dot(T[:, :, 0], np.dot(Tmcp_l, Tmcp_j))
    T[:, :, 2] = np.dot(T[:, :, 1], np.dot(Tpip_l, Tpip_j))
    T[:, :, 3] = np.dot(T[:, :, 2], np.dot(Tdip_l, Tdip_j))
    T[:, :, 4] = np.dot(T[:, :, 3], T_ee)

    # symbolically derived jacobian
    J = np.zeros((3, 4))  # from joint space to task space, i.e. xdot = J*qdot

    J[0, 0] = -(l1 * np.cos(q[1]) + l2 * np.cos(q[1] + q[2]) + l3 * np.cos(q[1] + q[2] + q[3]) + l0) * np.sin(q[0])
    J[0, 1] = -(l1 * np.sin(q[1]) + l2 * np.sin(q[1] + q[2]) + l3 * np.sin(q[1] + q[2] + q[3])) * np.cos(q[0])
    J[0, 2] = -(l2 * np.sin(q[1] + q[2]) + l3 * np.sin(q[1] + q[2] + q[3])) * np.cos(q[0])
    J[0, 3] = -l3 * np.sin(q[1] + q[2] + q[3]) * np.cos(q[0])

    J[1, 0] = 0
    J[1, 1] = l1 * np.cos(q[1]) + l2 * np.cos(q[1] + q[2]) + l3 * np.cos(q[1] + q[2] + q[3])
    J[1, 2] = l2 * np.cos(q[1] + q[2]) + l3 * np.cos(q[1] + q[2] + q[3])
    J[1, 3] = l3 * np.cos(q[1] + q[2] + q[3])

    J[2, 0] = -(l1 * np.cos(q[1]) + l2 * np.cos(q[1] + q[2]) + l3 * np.cos(q[1] + q[2] + q[3]) + l0) * np.cos(q[0])
    J[2, 1] = (l1 * np.sin(q[1]) + l2 * np.sin(q[1] + q[2]) + l3 * np.sin(q[1] + q[2] + q[3])) * np.sin(q[0])
    J[2, 2] = (l2 * np.sin(q[1] + q[2]) + l3 * np.sin(q[1] + q[2] + q[3])) * np.sin(q[0])
    J[2, 3] = l3 * np.sin(q[0]) * np.sin(q[1] + q[2] + q[3])

    return T, J

