# load a network
from epynet import Network
import casadi as ca
import numpy as np
import math

# Create a water network model
network = Network("XXX.inp")  # Load EPANET file

# Read parameters
node_uid_to_index = {}
for n, node in enumerate(network.nodes):
    node_uid_to_index[node.uid] = n

L = np.zeros(len(network.pipes))
C = np.zeros(len(network.pipes))
E = np.zeros(len(network.pipes))
D = np.zeros(len(network.pipes))
Q_nom = np.zeros(len(network.pipes))
for n, pipe in enumerate(network.pipes):
    L[n] = pipe.length
    D[n] = pipe.diameter / 1000  # Conversion of mm to m
    Q_nom[n] = 0.1  # Initial guess
    if pipe.roughness > 40:
        C[n] = pipe.roughness
    else: 
        E[n] = pipe.roughness * 1e-3

d = np.zeros(len(network.nodes))
for n, junction in enumerate(network.junctions):
    d[n] = junction.basedemand / 1000   # Conversion of l/sec to m3/s

# Setting generic constants
n_theta_steps = 10
sol = 0
eps = 1e-12
g = 9.81
pi = math.pi
kvisc = 1e-6    # kinematic viscosity (of water at 20 degrees C) m^2/s
smpa_a = 1e-12  # Smoothing parameter a
smpa_d = 1e-12  # Smoothing parameter d

# Smoothed absolute value function
sabs = lambda x: ca.sqrt(x ** 2 + eps)

for theta in np.linspace(0.0, 1.0, n_theta_steps):

    # Symbols
    Q = ca.MX.sym("Q", (len(network.pipes)))
    H = ca.MX.sym("H", (len(network.nodes)))

    # Set boundaries
    start = []
    for n, reservoir in enumerate(network.reservoirs):
        start.append(H[node_uid_to_index[reservoir.uid]] - reservoir.elevation)

    # Setting dictionary for Mass Conservation equations
    mc = {}  
    for n, junction in enumerate(network.junctions):
        mc[junction.uid] = -d[n]

    eq = []

    if C[0] > 40:
        # Hazen-Williams equations each pipe
        for n, pipe in enumerate(network.pipes):
            eq.append(
                (
                    (
                        H[node_uid_to_index[pipe.upstream_node.uid]]
                        - H[node_uid_to_index[pipe.downstream_node.uid]]
                    )
                    / L[n]
                    - (
                        10.67
                        * C[n] ** -1.852
                        * D[n] ** -4.8704
                        * Q_nom[n] ** 1.852
                    )
                    - (
                        10.67
                        * 1.852
                        * C[n] ** -1.852
                        * D[n] ** -4.8704
                        * Q_nom[n] ** 0.852
                    )
                    * (Q[n] - Q_nom[n])
                )
                * (1 - theta)
                + (
                    (
                        H[node_uid_to_index[pipe.upstream_node.uid]]
                        - H[node_uid_to_index[pipe.downstream_node.uid]]
                    )
                    / L[n]
                    - (
                        10.67
                        * C[n] ** -1.852
                        * D[n] ** -4.8704
                        * Q[n]
                        * sabs(Q[n]) ** 0.852
                    )
                )
                * theta
            )

            # Mass conservation equations each node
            if pipe.upstream_node.uid in mc:
                mc[pipe.upstream_node.uid] -= Q[n]
            if pipe.downstream_node.uid in mc:
                mc[pipe.downstream_node.uid] += Q[n]

    else:
        # Initialize DW approximation
        alpha = np.zeros(len(network.pipes))
        beta = np.zeros(len(network.pipes))
        delta = np.zeros(len(network.pipes))
        labda = np.zeros(len(network.pipes))
        b = np.zeros(len(network.pipes))
        c = np.zeros(len(network.pipes))
        gamma = np.zeros(len(network.pipes))
        for n, pipe in enumerate(network.pipes):
            alpha[n] = 2.51 / (4 / (pi * kvisc * D[n]))
            beta[n] = E[n] / D[n] / 3.71
            delta[n] = 2 * alpha[n] / (beta[n] * np.log(10))
            labda[n] = (2 * np.log10(beta[n])) ** -2
            b[n] = 2 * delta[n]
            c[n] = (np.log10(beta[n]) + 1) * delta[n] ** 2 - smpa_a ** 2 / 2
            gamma[n] = 8 * L[n] / (pi ** 2 * g * D[n] ** 5)
        # Darcy-Weisbach equations each pipe
        for n, pipe in enumerate(network.pipes):
            eq.append(
                (
                    (
                        H[node_uid_to_index[pipe.upstream_node.uid]]
                        - H[node_uid_to_index[pipe.downstream_node.uid]]
                    )
                    / L[n]
                    - gamma[n]
                    / L[n]
                    * labda[n]
                    * (
                        ca.sqrt(Q_nom[n] ** 2 + smpa_a ** 2)
                        + b[n]
                        + c[n] / ca.sqrt(Q_nom[n] ** 2 + smpa_d ** 2)
                    )
                    * Q[n]
                )
                * (1 - theta)
                + (
                    (
                        H[node_uid_to_index[pipe.upstream_node.uid]]
                        - H[node_uid_to_index[pipe.downstream_node.uid]]
                    )
                    / L[n]
                    - gamma[n]
                    / L[n]
                    * labda[n]
                    * (
                        ca.sqrt(Q[n] ** 2 + smpa_a ** 2)
                        + b[n]
                        + c[n] / ca.sqrt(Q[n] ** 2 + smpa_d ** 2)
                    )
                    * Q[n]
                )
                * theta
            )

            # Mass conservation equations each node
            if pipe.upstream_node.uid in mc:
                mc[pipe.upstream_node.uid] -= Q[n]
            if pipe.downstream_node.uid in mc:
                mc[pipe.downstream_node.uid] += Q[n]

    # Root finder
    START = ca.vertcat(*start)
    EQ = ca.vertcat(*eq)
    MC = ca.vertcat(*mc.values())
    f = ca.Function("f", [ca.vertcat(Q, H)], [ca.vertcat(START, EQ, MC)])
    F = ca.rootfinder("F", "newton", f)
    sol = F(sol)
    print(sol)

np.savetxt(
    "HomotopyH_ref.csv",
    sol[(len(network.pipes)) : ((len(network.pipes)) + (len(network.nodes)))],
)
