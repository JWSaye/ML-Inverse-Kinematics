import numpy as np
from sympy import symbols, pi, sin, cos, simplify
from sympy.matrices import Matrix
import sympy as sp
import pandas as pd

def precompute_dh_matrices():
    theta1, theta2, theta3, theta4, theta5, theta6 = symbols('theta1:7')
    alpha0, alpha1, alpha2, alpha3, alpha4, alpha5 = symbols('alpha0:6')
    d1, d2, d3, d4, d5, d6 = symbols('d1:7')    
    a0, a1, a2, a3, a4, a5 = symbols('a0:6')

    kuka_s = {alpha0:   -pi/2,  d1:  0.675,      a0:   0.260,
              alpha1:   0,      d2:     0,       a1:   0.68,    
              alpha2:   pi/2,   d3:     0,       a2:   0,       theta2: (theta2 - pi/2),
              alpha3:  -pi/2,   d4:  -0.67,      a3:   0,
              alpha4:   pi/2,   d5:     0,       a4:   0,
              alpha5:   pi,     d6:     -0.158,  a5:   0, }

    def build_mod_dh_matrix(s, theta, alpha, d, a):
        Ta_b = Matrix([
            [cos(theta), -cos(alpha)*sin(theta),  sin(alpha)*sin(theta), a*cos(theta)],
            [sin(theta),  cos(alpha)*cos(theta), -sin(alpha)*cos(theta), a*sin(theta)],
            [0,           sin(alpha),             cos(alpha),            d           ],
            [0,           0,                      0,                     1           ]
        ])
        return Ta_b.subs(s)

    T0_1 = build_mod_dh_matrix(s=kuka_s, theta=theta1, alpha=alpha0, d=d1, a=a0)
    T1_2 = build_mod_dh_matrix(s=kuka_s, theta=theta2, alpha=alpha1, d=d2, a=a1)
    T2_3 = build_mod_dh_matrix(s=kuka_s, theta=theta3, alpha=alpha2, d=d3, a=a2)
    T3_4 = build_mod_dh_matrix(s=kuka_s, theta=theta4, alpha=alpha3, d=d4, a=a3)
    T4_5 = build_mod_dh_matrix(s=kuka_s, theta=theta5, alpha=alpha4, d=d5, a=a4)
    T5_6 = build_mod_dh_matrix(s=kuka_s, theta=theta6, alpha=alpha5, d=d6, a=a5)

    T_total = simplify(T0_1 * T1_2 * T2_3 * T3_4 * T4_5 * T5_6)

    return T_total

# Precompute the symbolic expression
T_symbolic = precompute_dh_matrices()

# Convert symbolic expression to a function that takes numpy arrays as input
T_numeric = sp.lambdify(['theta1', 'theta2', 'theta3', 'theta4', 'theta5', 'theta6'], T_symbolic, 'numpy')

def calculate_position_vectorized(thetas):
    # thetas should be a 2D array with shape (n_points, 6)
    results = np.array([T_numeric(*theta) for theta in thetas])
    return results

# Usage in main script:
number_points = 1000
thetas = np.column_stack([
    np.random.uniform(-1.57, 1.57, number_points),
    np.random.uniform(-1.05, 1.05, number_points),
    np.random.uniform(-1.05, 1.05, number_points),
    np.random.uniform(-0.78, 0.78, number_points),
    np.random.uniform(-0.78, 0.78, number_points),
    np.random.uniform(-0.78, 0.78, number_points)
])

results = calculate_position_vectorized(thetas)

# Extract positions and orientations
positions = results[:, :3, 3]
n = results[:, :3, 0]
o = results[:, :3, 1]
a = results[:, :3, 2]

# Create DataFrame
df = pd.DataFrame(np.hstack((positions, n, o, a, np.degrees(thetas))))
df = df.iloc[1:]
df.to_csv(r'.\datasets_6DOF\d6DOF.csv')