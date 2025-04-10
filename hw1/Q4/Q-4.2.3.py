import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def calculate_LVM_Model(listOfXYZ, time, alpha, beta, gamma, delta, phi, rho, epsilon):
    x = listOfXYZ[0]
    y = listOfXYZ[1]
    z = listOfXYZ[2]
    dX_dt = (alpha * x) - (beta * x * y) - (phi * x * z)
    dY_dt = (gamma * x * y) - (delta * y)
    dZ_dt = (rho * x * z) - (epsilon * z)
     
    return_array = [dX_dt, dY_dt, dZ_dt]
    
    return return_array

# take S0, I0, R0 and run them through "calculate_SIR_Model"
# find the integral of each of the dy/dt values
# plot those values on a graph 
# run the newly integrated values back into "calculate_SIR_Model"
# repeat - take new SIR and run it back through "calculate_SIR_Model"
def simulate_LVM_Model(alpha, beta, gamma, delta, phi, rho, epsilon, xt, yt, zt, max_time, time_steps):
    initial_List_XYZ = [xt, yt, zt]
    # np.linspace creates an array of time points from 0 - max_time with increments of time_steps
    time = np.linspace(0, max_time, time_steps)
    # odeint calculates basic differential equations 
    y = odeint(calculate_LVM_Model, y0 = initial_List_XYZ, t = time, args = (alpha, beta, gamma, delta, phi, rho, epsilon))
    return y, time

alpha = 1
beta = 1
gamma = 1
delta = 1.5
phi = 1
rho = 1
epsilon = 2

xyz_list_tuples = [(4, 2, 5), (0, 0, 0), (epsilon / rho, 0, alpha / phi), (delta / gamma, alpha / beta, 0)]

max_time = 100
time_steps = 1000

for tuple in xyz_list_tuples:
    x0 = tuple[0]
    y0 = tuple[1]
    z0 = tuple[2]

    results1, t1 = simulate_LVM_Model(alpha, beta, gamma, delta, phi, rho, epsilon, x0, y0, z0, max_time, time_steps)
    x1 = results1.T[0]
    y1 = results1.T[1]
    z1 = results1.T[2]

    plt.plot(t1, x1, label='Humans')
    plt.plot(t1, y1, label='Zombies - Variant 1')
    plt.plot(t1, z1, label='Zombies - Variant 2')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.legend()
    plt.title('LVM Model (α = 1, β = 1, γ = 1, δ = 1.5, ϕ = 1, ρ = 1, ϵ = 2,' 
              + ' x(0) = ' + str(x0) + ', y(0) = ' + str(y0) + ', z(0) = ' + str(z0) + ')')
    plt.show()