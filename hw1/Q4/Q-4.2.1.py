import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# listOfXY - contains x (population count that's human) and y (population count that's zombie)
# alpha * x = people added to the human population (births, migrations, etc.)
# -beta * x * y = people eaten by zombies
# gamma * x * y = zombies added to zombie population
# -delta * y = zombies that die out
# dX_dt - change of human population w/ respect to time
# dY_dt - change of zombie population w/ respect to time
def calculate_LVM_Model(listOfXY, time, alpha, beta, gamma, delta):
    x = listOfXY[0]
    y = listOfXY[1]
    dX_dt = (alpha * x) - (beta * x * y)
    dY_dt = (gamma * x * y) - (delta * y)
     
    return_array = [dX_dt, dY_dt]
    
    return return_array

# take S0, I0, R0 and run them through "calculate_SIR_Model"
# find the integral of each of the dy/dt values
# plot those values on a graph 
# run the newly integrated values back into "calculate_SIR_Model"
# repeat - take new SIR and run it back through "calculate_SIR_Model"
def simulate_LVM_Model(alpha, beta, gamma, delta, xt, yt, max_time, time_steps):
    initial_List_XY = [xt, yt]
    # np.linspace creates an array of time points from 0 - max_time with increments of time_steps
    time = np.linspace(0, max_time, time_steps)
    # odeint calculates basic differential equations 
    y = odeint(calculate_LVM_Model, y0 = initial_List_XY, t = time, args = (alpha, beta, gamma, delta))
    return y, time

xy_list_tuples = [(5, 2), (0, 0), (1, 1)]
alpha = 1
beta = 1
gamma = 1
delta = 1
max_time = 100
time_steps = 1000

for tuple in xy_list_tuples:
    x0 = tuple[0]
    y0 = tuple[1]

    results1, t1 = simulate_LVM_Model(alpha, beta, gamma, delta, x0, y0, max_time, time_steps)
    x1 = results1.T[0]
    y1 = results1.T[1]

    plt.plot(t1, x1, label='Humans')
    plt.plot(t1, y1, label='Zombies')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.legend()
    plt.title('LVM Model (α = 1, β = 1, γ = 1, δ = 1,' + ' xt = ' + str(x0) 
              + ', yt = ' + str(y0) + ')')
    plt.show()
