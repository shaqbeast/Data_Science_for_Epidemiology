import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# beta - probability of S individual getting infected given they interact with I
# gamma - probability of I individual recovering 
# S - fraction of population that is S
# I - fraction of population that is I
# R - fraction of population that is R
# dX_dt - a list of tuples that contains 
def calculate_SIR_Model(listOfSIR, time, beta, gamma):
    S = listOfSIR[0]
    I = listOfSIR[1]
    R = listOfSIR[2]
    dS_dt = (-beta * S * I)
    dI_dt = (beta * S * I) - (gamma * I)
    dR_dt = gamma * I 
     
    dX_dt = [dS_dt, dI_dt, dR_dt]
    
    return dX_dt

# take S0, I0, R0 and run them through "calculate_SIR_Model"
# find the integral of each of the dy/dt values
# plot those values on a graph 
# run the newly integrated values back into "calculate_SIR_Model"
# repeat - take new SIR and run it back through "calculate_SIR_Model"
def simulate_SIR_Model(beta, gamma, S0, I0, R0, max_time, time_steps):
    initial_List_SIR = [S0, I0, R0]
    # np.linspace creates an array of time points from 0 - max_time with increments of time_steps
    time = np.linspace(0, max_time, time_steps)
    # odeint calculates basic differential equations 
    y = odeint(calculate_SIR_Model, y0 = initial_List_SIR, t = time, args = (beta, gamma))
    return y, time

    
beta1 = 0.05
gamma1 = 0.1
S0 = 0.95
I0 = 0.05
R0 = 0
max_time = 200
time_steps = 1000

results1, t1 = simulate_SIR_Model(beta1, gamma1, S0, I0, R0, max_time, time_steps)
S1 = results1.T[0]
I1 = results1.T[1]
R1 = results1.T[2]


plt.plot(t1, S1, label='Susceptible')
plt.plot(t1, I1, label='Infected')
plt.plot(t1, R1, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Fraction of Population')
plt.legend()
plt.title('SIR Model (β = ' + str(beta1) + ' γ = ' + str(gamma1) + ')')
plt.show()

    
        
    
    
    
    