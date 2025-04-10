import numpy as np 
from scipy.optimize import minimize 
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import csv 
# make sure robserved and Rt are pretty similar when they're plotted

cases = [] # I observed
deaths = [] # R observed
Rt_values = []
R_observed_values = []

# extract data from COVID file
with open ('/Users/shaqbeast/CSE 8803 EPI/hw1/Q1/COVID19_GA.csv', mode='r') as file:
    csv_reader = csv.reader(file)
    # skip the first row
    header = next(csv_reader)
    for row in csv_reader:
        cases.append(float(row[1]))
        deaths.append(float(row[2]))

# initilizaing S0, I0, R0
I0 = cases[0]
R0 = deaths[0]
S0 = 1 - R0 - I0
max_time = len(deaths)



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

def simulate_SIR_Model(beta, gamma, S0, I0, R0, max_time):
    initial_List_SIR = [S0, I0, R0]
    # np.linspace creates an array of time points from 0 - max_time with increments of time_steps
    time = np.linspace(0, max_time, len(deaths))
    # odeint calculates basic differential equations 
    y = odeint(calculate_SIR_Model, y0 = initial_List_SIR, t = time, args = (gamma, beta))
    return y, time

def objective_function(listOfPararameters, R_observed):
    # get the solution
    # compute the objective function 
    # return 
    
    # unpacking beta and gamma
    gamma = listOfPararameters[0]
    beta = listOfPararameters[1]
        
    # solve the SIR model for all time periods for a given value of gamma and beta
    y, time = simulate_SIR_Model(beta, gamma, S0, I0, R0, max_time)
    Rt = y.T[2]
    
    # compute objective function 
    sum = 0
    for t in range(1, len(R_observed)): 
        difference = (Rt[t] - R_observed[t]) ** 2
        sum += difference

    return sum 

# optimizing gamma and beta
listOfParm = [1e-5, 1e-5]
bounds = [(0.0, 1.0), (0.0, 1.0)] 
result = minimize(objective_function, listOfParm, args=(deaths), tol=1e-10)

# unpacking to get optimal parameter values
optimal_beta, optimal_gamma = result.x

solution, t = simulate_SIR_Model(optimal_gamma, optimal_beta, S0, I0, R0, max_time)

S = solution.T[0]
I = solution.T[1]
R = solution.T[2]

plt.plot(t, S, label='Susceptible (S)')
plt.plot(t, I, label='Infected (I)')
plt.plot(t, R, label='Recovered (R)')
plt.xlabel('Time')
plt.ylabel('Fraction of Population')
plt.title(f'SIR Model with beta={optimal_beta}, gamma={optimal_gamma}')
plt.legend()
plt.show()

plt.plot(deaths, label='Observed R(t)')
plt.plot(R, label='Modeled R(t)')
plt.xlabel('Time')
plt.ylabel('Fraction of Population')
plt.title('Observed vs Modeled Recovered (R)')
plt.legend()
plt.show()

print("Optimal Beta: " + str(optimal_beta))
print("Optimal Gamma: " + str(optimal_gamma))





    
    