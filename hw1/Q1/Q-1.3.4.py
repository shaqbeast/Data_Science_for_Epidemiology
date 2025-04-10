import numpy as np 
import networkx as nx 
from scipy.optimize import minimize 
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from random import random
import csv 
import math
import random
# make sure robserved and Rt are pretty similar when they're plotted

G = nx.Graph()
'''CHANGE VALUES OF N HERE'''
N = 500

# Add however many nodes into the graph
G.add_nodes_from(range(N))

for i in range(N):
    for j in range(i + 1, N):
        if random.random() < 0.5:  # 5% chance to add an edge
            G.add_edge(i, j)

def calculate_SIR_Model(listOfSIR, time, beta, gamma):
    S = listOfSIR[0]
    I = listOfSIR[1]
    R = listOfSIR[2]
    dS_dt = (-beta * S * I)
    dI_dt = (beta * S * I) - (gamma * I)
    dR_dt = gamma * I 
     
    dX_dt = [dS_dt, dI_dt, dR_dt]
    
    return dX_dt

def simulate_SIR_Network(beta, gamma, S0, I0, R0, max_time):
    
    # we're going to create an array for each category in simulation
    # fill it with zeros
    # based off the initial fraction passed in the method, assign a 1 to each of the arrays randomly
    # based on the array index, the person that is THAT category will be GraphNode = indexOfS,I,R + 1
    N = G.number_of_nodes()
    S = np.zeros(N) 
    I = np.zeros(N)
    R = np.zeros(N)
    t = 0
    results = []
    results.append((t, S0, I0, R0))

    
    # Randomly populate 1s in S, I, R arrays based off decimal
    # k is the number of ones to assign 
    k_S= int(N * S0)
    k_I = int(N * I0)
    k_R = int(N * R0)
    
    # replace=False makes sure no index is selected twice
    indices_S = np.random.choice(N, k_S, replace=False)
    indices_I = np.random.choice(N, k_I, replace=False)
    indices_R = np.random.choice(N, k_R, replace=False)
    
    # replace 1s at specified indices 
    S[indices_S] = 1
    I[indices_I] = 1
    R[indices_R] = 1
    
    # find the places where a node is infected
    # if the surrounding nodes are S, find the probability that node becomes an I with beta variable
    # if the random prob is less than or equal to beta, that S node becomes I
    # again, find the places where the node is infected
    # if the random prob is less than or equal to gamma, that node becomes R
    # do this for each time step
    # do this for each node in G
    
    time_step = 1
    # for a period of time
    while t <= max_time: 
        # go through all the nodes in the graph
        current_I_Nodes = []
        for node in G.nodes():
            if I[int(node) - 1] == 1:
                current_I_Nodes.append(node)
            
        for node in current_I_Nodes:
            # find the infected nodes
            if I[int(node) - 1] == 1:
                # get the neighbors of the infected nodes
                neighbors = list(G.neighbors(node))
                # get one neighbor in the neighbors array
                random_index = random.randrange(0, len(neighbors))
                # go through all the neighbors if that infected node 
                # for neighbor_node in neighbors:
                    # see if that neighbor is in S
                if S[int(neighbors[random_index]) - 1] == 1:
                    random_num_beta = random.random()
                    if random_num_beta <= beta:
                        S[int(neighbors[random_index]) - 1] = 0
                        I[int(neighbors[random_index]) - 1] = 1
                # see if this I node turns into R
                random_num_gamma = np.random.random()
                if random_num_gamma <= gamma: 
                    I[int(node) - 1] = 0
                    R[int(node) - 1] = 1
        
        # find number of S, I, R in the arrays
        # get the fraction and return
        
        s_count = 0
        i_count = 0
        r_count = 0
        
        for num in S:
            if num == 1:
                s_count += 1

        for num in I:
            if num == 1:
                i_count += 1
                
        for num in R:
            if num == 1:
                r_count += 1 
        s_t = s_count / N
        i_t = i_count / N
        r_t = r_count / N
        
        results.append((t, s_t, i_t, r_t))
        t += time_step
        
    return results

cases = [] # I observed
deaths = [] # R observed
Rt_values = []
R_observed_values = []
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
max_time = 200

# unpacking to get optimal parameter values
optimal_beta = 0.04926084359985644
optimal_gamma = 9.175149064429577e-06

solution = simulate_SIR_Network(optimal_beta, optimal_gamma, S0, I0, R0, max_time)

S = []
I = []
R = []
t = []
time = 0
# get all values from the solution and put them in their respective array 
while time <= max_time:
    current_t, sim_S, sim_I, sim_R = solution[time]
    t.append(current_t)
    S.append(sim_S)
    I.append(sim_I)
    R.append(sim_R)
    time += 1

plt.plot(t, S, label='Susceptible (S)')
plt.plot(t, I, label='Infected (I)')
plt.plot(t, R, label='Recovered (R)')
plt.xlabel('Time')
plt.ylabel('Fraction of Population')
plt.title(f'SIR Model with beta={optimal_beta}, gamma={optimal_gamma}')
plt.legend()
plt.show()