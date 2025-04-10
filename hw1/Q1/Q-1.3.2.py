import numpy as np 
import networkx as nx 
import matplotlib.pyplot as plt
import random

G = nx.Graph()

# np.random.seed(60) # every time program is run, we get the same random numbers 

# extract nodes from example.txt
with open('/Users/shaqbeast/CSE 8803 EPI/hw1/Q1/example.txt', 'r') as file:
    for line in file:
        array_Nodes = line.strip().split()
        node1 = array_Nodes[0]
        node2 = array_Nodes[1]
        
        G.add_node(node1)
        G.add_node(node2)
        G.add_edge(node1, node2)

# beta = transmission rate
# gamma = recovery rate 
# S0, I0, R0 = initial fraction of S, I, R individuals (decimlal)
# max_time = the max amount of time we go up to for simulation 
# return(t, S(t), I(t), R(t))
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
        
        
        
        
        
        
        
# Set parameters and initial conditions
beta = 0.05
gamma = 0.1
S0 = 0.95
I0 = 0.05
R0 = 0
max_time = 200
time = 0
num_simulations = 50
time_step = 1

# Simulate the SIR model multiple times and average the results
# when i call simulate, i get 1 simulation that runs from 0 - max_time
# Need to average S(t), I(t), R(t) at each time-step
# Plot each avg time-step
results_avg = []
for index in range(num_simulations):
    results = simulate_SIR_Network(beta, gamma, S0, I0, R0, max_time)
    results_avg.append(results)

avg_S = []
avg_I = []
avg_R = []
avg_t = []
# go through each time point
# find average of each time point in all simulations
while time <= max_time:
    temp_S = []
    temp_I = []
    temp_R = []
    for simulation in results_avg:
        t, sim_S, sim_I, sim_R = simulation[time]
        temp_S.append(sim_S)
        temp_I.append(sim_I)
        temp_R.append(sim_R)
   
    sum_S = 0.0
    sum_I = 0.0
    sum_R = 0.0
    for value in temp_S:
        sum_S += value
    for value in temp_I:
        sum_I += value
    for value in temp_R:
        sum_R += value
    
    avg_value_St = sum_S / num_simulations
    avg_value_It = sum_I / num_simulations
    avg_value_Rt = sum_R / num_simulations
    
    avg_S.append(avg_value_St)
    avg_I.append(avg_value_It)
    avg_R.append(avg_value_Rt)
    avg_t.append(t)
        
    time += time_step
        

plt.plot(avg_t, avg_S, label='Susceptible')
plt.plot(avg_t, avg_I, label='Infected')
plt.plot(avg_t, avg_R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Fraction of Population')
plt.legend()
plt.title('SIR Model (β = ' + str(beta) + ' γ = ' + str(gamma) + ')')
plt.show()

# results_avg = np.mean(results_avg, axis=0)

# Plot the average results



