from modules import sis_model
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt 

G_ba = nx.barabasi_albert_graph(1000, 2)
num_simulations = 200
infected = []
for i in range(1000):
    infected.append(i)

# get 200 simulations
# each simulation has 500 arrays at one index
simulations = []
for runs in range(num_simulations):
    states = sis_model.simulate_SIS(G_ba, infected=infected)
    simulations.append(states)

# get average at each time step
avg_values = []
for t in range(501):
    states_at_t = []
    avg_infected = 0
    infected_count = 0
    
    for simulation in simulations:
        states_at_t.append(simulation[t])

    for state in states_at_t:
        for node in state:
            if node == 1:
                infected_count += 1
    
    avg_infected = infected_count / num_simulations
    avg_values.append(avg_infected)
    
    print(str(t) + ": " + str(avg_infected) + ", ", end="")

time = []
for t in range(501):
    time.append(t)
plt.plot(time, avg_values)
plt.xlabel("Time Steps")
plt.ylabel("Number Infected")
plt.title("Q1.1")

plt.show()
                
        
        
        
    

