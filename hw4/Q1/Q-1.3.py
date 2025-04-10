from modules import sis_model
from modules import utils
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import PIL

def find_avg(simulations, max_time):
    '''
    Finds avg values and stores in an array
    '''
    avg_values = []
    
    for t in range(max_time + 1):
        states_at_t = []
        avg_infected = 0
        infected_count = 0
        
        for simulation in simulations:
            states_at_t.append(simulation[t])

        for state in states_at_t:
            for node in state:
                if node == 1:
                    infected_count += 1
        
        avg_infected = infected_count / 200
        
        avg_values.append(avg_infected)
    
    return avg_values


k = 100
num_simulations = 50 # supposed to be 50
max_time = 500
rng = np.random.RandomState(42)
infected = []
for i in range(1000):
    infected.append(i)

# (a) no nodes have been removed
G_ba_a = nx.barabasi_albert_graph(1000, 2)
simulations_a = [] 
for runs in range(num_simulations):
    states = sis_model.simulate_SIS(G_ba_a, infected=infected)
    simulations_a.append(states)

averages_A = find_avg(simulations_a, max_time)

# (b) nodes have been removed according to FRIENDS
G_ba_b = nx.barabasi_albert_graph(1000, 2)
nodes = utils.sample_neighbors(G=G_ba_b, num_nodes=k, rng=rng)
Graph_B = utils.remove_nodes_from_graph(G_ba_b, nodes)
infected_b = []
for node in Graph_B.nodes():
    infected_b.append(node)

simulations_b = []
for runs in range(num_simulations):
    states = sis_model.simulate_SIS(Graph_B, infected=infected_b)
    simulations_b.append(states)

averages_B = find_avg(simulations_b, max_time)

# (c) nodes have been removed according to RANDOM
G_ba_c = nx.barabasi_albert_graph(1000, 2)
nodes = utils.sample_nodes(G=G_ba_c, num_nodes=k, rng=rng)
Graph_C = utils.remove_nodes_from_graph(G_ba_c, nodes)
infected_c = []
for node in Graph_C.nodes():
    infected_c.append(node)

simulations_c = []
for runs in range(num_simulations):
    states = sis_model.simulate_SIS(Graph_C, infected=infected_c)
    simulations_c.append(states)

averages_C = find_avg(simulations_c, max_time)

plt.title("Q1.3 Plot - 3 Immunization Strategies")
plt.xlabel("Time Steps")
plt.ylabel("Number of Infected")
plt.plot(averages_A, label="(a) - no nodes removed", color="red")
plt.plot(averages_B, label="(b) - FRIENDS strategy", color="blue")
plt.plot(averages_C, label="(c) - RANDOM strategy", color="green")
plt.legend()

plt.show()