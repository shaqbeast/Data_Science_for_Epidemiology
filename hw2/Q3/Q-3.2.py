import numpy as np 
import networkx as nx 
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh
import EoN 

caylee_tree = nx.Graph()

with open ('/Users/shaqbeast/CSE 8803 EPI/hw2/Q3/cayley.txt', 'r') as file:
    for line in file:
        array_Nodes = line.strip().split()
        node1 = array_Nodes[0]
        node2 = array_Nodes[1]
        
        caylee_tree.add_node(node1)
        caylee_tree.add_node(node2)
        caylee_tree.add_edge(node1, node2)        

beta = 0.2
gamma = 0 # gamma is 0 because we don't have recovery in SI model
max_time = 10
initial_infexted = ['0']
infected_nodes = [] # keeps track of all 100 simulations of infections

# run fast_SIR with a gamma of 0 
for i in range(100): 
    full_data = EoN.fast_SIR(caylee_tree, beta, gamma, initial_infecteds=initial_infexted, return_full_data=True)
    infected_time10 = full_data.get_statuses(None, 10)
    list_of_infected = list()
    for key, value in infected_time10.items():
        if value == 'I':
            list_of_infected.append(key)
            
    infected_nodes.append(list_of_infected)        
    # infected_nodes.append(' '.join(list_of_infected))

# print(infected_nodes)

# create a subgraph for each of the infected nodes out of 100 simulations
# run the center function on each of these graphs
# see where the center is
# if the center is 0, increase counter by 1
counter = 0
for sim_inf_nodes in infected_nodes:
    subgraph = caylee_tree.subgraph(sim_inf_nodes)
    center = nx.center(subgraph)
    # print(center)
    for i in center:
        if i == '0':
            counter += 1

accuracy = counter / 100
print("Accuracy of Algo: " + str(accuracy))