import numpy as np 
import networkx as nx 
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh
import EoN 

G1 = nx.Graph()
G2 = nx.Graph()
G3 = nx.Graph()
G4 = nx.Graph()
G5 = nx.Graph()
G6 = nx.Graph()
G7 = nx.Graph()
G8 = nx.Graph()
G9 = nx.Graph()

def open_file(network_txt, G):
    with open(network_txt, 'r') as file:
        for _ in range(4):  # Skip the first 4 lines
            next(file)
        for line in file:
            array_Nodes = line.strip().split()
            node1 = array_Nodes[0]
            node2 = array_Nodes[1]
            
            G.add_node(node1)
            G.add_node(node2)
            G.add_edge(node1, node2)
        
open_file('/Users/shaqbeast/CSE 8803 EPI/hw2/Q2/networks/network1.txt', G1)
open_file('/Users/shaqbeast/CSE 8803 EPI/hw2/Q2/networks/network2.txt', G2)
open_file('/Users/shaqbeast/CSE 8803 EPI/hw2/Q2/networks/network3.txt', G3)
open_file('/Users/shaqbeast/CSE 8803 EPI/hw2/Q2/networks/network4.txt', G4)
open_file('/Users/shaqbeast/CSE 8803 EPI/hw2/Q2/networks/network5.txt', G5)
open_file('/Users/shaqbeast/CSE 8803 EPI/hw2/Q2/networks/network6.txt', G6)
open_file('/Users/shaqbeast/CSE 8803 EPI/hw2/Q2/networks/network7.txt', G7)
open_file('/Users/shaqbeast/CSE 8803 EPI/hw2/Q2/networks/network8.txt', G8)
open_file('/Users/shaqbeast/CSE 8803 EPI/hw2/Q2/networks/network9.txt', G9)

array_graphs = [G1, G2, G3, G4, G5, G6, G7, G8, G9]

def create_aggregate(ro):
    G_aggregate = nx.Graph()
    dict_Aggregate = dict()
    for G in array_graphs:
        for edge_uv in G.edges():
            if edge_uv in dict_Aggregate:
                dict_Aggregate[edge_uv] += 1
            else:
                dict_Aggregate[edge_uv] = 1
    
    for edge_uv in dict_Aggregate.keys():
        u, v = edge_uv
        if dict_Aggregate[edge_uv] >= ro:
            G_aggregate.add_node(u)
            G_aggregate.add_node(v)
            G_aggregate.add_edge(u, v)
    
    return G_aggregate

agg_Graph = create_aggregate(1)
'''CHANGE BETA VALUE HERE'''
beta = 0.001 
'''CHANGE BETA VALUE HERE'''
gamma = 0.08
max_time = 10
simulations = 50

# running the simulations
initial_infected = ['1']
results_avg = []
report_times = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
for index in range(simulations):
    # tuple will be t, S, I
    t, S, I = EoN.fast_SIS(agg_Graph, beta, gamma, tmax=max_time, initial_infecteds=initial_infected)
    # subsample gives us proper report times that allow us to get fixed # of times 
    results = EoN.subsample(report_times=report_times, times=t, status1=S, status2=I)
    results_avg.append(results)

avg_I = []
time_step = 0
while time_step < max_time:
    temp_I = []
    for simulation in results_avg:
        # for each simulation, get the t, S, I for the specified time poi       
        sim_S, sim_I = simulation
        temp_I.append(sim_I[time_step])
    
    sum_I = 0.0
    for value in temp_I:
        sum_I += value
    avg_value_It = sum_I / simulations
    
    avg_I.append(avg_value_It)
    
    time_step += 1


    
# plt.plot(t, S, label='Susceptible')
plt.plot(report_times, avg_I, label='Infected')
plt.xlabel('Time')
plt.ylabel('Fraction of Population')
plt.legend()
plt.title('SIS Model (β = ' + str(beta) + ' γ = ' + str(gamma) + ')')
plt.show()

    