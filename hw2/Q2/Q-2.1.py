import numpy as np 
import networkx as nx 
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh

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
                   
# main flow 
ro_array = []
eigen_array = []

for ro in range(1, 10):
    Graph_ro = create_aggregate(ro)
    ro_array.append(ro)
    adjacency_matrix = nx.adjacency_matrix(Graph_ro, dtype=float)
    eigenvalues, _ = eigsh(adjacency_matrix, k=1, which='LM')
    eigen_array.append(eigenvalues[0])

plt.plot(ro_array, eigen_array)
plt.xlabel('Ro Values')
plt.ylabel('Eigenvalues')
plt.title('Eigenvalues vs Ro')
plt.show()
    
                
                
            

            
            