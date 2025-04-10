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

agg_Graph = create_aggregate(1)
beta = 0.01 
gamma = 0.16
adjacency_matrix = nx.adjacency_matrix(agg_Graph, dtype=float)
eigenvalues, _ = eigsh(adjacency_matrix, k=1, which='LM')
eigenvalue = eigenvalues[0]

s = eigenvalue * (beta / gamma)
s_values = [s]
num_of_nodes_removed = 0
num_removed_arr = [0]


while s >= 1:
    degrees = dict(agg_Graph.degree())
    # find what the max degree is in the graph
    largest_degree = 0
    for node, degree in degrees.items():
        if degree > largest_degree:
            largest_degree = degree

    # add all the nodes with the max degree to a list
    largest_degree_nodes = []
    for node, degree in degrees.items():
        if degree == largest_degree:
            largest_degree_nodes.append(int(node))

    node_to_remove = str(max(largest_degree_nodes))
    agg_Graph.remove_node(node_to_remove)
    num_of_nodes_removed += 1
    num_removed_arr.append(num_of_nodes_removed)

    adjacency_matrix = nx.adjacency_matrix(agg_Graph, dtype=float)
    eigenvalues, _ = eigsh(adjacency_matrix, k=1, which='LM')
    eigenvalue = eigenvalues[0]

    s = eigenvalue * (beta / gamma)
    s_values.append(s)

removed = len(num_removed_arr) - 1
final_s = s_values[40]
print("Number of Nodes Removed: " + str(removed))
print("Final s value: " + str(final_s))
plt.plot(num_removed_arr, s_values)
plt.xlabel('Number of Nodes Removed')
plt.ylabel('Values of s')
plt.title('Values of s vs Number of Nodes Removed (β = ' + str(beta) + ' γ = ' + str(gamma) + ')')
plt.show()
