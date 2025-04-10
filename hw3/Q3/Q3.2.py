import numpy as np 
import sis_model
import networkx as nx 
import matplotlib.pyplot as plt
import random 

def simulate_strategy(name, G):
    n = G.number_of_nodes()
    i_frac = 4 / n
    beta = 0.005

    time = []
    for i in range(101):
        time.append(i)

    s_runs = []
    i_runs = []
    for _ in range(20):
        si_dict = sis_model.simulate_t_steps_SI(G, i_frac, beta, 100, None, False)
        s = si_dict['S']
        i = si_dict['I']
        
        s_runs.append(s)
        i_runs.append(i)
    
    # calculate avg for s
    s_avg = []
    for t in range(101):
        s_t_array = []
        for run in s_runs:
            s_at_t = run[t]  
            s_t_array.append(s_at_t) 
        avg_s_at_t = np.mean(s_t_array) 
        s_avg.append(avg_s_at_t)
    
    # calculate avg for i
    i_avg = []
    for t in range(101):
        i_t_array = []
        for run in i_runs:
            i_at_t = run[t] 
            i_t_array.append(i_at_t) 
        avg_i_at_t = np.mean(i_t_array) 
        i_avg.append(avg_i_at_t)

    plt.plot(time, s_avg, label = "S")
    plt.plot(time, i_avg, label = "I")
    plt.xlabel("Time Step")
    plt.ylabel("Number of People")
    plt.title(name + " Strategy")
    plt.legend()
    plt.show()
    
    # get the daily infection values
    daily_infections = [i_avg[0]]

    for i in range(1, 101):
        daily_infected = i_avg[i] - i_avg[i - 1]
        daily_infections.append(daily_infected)

    # print(np.argmax(daily_infections))

    plt.plot(time, daily_infections)
    plt.xlabel('Time Steps')
    plt.ylabel('Number of People')
    plt.title('Daily Infected - ' + name + " Strategy")
    plt.show()

G = nx.Graph()

'''CHANGE FILE PATH HERE'''
with open('/Users/shaqbeast/CSE 8803 EPI/hw3/hw3/Q3/data/facebook_combined.txt', 'r') as file:
    for line in file:
        array_nodes = line.strip().split()
        node1 = array_nodes[0]
        node2 = array_nodes[1]
        
        G.add_node(node1)
        G.add_node(node2)
        G.add_edge(node1, node2)

data = np.load('/Users/shaqbeast/CSE 8803 EPI/hw3/hw3/Q3/data/rand_nodes.npy')
k = 100

'''Random Strategy'''
k_nodes = data[:k]
G_random = G.subgraph(k_nodes)
# nx.draw(G_random, with_labels=False, node_color='lightblue', edge_color='gray', node_size=500)
# plt.show()

simulate_strategy("Random", G_random)


'''Friend Strategy'''
friends_nodes = [] 
for node in k_nodes:
    neighbors = list(G.neighbors(node))
    if len(neighbors) > 0:
        friend = random.choice(neighbors)
        friends_nodes.append(friend)

G_friends = G.subgraph(friends_nodes)
simulate_strategy("Friends", G_friends)

'''Eigenvector Centrality'''
central = nx.eigenvector_centrality_numpy(G)
central_k_nodes = sorted(central.items(), key=lambda item: item[1], reverse=True)
ev_central_nodes = []

for i in range(k):
    node, measure = central_k_nodes[i]
    ev_central_nodes.append(node)
    
G_ev = G.subgraph(ev_central_nodes)
simulate_strategy("Eigenvector Centrality", G_ev)
    


