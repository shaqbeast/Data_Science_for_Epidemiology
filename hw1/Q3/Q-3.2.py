import numpy as np 
import csv
import networkx as nx 
from datetime import datetime
import matplotlib.pyplot as plt

# network array contains a list of tuples that show each edge that exists in the network
G = nx.DiGraph()
with open ('/Users/shaqbeast/CSE 8803 EPI/hw1/Q3/network.txt', mode='r') as file:
     for line in file:
        array_Nodes = line.strip().split()
        node1 = array_Nodes[0]
        node2 = array_Nodes[1]
        
        G.add_node(node1)
        G.add_node(node2)
        G.add_edge(node1, node2)

# ratings array contains a list of tuples that contain the date and movieIDs rated for each node/userID
# row[0] = userID
# row[1] = movieID

ratings = dict()
with open ('/Users/shaqbeast/CSE 8803 EPI/hw1/Q3/Ratings.timed.csv', mode='r') as file:
    csv_reader = csv.reader(file)
    header = next(csv_reader)
    for row in csv_reader:
        v_ID = int(row[0])
        movie_ID = int(row[1])
        date_string = row[3]
        date_object = datetime.strptime(date_string, "%Y/%m/%d %H:%M")
        key = v_ID
        value = (movie_ID, date_object)
        if key in ratings:
            ratings[key].append(value)
        else:
            ratings[key] = [value]

def calculate_pv(network, ratings):
    Av2u = 0
    Av = 0
    results = []
    
    # go through all nodes in the network
    for node in network.nodes():
        v_actions = ratings.get(int(node))
        if v_actions is None:
            Av = 0
        else:
            Av = len(v_actions)
        neighbors = list(network.neighbors(node))
        for neighbor_node in neighbors:
            u_actions = ratings.get(int(neighbor_node))
            Av2u = 0
            probability = 0
            if u_actions is not None and v_actions is not None:
                for v_tuple in v_actions:
                    v_movieID, v_date_time = v_tuple
                    for u_tuple in u_actions:
                        u_movieID, u_date_time = u_tuple
                        if ((v_movieID == u_movieID) and (v_date_time <= u_date_time)):
                            Av2u += 1
            else:
                Av2u = 0
            
            if Av == 0:
                probability = 0.0
            else: 
                probability = Av2u / Av
            tuple = (int(node), int(neighbor_node), probability)
            results.append(tuple)
            # print(tuple)
    
    return results
                        
                    
data = calculate_pv(G, ratings)
data_dict = dict()

for tuple in data:
    v, u, Pvu = tuple
    if v in data_dict:
        data_dict[v].append(Pvu)
    else:
        data_dict[v] = [Pvu]

# sum = 0
'''
for key in data_dict:
    # sum = 0
    for value in data_dict[key]:
        # sum += value
        plt.hist(value, bins=100, log=True)
'''
    
'''
# find the puv for each node
# add them up
# the final value gets added to the histogram
index = 0
sum = 0
for tuple in data:
    v, u, Pvu = tuple
    # if the index is not at the first edge
    if index > 0:
        previous_v, previous_u, previous_Pvu = data[index - 1]
        if v == previous_v:
            sum += Pvu
        else:
            print(sum)
            plt.hist(sum)
            # print(sum)
            sum = Pvu
    # 1st edge
    else:
        sum += Pvu

    index += 1
'''
#plot histogram
plt.hist(data_dict.values(), bins=5, log=True)
plt.title("Histogram for PU")
plt.xlabel("Probability")
plt.ylabel("Frequency(log scale)")
plt.show()

# output results into another csv file
header = ["v", "u", "probability"]
with open('Q3.2_output.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)
    
    writer.writerows(data)
    
    
                            
                            
                    
                
        
    
    