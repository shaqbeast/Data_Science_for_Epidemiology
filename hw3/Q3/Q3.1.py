import sis_model
import networkx as nx 
import matplotlib.pyplot as plt
import numpy as np 

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
        
beta = 0.005
num_nodes = G.number_of_nodes()
i_frac = 4 / num_nodes
time = []
for i in range(0, 101):
    time.append(i)

# these arrays contain the numbers of s and i 100 time steps for 1 run...100 runs
s_runs = []
i_runs = []
for _ in range(100):
    si_dict = sis_model.simulate_t_steps_SI(G, i_frac, beta, 100, None, False)
    # np arrays for the number of s and i
    s = si_dict["S"]
    i = si_dict["I"]
    s_runs.append(s)
    i_runs.append(i)

s_avg = []
# i will represent time steps
for t in range(101):
    # will contain the values of s at time step i for all runs
    s_t_array = []
    for run in s_runs:
        s_at_t = run[t] # get the s value at time step i for a specific run 
        s_t_array.append(s_at_t) # add that value to an array with all s values for time step i 
    # once we have all the values of s at i for all runs, we calculate mean
    avg_s_at_t = np.mean(s_t_array) 
    s_avg.append(avg_s_at_t)
    
# do it for i
# write assert statements at the beginning of loops or end of loops to see 
# if program is working how it's supposed to
i_avg = []
for t in range(101):
    i_t_array = []
    for run in i_runs:
        i_at_t = run[t] 
        i_t_array.append(i_at_t) 
    avg_i_at_t = np.mean(i_t_array) 
    i_avg.append(avg_i_at_t)


plt.plot(time, s_avg, label='Susceptible')
plt.plot(time, i_avg, label='Infected')
plt.xlabel('Time Steps')
plt.ylabel('Number of People')
plt.legend()
plt.title('SI Model (β = ' + str(beta) + ')')
plt.show()

# get the daily infection values
daily_infections = [i_avg[0]]

for i in range(1, 101):
    daily_infected = i_avg[i] - i_avg[i - 1]
    daily_infections.append(daily_infected)

print(np.argmax(time, daily_infections))

plt.plot(daily_infections)
plt.xlabel('Time Steps')
plt.ylabel('Number of People')
plt.title('Daily Infected')
plt.show()



