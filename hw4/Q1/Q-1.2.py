from modules import sis_model
from modules import utils
import networkx as nx
import numpy as np

k = 100
G_ba = nx.barabasi_albert_graph(1000,2)
rng = np.random.RandomState(42)

nodes = utils.sample_nodes(G=G_ba, num_nodes=k, rng=rng)
print("RANDOM POLICY: ")
print(list(nodes))

print()

print("NEIGHBOR POLICY: ")
neighbors = utils.sample_neighbors(G=G_ba, num_nodes=k, rng=rng)
print(neighbors)