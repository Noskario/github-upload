import networkx as nx
from matplotlib import pyplot as plt

g=nx.erdos_renyi_graph(20,.2,directed=True)
nx.draw_networkx(g)
#plt.show()
plt.savefig('so_example.png', bbox_inches='tight')