from signal_processing import SignalProcessingGraph
import networkx as nx
import numpy as np


class RingGraph(SignalProcessingGraph):
    def __init__(self, N):
        super().__init__()
        for n in range(N):
            self.add_node(n, value=0, x=100 * n, y=0)
        self.nodes[0]['value'] = 1
        for n in range(N - 1):
            self.add_edge(n, n + 1, weight=1)
        self.add_edge(N - 1, 0, weight=1)


class WattsStrogatzGraph(SignalProcessingGraph):
    def __init__(self, N, K, beta):
        super().__init__()
        nw = nx.watts_strogatz_graph(N, K, beta)
        for n in nw.nodes:
            nw.nodes[n]['x'] = 900 * np.sin(2 * np.pi * n / N)
            nw.nodes[n]['y'] = 900 * np.cos(2 * np.pi * n / N)

        for n in nw.nodes:
            nw.nodes[n]['value'] = 40 * (np.sin(1.5 * np.pi * n / N)) ** 2 - 4
            # nw.nodes[n]['size'] = 0
        for n in nw.nodes:
            self.add_node(n)
            for attr in nw.nodes[n]:
                self.nodes[n][attr] = nw.nodes[n][attr]
            self.nodes[n]['title'] = str(self.nodes[n]['value'])
        for e in nw.edges:
            self.add_edge(e[1], e[0])
            self.add_edge(*e)
        for e in self.edges:
            self.edges[e]['hidden'] = False
            self.edges[e]['weight'] = 1
