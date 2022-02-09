from signal_processing import SignalProcessingGraph


class Halfer(SignalProcessingGraph):
    def __init__(self, N):
        super().__init__()
        for i in range(N + 1):
            self.add_node(i, value=0, x=100 * i, y=0)
        self.nodes[N]['value'] = 1
        self.nodes[0]['x'] = 100
        self.nodes[0]['y'] = 100
        self.add_edge(0, 1, weight=1)
        for i in range(1, N):
            self.add_edge(i, i + 1, weight=1 / 2)
            self.add_edge(i, 0, weight=1 / 2)
        self.add_edge(N, 0, weight=1)


class Triangle(SignalProcessingGraph):
    def __init__(self, a=1):
        super().__init__()
        self.add_node(1, value=0, x=0, y=0)
        self.add_node(2, value=1, x=100, y=100)
        self.add_node(3, value=0, x=-100, y=100)
        self.add_edge(1, 2, weight=1)
        self.add_edge(2, 3, weight=1)
        self.add_edge(3, 1, weight=a)
