import read_in_data
from wilson import WilsonGraph


class FacebookGraph(WilsonGraph):
    def __init__(self):
        super().__init__()
        nw = read_in_data.G_fb
        for n in nw.nodes:
            nw.nodes[n]['x'] = 600 * read_in_data.spring_pos[n][0]
            nw.nodes[n]['y'] = 600 * read_in_data.spring_pos[n][1]

        for n in nw.nodes:
            nw.nodes[n]['value'] = read_in_data.values[n]-8
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
