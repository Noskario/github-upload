import signal_processing,square_graph
g = square_graph.SquareSignalProcessingGraph(60, standardweights=False)
q_list, q_prime_list, graph_list = signal_processing.multiresolution(g, steps=3)

for g in graph_list:
    g.show("g.html", color_roots=False)

for i,g in enumerate(graph_list):
    g.create_picture(f'graph_{i}.pdf')
