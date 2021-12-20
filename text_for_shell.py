import wilson,square_graph
g = square_graph.SquareWilson(60, standardweights=False)
q_list, q_prime_list, graph_list = wilson.multiresolution(g, steps=3)

for g in graph_list:
    g.show("g.html", color_roots=False)

for i,g in enumerate(graph_list):
    g.create_pdf(f'graph_{i}.pdf')
