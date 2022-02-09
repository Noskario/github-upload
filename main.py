import cProfile
import copy
import pstats
import random
import time

import matplotlib
import networkx as nx

import example_graphs
import non_reversible_graphs
import numpy as np
import square_graph
import matplotlib.pyplot as plt

import signal_processing
import pickle


def foward(x, a, b, c, d):
    return np.piecewise(x, [(a < x) & (x < b), (b < x) & (x < c), (c < x) & (x < d), x <= a, x == b, x == c, x >= d],
                        [lambda x: 1 / 3 * (x - a) / (b - a), lambda x: 1 / 3 + 1 / 3 * (x - b) / (c - b),
                         lambda x: 2 / 3 + 1 / 3 * (x - c) / (d - c), 0, 1 / 3, 2 / 3, 1])


def inverse(x, a, b, c, d):
    return np.piecewise(x, [(0 <= x) & (x < 1 / 3), (1 / 3 <= x) & (x < 2 / 3), (2 / 3 <= x) & (x <= 1), x < 0,
                            x > 1],
                        [lambda x: a + 3 * x * (b - a), lambda x: b + 3 * (x - 1 / 3) * (c - b),
                         lambda x: c + 3 * (x - 2 / 3) * (d - c), a, d])


def visualize_analysis_operator():
    g_o = square_graph.SquareSignalProcessingGraph(40, standardweights=False)
    g_o.wilson(q=12.345)
    vmin_original = g_o.get_minimum_value()
    vmax_original = g_o.get_maximum_value()
    vmax_original = max(-vmin_original, vmax_original)
    vmin_original = min(-vmax_original, vmin_original)
    print(f'{vmin_original=}')
    print(f'{vmax_original=}')
    g_o.create_picture(f'square_graph_roots_for_ana_rec.pdf', edgelist=[], node_size=3)
    # g_o.create_picture(f'square_roots_for_ana_rec.pdf', node_size=4, color_using_roots=True, edgelist=[])
    for q_prime in [.01, .06, .2, .3, .5, 1.5, 2.5, 4, 6.5, 9.5, 13, 18, 25, 50, 110, 230, 580, 1400, 6000, 15000]:
        # for q_prime in [20000,50000,100000,600000,2000000]:
        # for q_prime in [1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1]:
        g = copy.deepcopy(g_o)
        g.analysis_operator(q_prime)
        h = copy.deepcopy(g)
        h2 = copy.deepcopy(g)

        h.set_non_root_values_to_zero()
        h.reconstruction_operator(q_prime)
        for n in h2.nodes:
            if n in h2.roots:
                h2.nodes[n]['value'] = 0
        h2.reconstruction_operator(q_prime)
        vmin = min(h.get_minimum_value(), h2.get_minimum_value(), vmin_original) - 1
        vmax = max(h.get_maximum_value(), h2.get_maximum_value(), vmax_original) + 1
        vmin = min(vmin, -vmax)
        vmax = max(vmax, -vmin)

        red = plt.get_cmap('coolwarm', 256)(256)
        blue = plt.get_cmap('coolwarm', 256)(0)
        cmap = np.vstack(
            (
                matplotlib.colors.LinearSegmentedColormap.from_list('a', ['green', blue], 256)(np.linspace(0, 1, 256)),
                plt.get_cmap('coolwarm', 256)(np.linspace(0, 1, 256)),
                matplotlib.colors.LinearSegmentedColormap.from_list('a', [red, 'violet'], 256)(np.linspace(0, 1, 256)),
            )
        )
        cmap = matplotlib.colors.ListedColormap(cmap)
        ticks = [vmin, (vmin + vmin_original) / 2, vmin_original, vmin_original / 2, 0, vmax_original / 2,
                 vmax_original, (vmax_original + vmax) / 2, vmax]
        norm = matplotlib.colors.FuncNorm((lambda x: foward(x, vmin, vmin_original, vmax_original, vmax),
                                           lambda x: inverse(x, vmin, vmin_original, vmax_original, vmax)),
                                          vmin=vmin,
                                          vmax=vmax)
        print(f'{q_prime=}')
        print(f'{type(norm)=}')
        print(f'{norm=}')
        print(f'{norm(0)=}')
        print(f'{vmin=}')
        print(f'{vmax=}')
        print(f'{vmin_original=}')
        print(f'{vmax_original=}')
        g.create_picture_only_vertices(f'square_graph_analyzed_{q_prime=}.png', ticks=ticks, cmap=cmap,
                                       colorbar=False,
                                       norm=norm)
        g_o.create_picture_only_vertices(f'square_graph_original_{q_prime=}.png', ticks=ticks, cmap=cmap,
                                         colorbar=False,
                                         norm=norm)
        h.create_picture_only_vertices(f'square_graph_rec_without_detail_{q_prime=}.png', ticks=ticks,
                                       cmap=cmap, colorbar=False,
                                       norm=norm)
        h2.create_picture_only_vertices(f'square_graph_rec_only_detail_{q_prime=}.png', ticks=ticks, cmap=cmap,
                                        colorbar=False,
                                        norm=norm)
        g.create_picture_only_vertices(f'square_graph_colorbar_analyzed_{q_prime=}.png', ticks=ticks, cmap=cmap,
                                       norm=norm)
        g_o.create_picture_only_vertices(f'square_graph_colorbar_original_{q_prime=}.png', ticks=ticks,
                                         cmap=cmap,
                                         norm=norm)
        h.create_picture_only_vertices(f'square_graph_colorbar_rec_without_detail_{q_prime=}.png', ticks=ticks,
                                       cmap=cmap,
                                       norm=norm)
        h2.create_picture_only_vertices(f'square_graph_colorbar_rec_only_detail_{q_prime=}.png', ticks=ticks,
                                        cmap=cmap,
                                        norm=norm)
        for n in h2.nodes:
            h2.nodes[n]['value'] += h.nodes[n]['value']
        for n in h2.nodes:
            h2.nodes[n]['value'] -= g_o.nodes[n]['value']
        h2.create_picture_only_vertices(f'square_graph_colorbar_error_{q_prime=}.png', cmap=plt.cm.BrBG,
                                        norm=matplotlib.colors.CenteredNorm())


def visualize_downsampled_graph():
    graph = square_graph.SquareSignalProcessingGraph(30, standardweights=True)
    graph.wilson(3)
    dg = wilson.create_graph_from_matrix(graph.compute_Schur_complement(make_sparse=False), graph)
    dg.wilson(1.2345)
    print(len(dg.edges))
    dgd = wilson.create_graph_from_matrix(graph.compute_Schur_complement(make_sparse=True), graph)
    dgd.wilson(1.2345)
    print(len(dgd.edges))
    # We only draw edges that have high weights
    for e in dg.edges:
        if dg.edges[e]['weight'] / dg.graph['alpha'] > .01:
            dg.edges[e]['hidden'] = False
        else:
            dg.edges[e]['hidden'] = True
    print(len([e for e in dg.edges if dg.edges[e]['hidden'] == False]))
    dg.create_picture('downsampled_graph_edges_sparsified.pdf', color_using_roots=False, colorbar=False, node_color='gray',
                      edgelist=[e for e in dg.edges if dg.edges[e]['hidden'] == False],
                      node_size=1,  # edge_colorbar_position='bottom',
                      colorbar_for_edges=True)


def visualize_multiresolution():
    graph = square_graph.SquareSignalProcessingGraph(90, standardweights=False)
    q_list, q_prime_list, graph_list = wilson.multiresolution(graph, steps=6)
    vmax = graph.get_maximum_value()
    vmin = graph.get_minimum_value()
    vmax = max(vmax, -vmin)
    vmin = min(vmin, -vmax)

    for i, g in enumerate(graph_list):
        g.create_picture(f'analyzed_graph{i + 1}.png', color_using_roots=False, vmax=vmax, vmin=vmin, colorbar=False,
                         node_size=3)
    reconstr_graph_list = wilson.multi_reconstr(graph_list, q_prime_list)
    for i, g in enumerate(reconstr_graph_list):
        g.create_picture(f'reconstr_graph{i + 1}.png', color_using_roots=False, vmax=vmax, vmin=vmin, colorbar=False,
                         node_size=3)


def runtime_test_Schur_complement():
    nlist = [30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
    tlist = []
    for n in nlist:
        g = square_graph.SquareSignalProcessingGraph(n, standardweights=False)
        g.wilson(1.23456)
        q, _ = g._find_q(theta_1=.2, theta_2=1.)
        g.wilson(q)
        temp = time.perf_counter()
        g.compute_Schur_complement()
        tlist.append(time.perf_counter() - temp)
    nlist = np.array(nlist)
    tlist = np.array(tlist)
    plt.loglog(nlist, tlist / tlist[0], label='time')
    plt.loglog(nlist, (nlist / nlist[0]) ** 2, label='order 2')
    plt.loglog(nlist, (nlist / nlist[0]) ** 3, label='order 3')
    plt.loglog(nlist, (nlist / nlist[0]) ** 4, label='order 4')
    plt.legend()
    plt.show()
    return nlist, tlist


if __name__ == '__main__':
    g = square_graph.SquareSignalProcessingGraph(40, standardweights=False)
    print(type(g.nodes['23,1']))
    print(type(g.edges['2,3','3,3']))