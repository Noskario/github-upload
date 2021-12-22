import cProfile
import copy
import pstats
import random
import time

import matplotlib
import networkx as nx

# import example_graphs
import non_reversible_graphs
import numpy as np
import square_graph
import matplotlib.pyplot as plt

import wilson
import pickle


def triangletester():
    # g.show('tri.html',color_roots=True)
    # g.show('tri2.html',color_roots=False)
    counter = 0
    N = 7000
    for a in [.2]:
        debugger = []
        g = non_reversible_graphs.Triangle(a)
        qlist = [.7 * a, .8 * a, .85 * a, .9 * a, .93 * a, .95 * a, .97 * a, .98 * a, .99 * a, .995 * a, a, 1.02 * a,
                 1.3 * a]
        for q in qlist:
            hist = np.zeros(7)
            for _ in range(N):
                g.wilson(q)
                for i in range(1, 4):
                    if g.roots == {i}:
                        hist[i - 1] += 1
                    if g.roots == {1, 2, 3} - {i}:
                        hist[i + 2] += 1
                if g.roots == {1, 2, 3}:
                    hist[6] += 1
            plt.plot(["{1}", "{2}", "{3}", "{2,3}", "{1,3}", "{1,2}", "{1,2,3}"], hist / N, label=q)
            debugger.append(hist[0] / N)
        print(hist, sum(hist))
        plt.plot(np.array(qlist) / a, debugger)
        plt.legend()
        plt.show()


def triangleoptimizer():
    samplesize = 40000
    for a in [0, .001, .01, .05, .1, .4, .9]:
        g = non_reversible_graphs.Triangle(a)
        qlist = []
        plist = []
        for q in [i / 5 + .1 for i in range(15)]:
            qlist.append(q)
            plist.append(0)
            for _ in range(samplesize):
                g.wilson(q)
                if g.roots == {2, 3}:
                    plist[-1] += 1 / samplesize
        plt.plot(qlist, plist, label='a=' + str(a))
    plt.legend()
    plt.show()


def showhalfer_singletons():
    N = 100
    samplesize = 5000
    g = non_reversible_graphs.Halfer(N)
    zero_counter = 0
    singelton = 0
    hist = np.zeros(N + 1)
    for _ in range(samplesize):
        g.wilson(.0031)
        for j in range(N + 1):
            if g.roots == {j}:
                hist[j] += 1
        if g.roots == {0}:
            zero_counter += 1
        if len(g.roots) == 1:
            singelton += 1
    print(zero_counter / samplesize)
    print(singelton / samplesize)
    print(zero_counter / singelton)
    hist = np.array(hist)
    plt.plot(range(7), hist[:7] / singelton)
    plt.plot(range(7), np.array([1, 1, .5, .25, .125, 1 / 16, 1 / 32]) / 3)
    plt.show()
    g.show('gegenbeispiel.html')


def showhalfer_q():
    N = 100
    samplesize = 100
    g = non_reversible_graphs.Halfer(N)
    qlist = [.001, .002, .004, .007, .01, .02, .03, .05, .07, .1]
    K = len(qlist)
    hist1 = np.zeros(K)
    hist2 = np.zeros(K)
    hist3 = np.zeros(K)
    hist4 = np.zeros(K)
    for _ in range(samplesize):
        for j in range(K):
            g.wilson(qlist[j])
            if len(g.roots) == 1:
                hist1[j] += 1
            if len(g.roots) == 2:
                hist2[j] += 1
            if len(g.roots) == 3:
                hist3[j] += 1
            if len(g.roots) == 4:
                hist4[j] += 1
    plt.plot(qlist, hist1 / samplesize, label='one root')
    plt.plot(qlist, hist2 / samplesize, label='two roots')
    plt.plot(qlist, hist3 / samplesize, label='three roots')
    plt.plot(qlist, hist4 / samplesize, label='four roots')
    plt.legend()
    plt.show()


def waermepockets():
    g = square_graph.SquareWilsonConstantValues(40)
    g.nodes['3,6']['value'] = 3000
    g.wilson(.4)
    g.analysis_operator(-1)
    g.show('roots.html')
    g.show('g_orig.html', color_roots=False)
    g.wilson(2.3)
    g.analysis_operator(1.8)
    g.show('roots2.html')
    g.show('g_ana.html', color_roots=False)
    g.only_f_breve_reconstruction_operator(2.3)
    g.show('only_breve.html', color_roots=False)


# waermepockets()

# g.set_non_root_values_to_zero()
# g.show('g_without_detail.html', color_roots=False)
# g.reconstruction_operator(1.8)
# g.show('g_reconstr.html', color_roots=False)


def boese_funktioniert_nicht():
    g = square_graph.SquareWilsonConstantValues(40)
    g.nodes['3,6']['value'] = 3000
    g.wilson(.4)
    s1 = g.roots
    g.analysis_operator(-1)
    # g.show('root_distr.html')
    # g.show('exp.html', color_roots=False)
    maximum = g.get_maximum_value()
    minimum = g.get_miminum_value()
    maximum = max(maximum, 1)
    minimum = min(minimum, -1)
    maximum = max(maximum, -minimum)
    minimum = min(minimum, -maximum)
    print('maximum unedited:', g.get_maximum_value())
    print('minimum unedited:', g.get_miminum_value())
    # g.show('g_orig.html', color_roots=False, maximum=maximum, minimum=minimum)
    # g.show('roots.html')
    g.wilson(5.9)
    s2 = g.roots
    print(s1 - s2)
    print(s2 - s1)
    print(len(s1), len(s2), len(s1 - s2), len(s2 - s1))
    v = g.convert_values_to_np_array()
    g.analysis_operator(10)
    print('maximum edited:', g.get_maximum_value())
    print('minimum edited:', g.get_miminum_value())
    # g.show('g_ana.html', color_roots=False, maximum=maximum, minimum=minimum)
    # g.set_non_root_values_to_zero()
    # g.show('g_ana_to_zero.html', color_roots=False)
    g.reconstruction_operator(10)
    g.show('recr.html', color_roots=False, maximum=maximum, minimum=minimum)
    for node in g.nodes:
        g.nodes[node]['value'] -= v[g.nodes[node]['node_number']]
    g.show('error.html', color_roots=False, maximum=maximum, minimum=minimum)
    print('Maximaler Error:', g.get_maximum_value())
    print('Minimaler Error:', g.get_miminum_value())
    for node in g.nodes:
        g.nodes[node]['value'] = v[g.nodes[node]['node_number']]
    g.analysis_operator(10)
    g.only_f_breve_reconstruction_operator(1.0)
    g.show('only_X_breve.html', color_roots=False, maximum=maximum, minimum=minimum)
    for node in g.nodes:
        g.nodes[node]['value'] = v[g.nodes[node]['node_number']]
    g.analysis_operator(10)
    g.only_f_bar_reconstruction_operator(1.0)
    g.show('only_X_bar.html', color_roots=False, maximum=maximum, minimum=minimum)


def show_signal_processing():
    g = square_graph.SquareWilson(70, standardweights=False)
    g.nodes['3,6']['value'] = 3000
    g.wilson(3.4)
    g.show('g.html', color_roots=True)
    v = g.convert_values_to_np_array()
    maximum = g.get_maximum_value()
    minimum = g.get_miminum_value()
    maximum = max(maximum, 1)
    minimum = min(minimum, -1)
    maximum = max(maximum, -minimum)
    minimum = min(minimum, -maximum)
    print('maximum unedited:', g.get_maximum_value())
    print('minimum unedited:', g.get_miminum_value())
    g.show('g_orig.html', color_roots=False, maximum=maximum, minimum=minimum)
    g.analysis_operator(10)
    print('maximum edited:', g.get_maximum_value())
    print('minimum edited:', g.get_miminum_value())
    g.show('g_ana.html', color_roots=False, maximum=maximum, minimum=minimum)
    # g.set_non_root_values_to_zero()
    # g.show('g_ana_to_zero.html', color_roots=False)
    g.reconstruction_operator(10)
    g.show('recr.html', color_roots=False, maximum=maximum, minimum=minimum)
    for node in g.nodes:
        g.nodes[node]['value'] -= v[g.nodes[node]['node_number']]
    g.show('error.html', color_roots=False, maximum=maximum, minimum=minimum)
    print('Maximaler Error:', g.get_maximum_value())
    print('Minimaler Error:', g.get_miminum_value())
    for node in g.nodes:
        g.nodes[node]['value'] = v[g.nodes[node]['node_number']]
    g.analysis_operator(10)
    g.only_f_breve_reconstruction_operator(1.0)
    g.show('only_X_breve.html', color_roots=False, maximum=maximum, minimum=minimum)
    for node in g.nodes:
        g.nodes[node]['value'] = v[g.nodes[node]['node_number']]
    g.analysis_operator(10)
    g.only_f_bar_reconstruction_operator(1.0)
    g.show('only_X_bar.html', color_roots=False, maximum=maximum, minimum=minimum)


# show_signal_processing()


def zeug():
    g = square_graph.SquareWilson(50, standardweights=False)
    g.wilson(2.7)
    g.show('g.html')
    g.number_the_nodes()
    g.show('g.html', color_roots=False)
    for _ in range(5):
        g.analysis_operator(.3)
        g.show('ana.html', color_roots=False)


# print(g.nodes.data())
# print(g.create_Laplacian('non-roots', 'roots').toarray())
# print(g.convert_values_to_np_array().toarray())
print('###############################')


# g.analysis_operator(1)


# g.analysis_operator(.4)
# mat,tr_dict,_ = g.compute_Schur_complement()
# print(mat.toarray())
# g_down=wilson.create_graph_from_matrix(mat, tr_dict, g)
# g_down.wilson(.4)
# g_down.show('g_down.html')

def laufzeitentest_ana_recr():
    nlist = []
    alist = []
    rlist = []
    slist = []
    for i in range(14):
        n = 7 * i + 10
        g = square_graph.SquareWilson(n)
        g.wilson(.6)
        g.number_the_nodes()
        temp = time.time()
        g.analysis_operator(.3)
        alist.append(time.time() - temp)
        temp = time.time()
        L_Schur = g.compute_Schur_complement()
        slist.append(time.time() - temp)
        temp = time.time()
        g.reconstruction_operator(.3, L_Schur=L_Schur)
        rlist.append(time.time() - temp)
        nlist.append(n)
    alist = np.array(alist)
    alist = alist ** .25
    rlist = np.array(rlist)
    rlist = rlist ** .25
    slist = np.array(slist)
    slist = slist ** .25
    plt.plot(nlist, alist, label='ana')
    plt.plot(nlist, slist, label='nur Schur')
    plt.plot(nlist, rlist, label='reconstr')
    plt.legend()
    plt.show()


def laufzeitentest():
    nlist = []
    tlist = []
    for i in range(10):
        n = 5 * i + 30
        g = square_graph.SquareWilson(n)
        g.wilson(.1)
        temp = time.time()
        g.compute_Schur_complement()
        tlist.append(time.time() - temp)
        nlist.append(n)
    tlist = np.array(tlist)
    tlist = tlist ** .25
    plt.plot(nlist, tlist)
    plt.show()


# laufzeitentest_ana_recr()

# for item in temp:
#     for ele in item:
#         # print in a single line
#         print(ele, end=" ")
#     print()
# print(temp.dot(mat.toarray()))

# showhalfer_singletons()


def recruntersuchung():
    a = .3
    g = non_reversible_graphs.Triangle(a)
    g.wilson(.01)
    while g.roots != {2, 3}:
        g.wilson(.01)
    print(g.roots)
    g.reconstruction_operator(.2)
    g.show('recr_tri.html', color_roots=False)


def halfer_tester_multiresolution():
    g = non_reversible_graphs.Halfer(N=100)
    g.wilson(1.23456789)
    g.set_weights()
    g.show('original.html', color_roots=False)
    time.sleep(1)
    print(f"{g.graph['alpha']=}")
    with cProfile.Profile() as pr:
        g2 = g.one_step_in_multiresolution_scheme()
        print('///////////////////////////')
        for n in g2.nodes:
            print(n, g2.nodes[n]['value'])
        time.sleep(1)
        g3 = g2.one_step_in_multiresolution_scheme()
        time.sleep(1)
        g4 = g3.one_step_in_multiresolution_scheme()
        time.sleep(1)
        g5 = g4.one_step_in_multiresolution_scheme()
        time.sleep(1)
        g6 = g5.one_step_in_multiresolution_scheme()
    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.print_stats()


def funktionstest():
    g1 = square_graph.SquareWilson(30, standardweights=False)
    g1.wilson(1.23456789)
    g1.show('g.html', color_roots=False)
    g2, q1, qp1 = g1.one_step_in_multiresolution_scheme()
    g3, q2, qp2 = g2.one_step_in_multiresolution_scheme()
    g2.copy_values_from_other_graph_wherever_they_exist(g3)
    g2.set_non_root_values_to_zero()
    g2.reconstruction_operator(qp2)
    g2.show('g2.html', color_roots=False)
    g1.copy_values_from_other_graph_wherever_they_exist(g2)
    g1.set_non_root_values_to_zero()
    g1.reconstruction_operator(qp1)
    g1.show('g1.html', color_roots=False)


def save_multistep_to_file(steps=5, g=None, name_of_graph='graph'):
    if g is None:
        g = example_graphs.WattsStrogatzGraph(100, 5, .3)

    # g = square_graph.SquareWilson(60, standardweights=False)
    # g.wilson(1.23456789)
    # q_list, q_prime_list, graph_list=wilson.multi_resolution_and_reconstr(g,3)
    # g.draw_optimizer_path(.01,1000)
    # g.coupled_cont(3,2.7)
    # g.show('g.html',)
    q_list, q_prime_list, graph_list = wilson.multiresolution(g, steps=steps)
    print('q_list')
    print(q_list)
    print('q_prime_list')
    print(q_prime_list)
    print('graph_list')
    print(graph_list)
    result = [q_list, q_prime_list, graph_list]
    with open(f'multi_res_{name_of_graph}_{steps}_steps.pkl', 'wb') as file:
        pickle.dump(result, file)
    return result


def read_multi_and_do_stuff(steps=5, g=None, name_of_graph='graph'):
    with open(f'multi_res_{name_of_graph}_{steps}_steps.pkl', 'rb') as file:
        if g is None:
            g = example_graphs.WattsStrogatzGraph(100, 5, .3)

        q_list, q_prime_list, graph_list = pickle.load(file)
        print(q_list, q_prime_list, graph_list)
        wilson.multi_reconstr(g, q_list[:], q_prime_list[:], graph_list[:])


def test_R():
    g = square_graph.SquareWilson(60, standardweights=False)
    g.wilson(1.23456789)
    g2, q, q_prime = g.one_step_in_multiresolution_scheme(name_of_graph=f'downsampling_step')
    g2.show("g2.html", color_roots=False)
    print(g2 is g)
    g.show("g.html", color_roots=False)
    g.set_all_values_to_zero()
    g.copy_values_from_other_graph_wherever_they_exist(g2)
    g.show('g.html', color_roots=False)
    print(f'{len(g.roots)=}')
    for i in range(2):
        g.show('g_old.html', color_roots=False)
        _g = copy.deepcopy(g)
        print(g is _g)
        _g.reconstruction_operator(q_prime * (.4 + .1 * i))
        g.show('g_new.html', color_roots=False)
        _g.show('_g.html', color_roots=False)
    g.show('end.html', color_roots=False)


def compare_the_two_creation_methods_of_the_Laplacian():
    size = []
    time_list = []
    time2_list = []
    for i in range(10):
        s = 30 + 10 * i
        size.append(s)
        g = square_graph.SquareWilson(s, standardweights=False)
        g.wilson(1.23456789)
        t = time.perf_counter()
        L = g.create_Laplacian()
        time_list.append(time.perf_counter() - t)
        t = time.perf_counter()
        L = g.create_Laplacian2()
        time2_list.append(time.perf_counter() - t)

    plt.plot(size, np.array(time_list) ** .5, label='square root of time')
    plt.plot(size, np.array(time2_list) ** .5, label='square root of time2')
    plt.legend()
    plt.show()
    print(time_list)
    print(time2_list)
    # save_multistep_to_file(3)
    # read_multi_and_do_stuff(3)


def test_coupling_distribution(N: int):
    g = non_reversible_graphs.Halfer(N)
    q_final = g.coupled_cont(10, .004)
    print(f'(We are in testing function... {q_final=}')
    res = np.zeros(N + 1)
    # g.show('Halfer.html')
    for i in range(N + 1):
        if i in g.roots:
            res[i] = 1
    print('%%%%%%%%%%%%%%%%%%%', g.nodes[0], g.nodes[1])
    return res, q_final


def test_wilson_distribution(N: int, q):
    g = non_reversible_graphs.Halfer(N)
    g.wilson(q)

    res = np.zeros(N + 1)
    # g.show('Halfer.html')
    for i in range(N + 1):
        if i in g.roots:
            res[i] = 1
    return res


def debug_coupled(number_of_tries=100):
    N = 10
    fr = np.zeros(N + 1)
    q_list = []
    for _ in range(number_of_tries):
        res, q_final = test_coupling_distribution(N)
        print(res)
        print(q_final)
        print('----------------')
        fr += res
        q_list.append(q_final)
    print(fr.sum())
    print(fr)
    plt.plot(range(N + 1), fr)
    plt.show()
    fr = np.zeros(N + 1)
    for q in q_list:
        fr += test_wilson_distribution(N, q)
    print(fr.sum())
    print(fr)
    plt.plot(range(N + 1), fr)
    plt.show()


def make_one_complete_multi(steps, graph, name_of_graph):
    g = copy.deepcopy(graph)
    save_multistep_to_file(steps, g, name_of_graph)
    g = copy.deepcopy(graph)
    read_multi_and_do_stuff(steps, g, name_of_graph)


def test_io_for_networkx_and_pdf_creating():
    print('Guten Morgen!')
    g = example_graphs.WattsStrogatzGraph(100, 20, .2)
    g = square_graph.SquareWilson(90, standardweights=False)
    g.wilson(12.3456)
    # g.show('nodes.html')
    temp = time.perf_counter()
    print([g.nodes[n]['value'] for n in g.nodes])
    g.create_pdf('roots.pdf')
    g.create_pdf('values.pdf', color_using_roots=False)
    print(time.perf_counter() - temp)

    temp = time.perf_counter()
    nx.write_gexf(g, "test.gexf")
    # g.show('expampleSquare.html', color_roots=False)
    print('gexfdump:', time.perf_counter() - temp)
    temp = time.perf_counter()
    h = nx.read_gexf("test.gexf")
    print('gexfload:', time.perf_counter() - temp)
    temp = time.perf_counter()
    with open('test.pickle', 'wb') as file:
        pickle.dump(g, file)
    print('pickledump:', time.perf_counter() - temp)
    temp = time.perf_counter()
    with open('test.pickle', 'rb') as file:
        h = pickle.load(file)
    print('pickleload:', time.perf_counter() - temp)
    # h.show('g.html')

    # with open('multi_res_facebook_3_steps.pkl', 'rb') as file:  Reading in takes 70? seconds
    # with open('multi_res_5_steps_on_60_square.pkl', 'rb') as file:  # Reading in takes 128.20001770000002 seconds
    #     q_list, q_prime_list, graph_list = pickle.load(file)
    # print(f'Reading in takes {time.perf_counter() - temp} seconds')
    # print('************************')
    # print(q_list)
    # print(q_prime_list)
    # print(graph_list)
    # for g in graph_list:
    #    print(g.roots)
    #    g.show('g_60_temp.html', color_roots=False)
    # make_one_complete_multi(3, g, 'facebook')  # Takes about 20 min
    # g.wilson(3.45)
    # g.show('facebook.html')
    # g.show('facebook.html',color_roots=False)


def pdf_alternative():
    g = square_graph.SquareWilson(90, standardweights=False)
    g.stack_version_of_wilson(1.234, start_from_scratch=True, )
    g.create_pdf(vmin=-10000, vmax=10000, color_using_roots=False)
    for _ in range(1):
        temp = time.perf_counter()
        g.stack_version_of_wilson(12.3456, start_from_scratch=True)
        print(time.perf_counter() - temp)
        # g.show('gg.html')

    for _ in range(0):
        g.coupled_cont(120.345, 1.234)
    plt.clf()
    plt.scatter(
        x=[g.nodes[n]['x'] for n in g.nodes],
        y=[g.nodes[n]['y'] for n in g.nodes],
        c=[g.nodes[n]['value'] for n in g.nodes],
        cmap=matplotlib.cm.coolwarm

    )
    x_edges_start = np.array([g.nodes[e[0]]['x'] for e in g.edges if g.edges[e]['hidden'] == False])
    y_edges_start = np.array([g.nodes[e[0]]['y'] for e in g.edges if g.edges[e]['hidden'] == False])
    x_edges_end = np.array([g.nodes[e[1]]['x'] for e in g.edges if g.edges[e]['hidden'] == False])
    y_edges_end = np.array([g.nodes[e[1]]['y'] for e in g.edges if g.edges[e]['hidden'] == False])
    x_edges_diff = x_edges_end - x_edges_start
    y_edges_diff = y_edges_end - y_edges_start
    for i in range(len(x_edges_diff)):
        plt.arrow(x_edges_start[i], y_edges_start[i], x_edges_diff[i], y_edges_diff[i], width=.08)

    plt.show()
    # g.create_pdf()


def halfer_root_0_picture():
    g = non_reversible_graphs.Halfer(10)

    g.stack_version_of_wilson(.02, start_from_scratch=True, renumber_roots_after_finishing=True)
    while g.roots != {0}:
        g.stack_version_of_wilson(.02, start_from_scratch=True, renumber_roots_after_finishing=True)
    g.create_pdf('halfer_roots.pdf')
    g.create_pdf('halfer_color_start.pdf', color_using_roots=False, print_values=True)
    g.analysis_operator(1)
    print([g.nodes[n] for n in g.nodes])
    g.create_pdf('halfer_color_ana.pdf', color_using_roots=False, print_values=True)
    g.reconstruction_operator(1)
    g.create_pdf('halfer_color_reconstr.pdf', color_using_roots=False, print_values=True)

    print([g.nodes[n] for n in g.nodes])


def save_coupling():
    g = square_graph.SquareWilson(5)
    i = 0
    q = 3
    g.stack_version_of_wilson(q, renumber_roots_after_finishing=False, start_from_scratch=True)
    m = len(g.roots)
    while q > 1:
        i += 1

        q = g._coupled_q_next(q)
        g.create_pdf(f'graph_with_{i=}.pdf', node_size=280, arrowsize=14)

        m = len(g.roots)
        print(i, q)


def print_stored_root_distribution_for_halfer():
    # These are the numbers from a calculation that I did some time ago that took about one hour
    coupled_list = np.array([3310, 3285, 1719, 882, 458, 224, 127, 69, 51, 22, 29.])
    direct_list = np.array([3332, 3457, 1639, 847, 403, 186, 137, 60, 48, 28, 25.])

    samplesize = 10000
    width = .3
    plt.bar(np.arange(11) - width, coupled_list / samplesize, width=width,
            label='Coupling process stopped for q<0.004\nSamplesize:10000')
    plt.bar(np.arange(11), direct_list / samplesize, width=width,
            label="Direct Application of Wilson's algorithm\n for the same q as above\nSamplesize:10000")
    # Now we compute the expected proportions
    exp_list = np.array([1.] + [.5 ** i for i in range(10)])
    exp_list = exp_list / exp_list.sum()
    plt.bar(np.arange(11) + width, exp_list, width=width,
            label="Theoretical probability for q->0")
    plt.xticks(np.arange(11), np.arange(11))
    # plt.title('Probability for a vertex becoming a root: \nNumerical approximations and theoretical results')
    plt.legend()

    plt.show()


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
    g_o = square_graph.SquareWilson(90, standardweights=False)
    g_o.wilson(q=12.345)
    vmin_original = g_o.get_miminum_value()
    vmax_original = g_o.get_maximum_value()
    vmax_original = max(-vmin_original, vmax_original)
    vmin_original = min(-vmax_original, vmin_original)
    print(f'{vmin_original=}')
    print(f'{vmax_original=}')
    g_o.create_pdf(f'square_graph_roots_for_ana_rec.pdf', edgelist=[], node_size=3)
    # g_o.create_pdf(f'square_roots_for_ana_rec.pdf', node_size=4, color_using_roots=True, edgelist=[])
    for q_prime in np.arange(10) + .5:
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
        vmin = min(h.get_miminum_value(), h2.get_miminum_value(), vmin_original) - 1
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
        g.create_pdf_using_custom_color_scheme(f'square_graph_analyzed{q_prime=}.pdf', ticks=ticks, cmap=cmap,
                                               norm=norm)
        g_o.create_pdf_using_custom_color_scheme(f'square_graph_original{q_prime=}.pdf', ticks=ticks, cmap=cmap,
                                                 norm=norm)
        h.create_pdf_using_custom_color_scheme(f'square_graph_rec_without_detail{q_prime=}.pdf', ticks=ticks, cmap=cmap,
                                               norm=norm)
        h2.create_pdf_using_custom_color_scheme(f'square_graph_rec_only_detail{q_prime=}.pdf', ticks=ticks, cmap=cmap,
                                                norm=norm)
        for n in h2.nodes:
            h2.nodes[n]['value'] += h.nodes[n]['value']
        h2.create_pdf_using_custom_color_scheme(f'square_graph_rec_sum{q_prime=}.pdf', ticks=ticks, cmap=cmap,
                                                norm=norm)
        for n in h2.nodes:
            h2.nodes[n]['value'] -= g_o.nodes[n]['value']
        h2.create_pdf_using_custom_color_scheme(f'square_graph_error{q_prime=}.pdf', cmap=plt.cm.BrBG,
                                                norm=matplotlib.colors.CenteredNorm())


if __name__ == '__main__':
    visualize_analysis_operator()
