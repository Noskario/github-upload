import matplotlib.cm
import matplotlib.pyplot as plt
import matplotlib.colors
import pyvis
import networkx as nx
import random
import time
import copy
import scipy.sparse
import scipy.sparse.linalg
import numpy as np
from typing import List


class WilsonGraph(nx.DiGraph):
    def __init__(self):
        super().__init__()
        self.weights_are_set = False
        self.roots = set()

    def set_weights(self):
        weightlist = []
        for v in self.nodes:
            w = self.get_node_weight(v)
            self.nodes[v]['weight'] = w
            weightlist.append(w)
        alpha = max(weightlist)
        self.graph['alpha'] = alpha
        self.weights_are_set = True

    def LERW(self, start, sinks, q=0):
        activenodeid = start
        # Jetzt starte Skelettprozess (künstlich beschleunigt, indem alpha durch w(x) ersetzt)
        trajectory = [start]
        while True:
            r = random.random()
            if r < q / (self.nodes[activenodeid]['weight'] + q):  # LERW killed with exp time
                self.roots.add(activenodeid)
                return trajectory
            chosennodeid = activenodeid
            temp = q
            # print('~',end='')
            for nbr in self[activenodeid]:
                temp += self[activenodeid][nbr]['weight']
                if r < temp / (self.nodes[activenodeid]['weight'] + q):
                    chosennodeid = nbr
                    break
            if chosennodeid in sinks:
                trajectory.append(chosennodeid)
                return trajectory
            if chosennodeid in trajectory:
                while chosennodeid in trajectory:
                    trajectory.pop()
                trajectory.append(chosennodeid)
                activenodeid = chosennodeid
            else:
                trajectory.append(chosennodeid)
                activenodeid = chosennodeid

    # def wilson(self, q, sinks=None, reformat_graph=True, renumber_roots_after_finishing=True):
    #     """Applies Wilson's Algorithm to a given graph with given sinks and killing rate q
    #     Wilson's Algorithm outputs a random forest obeying a specific distribution
    #
    #     returns a graph where roots of the forest are orange/black and vertices, that are no roots are navy/blue"""
    #     # Graph vorbereiten, auf Standardwerte setzen
    #     if sinks is None:
    #         sinks = []
    #     sinks = set(sinks)
    #     if reformat_graph:
    #         self.roots = copy.copy(sinks)
    #         for e in self.edges:
    #             self.edges[e]['hidden'] = True
    #         # self.show('test.html')
    #     temp_time = time.time()
    #     if not self.weights_are_set:
    #         self.set_weights()
    #
    #     activenodes = (set(self.nodes) - sinks)
    #     counter = 0
    #     edges_that_form_the_spanning_forest = []
    #     temp_time1 = 0
    #     while bool(activenodes):
    #         counter += 1
    #         # choose an arbitrary vertex and start a LERW
    #         start = activenodes.pop()
    #         activenodes.add(start)
    #         path = self.LERW(start, sinks, q)
    #         temp = time.time()
    #         for i in range(len(path) - 1):
    #             self.edges[path[i], path[i + 1]]['hidden'] = False
    #         temp_time1 += time.time() - temp
    #         # Vertices of the path are now also sinks, since they belong to the constructed random forest
    #         sinks.update(path)
    #         for n in path:
    #             activenodes.discard(n)
    #
    #     # print('counter=', counter)
    #     # print('wison_graphic', temp_time1)
    #     wilson_time = time.time() - temp_time
    #     # print('wison_without_graphics', wilson_time - temp_time1)
    #     # print('wison', wilson_time)
    #     if renumber_roots_after_finishing:
    #         self.number_the_nodes()
    #     return edges_that_form_the_spanning_forest
    def wilson(self, q: float, roots: set = None, ):
        self.stack_version_of_wilson(q, renumber_roots_after_finishing=True, start_from_scratch=True, roots=roots)

    def show(self, graphname, color_roots=True, minimum=None, maximum=None):
        """
        :param graphname: has to have the format "name_of_graph.html"
        :return: does not return anything, but draws the graph creating an html-file
        """
        h = pyvis.network.Network('800px', '1000px')
        g = nx.Graph()
        if color_roots:
            self.color_roots()
        else:
            self.color_using_values(minimum=minimum, maximum=maximum)
        for n in self.nodes:
            g.add_node(n)
            for attr in self.nodes[n]:
                g.nodes[n][attr] = self.nodes[n][attr]
            g.nodes[n]['title'] = str(g.nodes[n]['value'])
        for e in self.edges:
            try:
                if self.edges[e]['hidden'] == False:
                    g.add_edge(*e)
            except KeyError:
                # This only happens if edge has no attribute 'hidden'. In this case the graph is probably "plain"
                # in the sense that none of my algorithms has touched it. It makes sense to assume that
                # every edge should be plotted in case nobody has ever set it hidden using
                # self.edges[e]['hidden'] = True
                g.add_edge(*e)
        h.from_nx(g)
        h.show_buttons()
        h.toggle_physics(False)
        h.show(graphname)
        time.sleep(.2)

    def create_pdf(self, filename='graph.pdf', color_using_roots=True, print_values=False, show_immediately=False,
                   colorbar=True, colorbar_for_edges=False, **kwargs):
        plt.clf()
        if print_values:
            kwargs['labels'] = {n: "%.2g" % self.nodes[n]['value'] for n in self.nodes}
            kwargs['with_labels'] = True,
            if 'font_size' not in kwargs:
                kwargs['font_size'] = 5
        if color_using_roots:
            node_color = ['darkred' if n in self.roots else 'palevioletred' for n in self.nodes]
            kwargs['node_color'] = node_color
            edgelist = [e for e in self.edges if self.edges[e]['hidden'] == False]
            if 'edgelist' not in kwargs:
                kwargs['edgelist'] = edgelist

        else:

            if 'vmin' not in kwargs or 'vmax' not in kwargs:
                vmin = self.get_minimum_value()
                vmax = self.get_maximum_value()
                vmin = min(vmin, -1)
                vmax = max(vmax, 1)
                vmin = min(vmin, -vmax)
                vmax = max(vmax, -vmin)
                kwargs['vmin'] = vmin
                kwargs['vmax'] = vmax
            if 'node_color' not in kwargs:
                node_color = [self.nodes[n]['value'] for n in self.nodes]
                kwargs['node_color'] = node_color
            if 'cmap' not in kwargs:
                kwargs['cmap'] = plt.cm.coolwarm
            cmap = kwargs['cmap']
            vmin = kwargs['vmin']
            vmax = kwargs['vmax']
            if colorbar:
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin, vmax))
                plt.colorbar(sm)
            if colorbar_for_edges:
                if 'edge_cmap' not in kwargs:
                    kwargs['edge_cmap'] = plt.cm.viridis_r
                edge_cmap = kwargs['edge_cmap']
                if 'edge_color' not in kwargs:
                    kwargs['edge_color'] = [self.edges[e]['weight'] for e in self.edges]
                edge_color = kwargs['edge_color']
                edge_vmin = min(edge_color)
                edge_vmax = max(edge_color)
                sm = plt.cm.ScalarMappable(cmap=edge_cmap, norm=plt.Normalize(edge_vmin, edge_vmax))
                plt.colorbar(sm)

            if 'edgelist' not in kwargs:
                kwargs['edgelist'] = []
        if 'node_size' not in kwargs:
            kwargs['node_size'] = 20
        if 'arrowsize' not in kwargs:
            kwargs['arrowsize'] = 3
        if 'with_labels' not in kwargs:
            kwargs['with_labels'] = False
        if 'width' not in kwargs:
            kwargs['width'] = .4
        nx.draw_networkx(self,
                         pos={n: (self.nodes[n]['x'], -self.nodes[n]['y']) for n in self.nodes},
                         **kwargs,
                         )
        plt.axis('equal')
        plt.box(False)
        plt.savefig(filename, bbox_inches='tight')
        if show_immediately:
            plt.show()

    def create_pdf_using_custom_color_scheme(self, filename='graph.pdf', show_immediately=False, title=None, ticks=None,
                                             colorbar=True,
                                             **kwargs):
        plt.clf()

        if 'norm' not in kwargs:  # vmin and vmax are not allowed if a color-norm is already given
            if 'vmin' not in kwargs or 'vmax' not in kwargs:
                vmin = self.get_minimum_value()
                vmax = self.get_maximum_value()
                vmin = min(vmin, -1)
                vmax = max(vmax, 1)
                vmin = min(vmin, -vmax)
                vmax = max(vmax, -vmin)
                kwargs['vmin'] = vmin
                kwargs['vmax'] = vmax

        if 'cmap' not in kwargs:
            kwargs['cmap'] = plt.cm.coolwarm
        if 's' not in kwargs:
            kwargs['s'] = 3

        plt.scatter(
            x=[self.nodes[n]['x'] for n in self.nodes],
            y=[-self.nodes[n]['y'] for n in self.nodes],
            c=[self.nodes[n]['value'] for n in self.nodes],
            **kwargs,
        )
        plt.axis('equal')
        if colorbar:
            plt.colorbar(ticks=ticks)
        plt.axis('off')
        plt.box(False)
        plt.savefig(filename, bbox_inches='tight')
        if show_immediately:
            plt.show()

    def get_node_weight(self, node):
        nodeweight = 0
        for e in self.edges(node):
            nodeweight += self.edges[e]['weight']
        return nodeweight

    def choose_random_neighbor(self, node):
        if not self.weights_are_set:
            self.set_weights()
        nodeweight = self.nodes[node]['weight']
        temp = 0
        r = nodeweight * random.random()
        for nbr in self[node]:
            temp += self[node][nbr]['weight']
            if temp > r:
                return nbr
        raise 'No neighbor chosen. That is strange.'

    def color_roots(self):
        for v in self.nodes:
            self.nodes[v]['color'] = 'palevioletred'
        for v in self.roots:
            self.nodes[v]['color'] = 'darkred'

    def get_maximum_value(self):
        l = []
        for v in self.nodes:
            l.append(self.nodes[v]['value'])
        return max(l)

    def get_minimum_value(self):
        l = []
        for v in self.nodes:
            l.append(self.nodes[v]['value'])
        return min(l)

    def color_using_values(self, maximum=None, minimum=None):
        if maximum is None:
            maximum = self.get_maximum_value()
        if minimum is None:
            minimum = self.get_minimum_value()
        # begin: Damit null weiß wird
        minimum = min(minimum, -1)
        maximum = max(maximum, 1)
        minimum = min(minimum, -maximum)
        maximum = max(maximum, -minimum)
        # end
        norm = matplotlib.colors.Normalize(vmin=minimum, vmax=maximum)
        cmap = matplotlib.cm.coolwarm
        for v in self.nodes:
            self.nodes[v]['color'] = matplotlib.colors.to_hex(cmap(norm(self.nodes[v]['value'])))

    def _process_activated_node2(self, x, q2):
        """
        :param x:
        :param q2:
        :return:
        """
        number_of_reprocessed_nodes = 0
        chosen_nbr = self.choose_random_neighbor(x)
        # self.show(f'process_activated_node2({x=}---{chosen_nbr=}).html')
        y = chosen_nbr
        trajectory = [x, y]
        # We follow the already uncovered arrows until we either walk a loop
        # or go into a root (=another tree)
        while True:
            # Go one step
            for nbr in self[y]:
                if self.edges[y, nbr]['hidden'] == False:
                    y = nbr
                    trajectory.append(y)
                    break
            # Did we do a loop? Did we go to another root?
            if y == x:  # Loop
                # In this case we have to remove the loop and do t
                tree_of_x = self._get_tree_of_root(x)
                number_of_reprocessed_nodes += len(tree_of_x)
                self.roots.remove(x)
                for i in range(len(trajectory) - 1):
                    self.edges[trajectory[i], trajectory[i + 1]]['hidden'] = True
                # self.show(f'first_trajectory_is_a_loop:_{str(trajectory).replace(" ","_")}.html')
                self.stack_version_of_wilson(q=q2, active_nodes=tree_of_x)
                break
            elif y in self.roots:  # Another root
                number_of_reprocessed_nodes += 1
                self.edges[x, chosen_nbr]['hidden'] = False
                self.roots.remove(x)
                # print(f'First trajectory goes to root: {trajectory}')
                break
        # print(f'{number_of_reprocessed_nodes=}')
        # self.show('Funktionstest_der_verbesserten_gekoppelten_Variante.html')
        return number_of_reprocessed_nodes

    def _coupled_q_next(self, q):
        r = random.random()
        m = len(self.roots)
        alpha = self.graph['alpha']
        q_next = alpha / (r ** (-1 / m) * ((alpha + q) / q) - 1)
        x = random.choice(list(self.roots))
        # print('\nMache die Sache neu mit x=', x)
        # self.show('coupling_step.html')
        self._process_activated_node2(x, q_next)
        # print(f'{q_next =}')
        return q_next

    def coupled_cont(self, qmax, q_stop):
        q = qmax
        self.stack_version_of_wilson(q, renumber_roots_after_finishing=False, start_from_scratch=True)
        while q > q_stop:
            q = self._coupled_q_next(q)
        self.number_the_nodes()
        return q

    def get_tree_predecessors(self, node):
        s = set()
        # print(f'{node=}')
        # print(f'{self.roots}')
        for nbr in self.predecessors(node):
            if self.edges[nbr, node]['hidden'] == False:
                s.add(nbr)
        # print(f'~~~{s=}')
        return s

    def _get_tree_of_root(self, node):
        counter = 0
        # self.show('gettreeofroot.html')
        explorer_set = set()
        explorer_set.add(node)
        explored_set = set()
        aux_set = set()
        while explorer_set:
            counter += 1
            if counter > len(self.nodes):
                for e in self.edges:
                    print(e, self.edges[e])
                print(f'{self.roots=}')
                print(explorer_set)
                self.show('error.html')
                raise "Tree was not really a tree..."
            # print(f'{explorer_set=}')
            for x in explorer_set:
                aux_set.update(self.get_tree_predecessors(x))
                explored_set.add(x)
            # print(f'{explored_set=}')
            explorer_set = aux_set
            aux_set = set()
        return explored_set

    def number_the_nodes(self):
        i = 0
        for node in self.roots:
            self.nodes[node]['node_number'] = i
            i += 1
        for node in set(self.nodes) - self.roots:
            self.nodes[node]['node_number'] = i
            i += 1
        return i

    def create_Laplacian2(self, selected_rows='all', selected_cols='all'):
        # Does the the same as create_Laplcian and is easier to read but 4 times slower, so I do not use it
        if selected_cols not in ['all', 'roots', 'non-roots']:
            raise ValueError("Only 'all', 'roots', 'non-roots' are accepted as arguments")
        if selected_rows not in ['all', 'roots', 'non-roots']:
            raise ValueError("Only 'all', 'roots', 'non-roots' are accepted as arguments")
        if not self.weights_are_set:
            self.set_weights()
        cols = set(self.nodes)
        col_index_shift = 0
        if selected_cols == 'roots':
            cols = (self.roots)
        if selected_cols == 'non-roots':
            cols -= set(self.roots)
            col_index_shift = len(self.roots)
        rows = set(self.nodes)
        row_index_shift = 0
        if selected_rows == 'roots':
            rows = (self.roots)
        if selected_rows == 'non-roots':
            rows -= (self.roots)
            row_index_shift = len(self.roots)
        mat = scipy.sparse.dok_matrix((len(rows), len(cols)), dtype=np.float64)
        for n in rows:
            for nbr in self[n]:
                if nbr in cols:
                    mat[self.nodes[n]['node_number'] - row_index_shift, self.nodes[nbr][
                        'node_number'] - col_index_shift] = self.edges[n, nbr]['weight']
        if selected_rows == selected_cols:
            for n in rows:
                mat[self.nodes[n]['node_number'] - row_index_shift, self.nodes[n]['node_number'] - row_index_shift] = - \
                    self.nodes[n]['weight']
        return mat.tocsc()

    def create_Laplacian(self, selected_rows='all', selected_cols='all') -> scipy.sparse.csc_matrix:
        if selected_cols not in ['all', 'roots', 'non-roots']:
            raise ValueError("Only 'all', 'roots', 'non-roots' are accepted as arguments")
        if selected_rows not in ['all', 'roots', 'non-roots']:
            raise ValueError("Only 'all', 'roots', 'non-roots' are accepted as arguments")
        if not self.weights_are_set:
            self.set_weights()
        row_list = []
        col_list = []
        weight_list = []
        row_index_shift = 0
        col_index_shift = 0

        chosen_rows = self.nodes
        if selected_rows == 'roots':
            chosen_rows = self.roots
        elif selected_rows == 'non-roots':
            chosen_rows = set(self.nodes) - set(self.roots)
            row_index_shift = len(self.roots)

        chosen_cols = self.nodes
        if selected_cols == 'roots':
            chosen_cols = self.roots
        elif selected_cols == 'non-roots':
            chosen_cols = set(self.nodes) - set(self.roots)
            col_index_shift = len(self.roots)

        for node in set(chosen_rows) & set(chosen_cols):
            row_list.append(self.nodes[node]['node_number'] - row_index_shift)
            col_list.append(self.nodes[node]['node_number'] - col_index_shift)
            weight_list.append(-self.nodes[node]['weight'])
        for node in chosen_rows:
            for nbr in self[node]:
                if nbr in chosen_cols:
                    row_list.append(self.nodes[node]['node_number'] - row_index_shift)
                    col_list.append(self.nodes[nbr]['node_number'] - col_index_shift)
                    weight_list.append(self.edges[node, nbr]['weight'])
        row_list = np.array(row_list)
        col_list = np.array(col_list)
        weight_list = np.array(weight_list)
        mat = scipy.sparse.csc_matrix((weight_list, (row_list, col_list)), shape=(len(chosen_rows), len(chosen_cols)),
                                      dtype=float)
        return mat

    def convert_to_numpy_matrix(self, selected_rows='all', selected_cols='all'):
        mat = self.create_Laplacian(selected_rows=selected_rows, selected_cols=selected_cols)
        return mat.toarray()

    def compute_Schur_complement(self):
        # self.show('test.html')
        mat_qq = self.create_Laplacian(selected_rows='roots', selected_cols='roots')
        mat_qu = self.create_Laplacian(selected_rows='roots', selected_cols='non-roots')
        mat_uq = self.create_Laplacian(selected_rows='non-roots', selected_cols='roots')
        mat_uu = self.create_Laplacian(selected_rows='non-roots', selected_cols='non-roots')

        mat_uu_inv = scipy.sparse.linalg.inv(mat_uu)
        if mat_uu_inv.shape == (1,):
            mat_uu_inv = scipy.sparse.csc_matrix(mat_uu_inv.reshape((1, 1)))

        return mat_qq - mat_qu @ mat_uu_inv @ mat_uq

    def convert_values_to_np_array(self):
        n = len(self.nodes)
        mat = np.zeros(n)
        for node in self.nodes:
            mat[self.nodes[node]['node_number']] = self.nodes[node]['value']
        return mat

    def analysis_operator(self, q):
        n = len(self.nodes)
        m = len(self.roots)
        L = self.create_Laplacian()
        A = scipy.sparse.identity(n, format='csc') - L / q
        f = self.convert_values_to_np_array()
        f_analysed = scipy.sparse.linalg.spsolve(A, f)
        f_analysed[m:] -= f[m:]
        # self.show('original.html', color_roots=False)
        for node in self.nodes:
            self.nodes[node]['value'] = f_analysed[self.nodes[node]['node_number']]
        # self.show('ana.html', color_roots=False)

    def reconstruction_operator(self, q, L_Schur=None):
        f_ana = self.convert_values_to_np_array()
        m = len(self.roots)
        n = len(self.nodes)
        f_bar = f_ana[:m]
        f_breve = f_ana[m:]
        L_breve_breve = self.create_Laplacian(selected_rows='non-roots', selected_cols='non-roots')
        L_bar_breve = self.create_Laplacian(selected_rows='roots', selected_cols='non-roots')
        L_breve_bar = self.create_Laplacian(selected_rows='non-roots', selected_cols='roots')
        if L_Schur is None:
            L_Schur = self.compute_Schur_complement()
        f = np.zeros(n)
        print(f'{L_Schur.shape=}')
        print(f'{type(L_Schur)=}')
        print(f'{L_Schur.data=}')
        print(f'{f[:m].shape=}')
        print(f'{f_bar.shape=}')
        print(f'{(1 / q * L_Schur.dot(f_bar)).shape=}')
        f[:m] += f_bar - 1 / q * L_Schur.dot(f_bar)
        f[:m] += L_bar_breve.dot(scipy.sparse.linalg.spsolve(-L_breve_breve, f_breve))
        f[m:] += scipy.sparse.linalg.spsolve(-L_breve_breve, L_breve_bar.dot(f_bar))
        f[m:] += q * scipy.sparse.linalg.spsolve(L_breve_breve, f_breve) - f_breve
        for node in self.nodes:
            self.nodes[node]['value'] = f[self.nodes[node]['node_number']]

    def set_non_root_values_to_zero(self):
        for node in set(self.nodes) - self.roots:
            self.nodes[node]['value'] = 0

    def set_all_values_to_zero(self):
        for node in self.nodes:
            self.nodes[node]['value'] = 0

    def gamma_inverse_approx(self, q):
        return 1 / q * len(self.nodes) / (1 + len(self.nodes) - len(self.roots))

    def beta_inverse_approx(self):
        return 1 / self.graph['alpha'] * (len(self.nodes) - len(self.roots)) / len(self.roots)

    def alpha_bar_approx(self, q):
        return q * (len(self.nodes) - len(self.roots)) / (1 + len(self.roots))

    def _find_q(self, theta_1, theta_2):
        qmax, q_stop = theta_2 * self.graph['alpha'], theta_1 * self.graph['alpha']
        q = qmax
        minimizer_q = qmax
        self.stack_version_of_wilson(qmax, renumber_roots_after_finishing=False, start_from_scratch=True)
        temp = self.alpha_bar_approx(q) * self.gamma_inverse_approx(q)
        minimizer_root_set = copy.copy(self.roots)
        while q > q_stop:
            q = self._coupled_q_next(q)
            if self.alpha_bar_approx(q) * self.gamma_inverse_approx(q) < temp:
                minimizer_q = q
                minimizer_root_set = copy.copy(self.roots)
                temp = self.alpha_bar_approx(q) * self.gamma_inverse_approx(q)
        self.number_the_nodes()
        return minimizer_q, minimizer_root_set

    def _find_q_prime(self, q, L_Schur=None):
        if L_Schur is None:
            L_Schur = self.compute_Schur_complement()
        downsampled_graph = create_graph_from_matrix(L_Schur, original_graph=self)
        downsampled_graph.set_weights()
        alpha_bar = downsampled_graph.graph['alpha']
        return 2 * alpha_bar * len(self.roots) / (len(self.nodes) - len(self.roots)), downsampled_graph

    def copy_values_from_other_graph_wherever_they_exist(self, g):
        for n in self.nodes:
            try:
                self.nodes[n]['value'] = copy.deepcopy(g.nodes[n]['value'])
                # print(type(g.nodes[n]['value']))
            except KeyError:
                pass

    def one_step_in_multiresolution_scheme(self, theta1=.2, theta2=1, name_of_graph='downsampled',
                                           show_created_graph=False):
        if not self.weights_are_set:
            self.set_weights()
        q, _ = self._find_q(theta1, theta2)
        self.stack_version_of_wilson(q, renumber_roots_after_finishing=True, start_from_scratch=True)
        L_Schur = self.compute_Schur_complement()
        q_prime, downsampled_graph = self._find_q_prime(q, L_Schur)
        self.analysis_operator(q_prime)
        downsampled_graph.copy_values_from_other_graph_wherever_they_exist(self)
        if show_created_graph:
            downsampled_graph.show(f'{name_of_graph}.html', color_roots=False)
        return copy.deepcopy(self), downsampled_graph, q, q_prime

    def stack_version_of_wilson(self, q, active_nodes: set = 'all', renumber_roots_after_finishing=False,
                                start_from_scratch=False, roots=None):
        """
        :param q:
        :param active_nodes: Set of nodes that are still active. Also 'all' is accepted.
        :param renumber_roots_after_finishing:
        :param start_from_scratch: set True if you want to delete all uncovered edges/roots and start all over
        :param roots: in case you start from scratch give the a priori roots here. Must be a set. Argument is ignored if start_from_scratch=True
        :return:
        """
        if not self.weights_are_set:
            self.set_weights()

        if start_from_scratch:
            if roots is None:
                roots = set()
            self.roots = copy.copy(roots)
            for e in self.edges:
                self.edges[e]['hidden'] = True

        if active_nodes == 'all':
            active_nodes = self.nodes

        active_nodes = set(active_nodes)
        sinks = set(self.nodes) - active_nodes
        sinks = copy.copy(sinks)
        while active_nodes:

            node_currently_in_use = active_nodes.pop()
            trajectory = [node_currently_in_use]
            while True:  # Go one step

                nbr = self.follow_arrow_if_exist_else_create_arrow_and_follow_or_get_killed(node_currently_in_use, q)

                if nbr is None:  # getting killed by exponential time
                    self.roots.add(node_currently_in_use)
                    sinks.update(trajectory)
                    break
                else:
                    if nbr in sinks:  # Walked into a sink
                        sinks.update(trajectory)
                        break
                    else:
                        if nbr in trajectory:  # Just created a loop
                            pos = trajectory.index(nbr)
                            trajectory.append(nbr)
                            # We delete all vertices from the just created loop except for nbr, because we start
                            # a new random walk from there
                            active_nodes.update(trajectory[pos + 1:-1])
                            # only keep nodes that are not in the loop
                            for i in range(pos, len(trajectory) - 1):
                                self.edges[trajectory[i], trajectory[i + 1]]['hidden'] = True
                            trajectory = trajectory[:pos + 1]

                        else:
                            trajectory.append(nbr)
                            active_nodes.remove(nbr)
                node_currently_in_use = nbr
        if renumber_roots_after_finishing:
            self.number_the_nodes()

    def follow_arrow_if_exist_else_create_arrow_and_follow_or_get_killed(self, start, q):
        result = None
        for nbr in self[start]:
            if self.edges[start, nbr]['hidden'] == False:
                return nbr
        # if there is no edge to follow, create one or get killed
        # do we kill?
        if random.random() < q / (q + self.nodes[start]['weight']):
            return None  # intepreted as kill
        # else: create new edge
        nbr = self.choose_random_neighbor(start)
        self.edges[start, nbr]['hidden'] = False
        return nbr


def multiresolution(g: WilsonGraph, steps=5) -> (List[float], List[float], List[WilsonGraph]):
    """

    :param g: Graph that gets analyzed
    :param steps: number of steps
    :return: list of all downsampled graphs, list of all q_prime's, list of all q's
    """
    downsampled_graph = copy.deepcopy(g)
    q_list = []
    q_prime_list = []
    graph_list = []

    for i in range(steps):
        g, downsampled_graph, q, q_prime = downsampled_graph.one_step_in_multiresolution_scheme(
            name_of_graph=f'downsampling_step_{i + 1}')
        q_list.append(q)
        q_prime_list.append(q_prime)
        graph_list.append(copy.deepcopy(g))
    return q_list, q_prime_list, graph_list


def multi_reconstr(graph_list: List[WilsonGraph], q_prime_list: List[float], modify_list=False):
    if not modify_list:
        graph_list = copy.deepcopy(graph_list)
    graph_list[-1].set_non_root_values_to_zero()
    graph_list[-1].reconstruction_operator(q_prime_list[-1])
    for i in reversed(range(len(graph_list) - 1)):
        graph_list[i].set_all_values_to_zero()
        graph_list[i].copy_values_from_other_graph_wherever_they_exist(graph_list[i + 1])
        graph_list[i].reconstruction_operator(q_prime_list[i])
    return graph_list


def multi_resolution_and_reconstr(g: WilsonGraph, steps=5):
    g.show('g.html', color_roots=False)
    q_list, q_prime_list, graph_list = multiresolution(g, steps)
    # print(f'{q_list=}')
    # print(f'{q_prime_list=}')
    multi_reconstr(graph_list, q_prime_list)


def create_graph_from_matrix(mat, original_graph=None):
    g = WilsonGraph()
    if original_graph is None:
        raise Exception('Procedure not yet written')
    if original_graph is not None:
        for node in original_graph.roots:
            g.add_node(node)
            for attr in original_graph.nodes[node]:
                g.nodes[node][attr] = original_graph.nodes[node][attr]
        dok_mat = scipy.sparse.dok_matrix(mat)
        node_number_dict = {}
        for node in original_graph.roots:
            node_number_dict[original_graph.nodes[node]['node_number']] = node
        for e in dok_mat.keys():
            if e[0] != e[1]:
                g.add_edge(node_number_dict[e[0]], node_number_dict[e[1]], weight=dok_mat[e], hidden=True)
        # for x in original_graph.roots:
        #     for y in original_graph.roots:
        #         if x != y and mat[original_graph.nodes[x]['node_number'], original_graph.nodes[y]['node_number']] != 0:
        #             g.add_edge(x, y, weight=mat[
        #                 original_graph.nodes[x]['node_number'], original_graph.nodes[y]['node_number']], hidden=False)
    return g
