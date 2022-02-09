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


class SignalProcessingGraph(nx.DiGraph):
    def __init__(self):
        super().__init__()
        self.weights_are_set = False
        self.roots = set()

    def set_weights(self):
        """
        Computes all vertex degrees. Also computes the maximal vertex degree
        :return: Nothing. Vertex degree is stored in the 'weight' attribute of all nodes, maximal vertex degree in
         the 'alpha' attribute of the graph
        """
        weightlist = []
        for v in self.nodes:
            w = self.get_node_weight(v)
            self.nodes[v]['weight'] = w
            weightlist.append(w)
        alpha = max(weightlist)
        self.graph['alpha'] = alpha
        self.weights_are_set = True

    def LERW(self, start, sinks, q=0):  # loop erased random walk
        """
        Samples a loop erased random walk started in some vertex with given sinks and q. I is not needed if you use the
         Diaconis Fulton stack representation to sample random spanning forests
        :param start: vertex where to start the random walks
        :param sinks: sinks for the random walk
        :param q: parameter for exponential killing time
        :return: trajectory of random walk
        """
        activenodeid = start
        # Start skeleton process (sped up by exchangin alpha by w(x))
        trajectory = [start]
        while True:
            r = random.random()
            if r < q / (self.nodes[activenodeid]['weight'] + q):  # LERW killed with exp time
                self.roots.add(activenodeid)
                return trajectory
            chosennodeid = activenodeid
            temp = q
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

    def wilson(self, q: float, roots: set = None, ):
        """
        Applies Wilson's algorithm on the graph. Samples a random spanning forest.
        :param q: exponential killing time
        :param roots: a priori sinks
        :return: nothing, but the 'hidden'-attribute of edges gets changed
        """
        self.stack_version_of_wilson(q, renumber_roots_after_finishing=True, start_from_scratch=True, roots=roots)

    def show(self, graphname, color_roots=True, minimum=None, maximum=None):
        """
        Creates an interactive html-image of the graph and opens it
        :param graphname: has to have the format "name_of_graph.html"
        :param color_roots: If true then roots are shown darker than non-roots. If false colors are derived from value on the vertices
        :param maximum: maximum value for color scheme if color_roots=False
        :param minimum: minimum value for color scheme if color_roots=False
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
        # h.show() writes a html-file and tells a browser to open it. In case you change the graph and use
        # show() it again the browser might mix up the old and the new html-files so you see wrong results.
        # So the sleep statement is supposed to give the browser the time to load the html-file completely
        # before it might changed again by another show().
        time.sleep(.2)

    def create_picture(self, filename='graph.pdf', color_using_roots=True, print_values=False, show_immediately=False,
                       colorbar=True, colorbar_for_edges=False, colorbar_position=None, **kwargs):
        """
        Creates a picture of the graph. Allows you to draw arrows for edges.
        """
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
                plt.colorbar(sm, location=colorbar_position)
            if colorbar_for_edges:
                if 'edge_cmap' not in kwargs:
                    kwargs['edge_cmap'] = plt.cm.viridis_r
                edge_cmap = kwargs['edge_cmap']
                if 'edgelist' not in kwargs:
                    kwargs['edgelist'] = self.edges
                if 'edge_color' not in kwargs:
                    kwargs['edge_color'] = [self.edges[e]['weight'] for e in kwargs['edgelist']]
                edge_color = kwargs['edge_color']
                edge_vmin = min(edge_color)
                edge_vmax = max(edge_color)
                sm = plt.cm.ScalarMappable(cmap=edge_cmap, norm=plt.Normalize(edge_vmin, edge_vmax))

                plt.colorbar(sm, location=colorbar_position)

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

    def create_picture_only_vertices(self, filename='graph.pdf', show_immediately=False, title=None, ticks=None,
                                     colorbar=True, colorbar_position=None,
                                     **kwargs):
        """
        Draws the nodes (ignores edges). Is appropiate for big graphs where drawing all edges is too slow/clumsy.
        """
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
            plt.colorbar(ticks=ticks, location=colorbar_position)
        plt.axis('off')
        plt.box(False)
        plt.savefig(filename, bbox_inches='tight')
        if show_immediately:
            plt.show()

    def get_node_weight(self, node) -> float:
        """
        Computes vertex degree of vertex
        
        :param node: 
        :return: vertex degree
        """
        nodeweight = 0
        for e in self.edges(node):
            nodeweight += self.edges[e]['weight']
        return nodeweight

    def choose_random_neighbor(self, node):
        """
        Choose random neighbor according to weights
        :param node:
        :return: id of chosen node
        """
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
        """
        Changes color-attribute of roots to darkred, all other vertices are colored palevioletred
        :return:
        """
        for v in self.nodes:
            self.nodes[v]['color'] = 'palevioletred'
        for v in self.roots:
            self.nodes[v]['color'] = 'darkred'

    def get_maximum_value(self) -> float:
        """
        Computes maximum value among all vertices
        :return: maximum vertex value
        """
        l = []
        for v in self.nodes:
            l.append(self.nodes[v]['value'])
        return max(l)

    def get_minimum_value(self) -> float:
        """
        Computes minimum value among all vertices
        :return: minimum vertex value
        """
        l = []
        for v in self.nodes:
            l.append(self.nodes[v]['value'])
        return min(l)

    def color_using_values(self, maximum=None, minimum=None):
        """
        Colors all vertices according to their values using coolwarm colorscheme. Maximum and minimum will be changed
         in order to ensure that maximum==-minimum and that both have modulus of at least one.

        """
        if maximum is None:
            maximum = self.get_maximum_value()
        if minimum is None:
            minimum = self.get_minimum_value()
        # begin: achieve that zero gets white/neutral color
        minimum = min(minimum, -1)
        maximum = max(maximum, 1)
        minimum = min(minimum, -maximum)
        maximum = max(maximum, -minimum)
        # end
        norm = matplotlib.colors.Normalize(vmin=minimum, vmax=maximum)
        cmap = matplotlib.cm.coolwarm
        for v in self.nodes:
            self.nodes[v]['color'] = matplotlib.colors.to_hex(cmap(norm(self.nodes[v]['value'])))

    def _process_activated_node(self, x, q2):
        """
        Process activated node when we couple random forests and compute forest with new parameter q2<q_old
        :param x: activated node
        :param q2: new parameter
        :return: number of reprocessed vertices
        """
        number_of_reprocessed_nodes = 0
        chosen_nbr = self.choose_random_neighbor(x)
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
                # In this case we have to remove the loop and reprocess the activated tree
                tree_of_x = self._get_tree_of_root(x)
                number_of_reprocessed_nodes += len(tree_of_x)
                self.roots.remove(x)
                for i in range(len(trajectory) - 1):
                    self.edges[trajectory[i], trajectory[i + 1]]['hidden'] = True
                self.stack_version_of_wilson(q=q2, active_nodes=tree_of_x)
                break
            elif y in self.roots:  # Another root
                number_of_reprocessed_nodes += 1
                self.edges[x, chosen_nbr]['hidden'] = False
                self.roots.remove(x)
                break
        return number_of_reprocessed_nodes

    def _coupled_q_next(self, q) -> float:
        """
        Compute new tree in coupling process
        :param q: q-value of which we have a sample given
        :return: new q-value, for which a new sample is produced
        """
        r = random.random()
        m = len(self.roots)
        alpha = self.graph['alpha']
        q_next = alpha / (r ** (-1 / m) * ((alpha + q) / q) - 1)
        x = random.choice(list(self.roots))
        self._process_activated_node(x, q_next)
        return q_next

    def coupled_cont(self, qmax, q_stop) -> float:
        """
        Runs a whole coupling process starting at qmax and stopped at q-stop. Does not store any results.
         There is no real use-case except for testing purposes.
        :param qmax: q where to start the coupling process
        :param q_stop: lower threshold for q to stop coupling
        :return: the value of q when we get the first time below q_stop
        """
        q = qmax
        self.stack_version_of_wilson(q, renumber_roots_after_finishing=False, start_from_scratch=True)
        while q > q_stop:
            q = self._coupled_q_next(q)
        self.number_the_nodes()
        return q

    def get_predecessors_of_node(self, node) -> set:
        """
        finds all vertices that have uncovered edges pointing to the given vertex
        :param node:
        :return: set of all vertices that have a visible edge pointing to the given vertex
        """
        s = set()
        for nbr in self.predecessors(node):
            if self.edges[nbr, node]['hidden'] == False:
                s.add(nbr)
        return s

    def _get_tree_of_root(self, node) -> set:
        """
        returns the set of all nodes with uncovered paths to given node
        :param node:
        :return: the set of all nodes with uncovered paths to given node
        """
        counter = 0
        explorer_set = set()
        explorer_set.add(node)
        explored_set = set()
        aux_set = set()
        while explorer_set:
            counter += 1
            if counter > len(self.nodes):
                # In this case there is a loop somewhere. There is no way it could take longer to
                # collect all vertices of a tree than  there are nodes in the graph
                for e in self.edges:
                    print(e, self.edges[e])
                print(f'{self.roots=}')
                print(f'{explorer_set=}')
                self.show('error.html')
                raise Exception(f"The set of vertices that eventually lead to vertex {node} contains a loop.")
            for x in explorer_set:
                aux_set.update(self.get_predecessors_of_node(x))
                explored_set.add(x)
            explorer_set = aux_set
            aux_set = set()
        return explored_set

    def number_the_nodes(self):
        """
        Numbers consecutively all vertices. Let m the number of roots, n the number of vertices. Then roots
         will get numbers 0..m-1, non roots m..n-1
        :return: number of all nodes
        """
        i = 0
        for node in self.roots:
            self.nodes[node]['node_number'] = i
            i += 1
        for node in set(self.nodes) - self.roots:
            self.nodes[node]['node_number'] = i
            i += 1
        return i

    def create_Laplacian(self, selected_rows='all', selected_cols='all') -> scipy.sparse.csc_matrix:
        """
        Creates sparse matrix that is the Laplacian of the graph restricted to given rows and columns
        :param selected_rows: must be in ['all', 'roots', 'non-roots']
        :param selected_cols: must be in ['all', 'roots', 'non-roots']
        :return: (restricted) Laplacian
        """
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

    def compute_Schur_complement(self, make_sparse=True) -> scipy.sparse.csc.csc_matrix:
        """
        Computes the Schur complement (sparsified, for performance improvements)
        :param make_sparse: if True, only keep important edges
        :return:
        """
        mat_qq = self.create_Laplacian(selected_rows='roots', selected_cols='roots')
        mat_qu = self.create_Laplacian(selected_rows='roots', selected_cols='non-roots')
        mat_uq = self.create_Laplacian(selected_rows='non-roots', selected_cols='roots')
        mat_uu = self.create_Laplacian(selected_rows='non-roots', selected_cols='non-roots')

        mat_uu_inv = scipy.sparse.linalg.inv(mat_uu)
        if mat_uu_inv.shape == (1,):  # Make sure it stays a matrix and does not get compressed to scalar
            mat_uu_inv = scipy.sparse.csc_matrix(mat_uu_inv.reshape((1, 1)))

        result = mat_qq - mat_qu @ mat_uu_inv @ mat_uq
        if make_sparse:
            # Now we have to make it sparser if we want to have reasonable runtimes.
            # Without any mathematical rigour I will do as follows:
            # I compute the sum of all edges and set all elements to zero that have a weight smaller than 10e-10 of that
            arr = np.array(result.toarray())
            weight_of_all_edges = -arr.trace()
            arr = np.where(arr < 10e-10 * weight_of_all_edges, 0, arr)
            # Now we have to correct the values on the diagonal
            # compute row-sum
            row_sum = arr.sum(1)
            np.fill_diagonal(arr, -row_sum)
            result = scipy.sparse.csc_matrix(arr)

        return result

    def convert_values_to_np_array(self):
        """
        computes numpy array with all the values. The order of the vertices is defined by their 'node_number'
        :return: numpy-array of shape (n,) filled with the values of the vertices of the graph
        """
        n = len(self.nodes)
        value_vector = np.zeros(n)
        for node in self.nodes:
            value_vector[self.nodes[node]['node_number']] = self.nodes[node]['value']
        return value_vector

    def analysis_operator(self, q):
        """
        analyzes the signal (changes the values in-place)
        :param q: parameter for exponential killing time
        """
        n = len(self.nodes)
        m = len(self.roots)
        L = self.create_Laplacian()
        A = scipy.sparse.identity(n, format='csc') - L / q
        f = self.convert_values_to_np_array()
        f_analysed = scipy.sparse.linalg.spsolve(A, f)
        f_analysed[m:] -= f[m:]
        for node in self.nodes:
            self.nodes[node]['value'] = f_analysed[self.nodes[node]['node_number']]

    def reconstruction_operator(self, q, L_Schur=None):
        """
        reconstructs the signal (changes the values in-place)
        :param q: parameter for exponential killing time
        :param L_Schur: Schur complement. If not provided it will be computed.
        """
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
        f[:m] += f_bar - 1 / q * L_Schur.dot(f_bar)
        f[:m] += L_bar_breve.dot(scipy.sparse.linalg.spsolve(-L_breve_breve, f_breve))
        f[m:] += scipy.sparse.linalg.spsolve(-L_breve_breve, L_breve_bar.dot(f_bar))
        f[m:] += q * scipy.sparse.linalg.spsolve(L_breve_breve, f_breve) - f_breve
        for node in self.nodes:
            self.nodes[node]['value'] = f[self.nodes[node]['node_number']]

    def reconstruction_operator_without_detail_nodes(self, q, L_Schur=None):
        """
        reconstructs the signal but only taking into account the approxmative vertices (changes the values in-place)
        :param q: parameter for exponential killing time
        :param L_Schur: Schur complement. If not provided it will be computed.
        """
        f_ana = self.convert_values_to_np_array()
        m = len(self.roots)
        n = len(self.nodes)
        f_bar = f_ana[:m]
        # f_breve = f_ana[m:]
        L_breve_breve = self.create_Laplacian(selected_rows='non-roots', selected_cols='non-roots')
        # L_bar_breve = self.create_Laplacian(selected_rows='roots', selected_cols='non-roots')
        L_breve_bar = self.create_Laplacian(selected_rows='non-roots', selected_cols='roots')
        if L_Schur is None:
            L_Schur = self.compute_Schur_complement()
        f = np.zeros(n)
        f[:m] += f_bar - 1 / q * L_Schur.dot(f_bar)
        # f[:m] += L_bar_breve.dot(scipy.sparse.linalg.spsolve(-L_breve_breve, f_breve))
        f[m:] += scipy.sparse.linalg.spsolve(-L_breve_breve, L_breve_bar.dot(f_bar))
        # f[m:] += q * scipy.sparse.linalg.spsolve(L_breve_breve, f_breve) - f_breve
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

    def _find_q(self, theta1, theta2) -> float:
        """
        Computes a good value of q for downsampling
        :param theta1: helps to adjust the lower bound for q
        :param theta2: helps to adjust the upper bound for q
        :return: minimizer_q: the best q
        """
        qmax, q_stop = theta2 * self.graph['alpha'], theta1 * self.graph['alpha']
        q = qmax
        minimizer_q = qmax
        self.stack_version_of_wilson(qmax, renumber_roots_after_finishing=False, start_from_scratch=True)
        temp = self.alpha_bar_approx(q) * self.gamma_inverse_approx(q)
        while q > q_stop:
            q = self._coupled_q_next(q)
            if self.alpha_bar_approx(q) * self.gamma_inverse_approx(q) < temp:
                minimizer_q = q
                temp = self.alpha_bar_approx(q) * self.gamma_inverse_approx(q)
        self.number_the_nodes()
        return minimizer_q

    def _find_q_prime(self, q, L_Schur=None):
        """
        Computes downsampled graph (using paramter q) and computes a good value of q_prime for siganl processing on
         this downsampled graph
        :param q: parameter for downsampling
        :param L_Schur: Schur complement (will be computed if not provided)
        :return (q_prime,downsampled_graph): q_prime and the downsampled graph
        """
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
            except KeyError:
                pass

    def one_step_in_multiresolution_scheme(self, theta1=.2, theta2=1, name_of_graph='downsampled',
                                           show_created_graph=False):
        """
        Computes a good good parameter q for downsampling and then the downsampled graph. Computes a good parameter
         q_prime for sharpness of the analyzing step.
        :param theta1: helps to adjust the lower bound for q
        :param theta2: helps to adjust the upper bound for q
        :param name_of_graph: name of html if the downdampled graph is shown
        :param show_created_graph: if True the downsampled graph is shown
        :return:
        """
        if not self.weights_are_set:
            self.set_weights()
        q = self._find_q(theta1, theta2)
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
        Stack version of Wilson's algorithm
        :param q: parameter of exponential killing time
        :param active_nodes: Set of nodes that are still active. Also 'all' is accepted.
        :param renumber_roots_after_finishing:
        :param start_from_scratch: set True if you want to delete all uncovered edges/roots and start all over
        :param roots: in case you start from scratch give the a priori roots here. Must be a set. Argument is ignored
         if start_from_scratch=True
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
        """
        Goes one step in the stack version of Wilson's algorithm
        :param start: vertex where we are right now
        :param q: parameter of exponential killing time
        :return: next vertex where to go or None if we get killed in the current vertex
        """
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


def multiresolution(g: SignalProcessingGraph, steps=5) -> (List[float], List[float], List[SignalProcessingGraph]):
    """
    Computes all downsamplings and analyzes them according to appropiate parameters.
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


def multi_reconstr(graph_list: List[SignalProcessingGraph], q_prime_list: List[float], modify_graph_list=False):
    """
    Reconstructs the result of the multiresolution-function (ignoring the detail vertices)
    :param graph_list: list of downsampled graphs with analyzed signals
    :param q_prime_list: list of parameters used for the analysis operator
    :param modify_graph_list: whether to keep the list of analyzed graphs or to overwrite their signals
    :return: list of graphs with recontructed signal
    """
    if not modify_graph_list:
        graph_list = copy.deepcopy(graph_list)
    graph_list[-1].reconstruction_operator_without_detail_nodes(q_prime_list[-1])
    for i in reversed(range(len(graph_list) - 1)):
        graph_list[i].copy_values_from_other_graph_wherever_they_exist(graph_list[i + 1])
        graph_list[i].reconstruction_operator_without_detail_nodes(q_prime_list[i])
    return graph_list


def multi_resolution_and_reconstr(g: SignalProcessingGraph, steps=5):
    """
    Computes the multi resolution scheme of a graphs and reconstructs it
    :param g: Graph that gets analyzed
    :param steps: number of steps
    """
    g.show('g.html', color_roots=False)
    q_list, q_prime_list, graph_list = multiresolution(g, steps)
    multi_reconstr(graph_list, q_prime_list)


def create_graph_from_matrix(mat, original_graph=None)->SignalProcessingGraph:
    """
    Create SignalProcessingGraph containing only the roots of the original graph, creating edges according
     to a matrix of edgeweights
    :param mat: expected to be the Schur-complement of the original_graph
    :param original_graph: graph of which the downsampled graph is constructed
    :return: downsampled graph
    """
    g = SignalProcessingGraph()
    if original_graph is None:
        raise NotImplementedError('Procedure not yet written')
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
    return g
