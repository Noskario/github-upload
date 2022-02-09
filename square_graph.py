from signal_processing import SignalProcessingGraph
import random


def cutter(n, threshold):
    if n < threshold:
        return n
    else:
        return 0


class SquareSignalProcessingGraph(SignalProcessingGraph):
    def __init__(self, n: int, standardweights=True):
        super().__init__()
        zeilen = n
        spalten = n
        if standardweights:
            for i in range(zeilen):
                for j in range(spalten):
                    self.add_node(str(i) + ',' + str(j), x=70 * i, y=70 * j,
                                  value=10.1 * cutter(i, n / 2) ** 2 - 133. * j - 100)

            for i in range(zeilen - 1):
                for j in range(spalten - 1):
                    r = random.random()
                    self.add_edge(str(i) + ',' + str(j), str(i + 1) + ',' + str(j), weight=1)
                    self.add_edge(str(i + 1) + ',' + str(j), str(i) + ',' + str(j), weight=1)
                    r = random.random()
                    self.add_edge(str(i) + ',' + str(j), str(i) + ',' + str(j + 1), weight=1)
                    self.add_edge(str(i) + ',' + str(j + 1), str(i) + ',' + str(j), weight=1)

            for i in range(spalten - 1):
                self.add_edge(str(zeilen - 1) + ',' + str(i), str(zeilen - 1) + ',' + str(i + 1), weight=1)
                self.add_edge(str(zeilen - 1) + ',' + str(i + 1), str(zeilen - 1) + ',' + str(i), weight=1)

            for i in range(zeilen - 1):
                self.add_edge(str(i) + ',' + str(spalten - 1), str(i + 1) + ',' + str(spalten - 1), weight=1)
                self.add_edge(str(i + 1) + ',' + str(spalten - 1), str(i) + ',' + str(spalten - 1), weight=1)
        else:
            for i in range(zeilen):
                for j in range(spalten):
                    self.add_node(str(i) + ',' + str(j), x=70 * i, y=70 * j,
                                  value=10.1 * cutter(i, n / 2) ** 2 - 133. * j - 100)

            for i in range(zeilen - 1):
                for j in range(spalten - 1):
                    r = random.random()
                    self.add_edge(str(i) + ',' + str(j), str(i + 1) + ',' + str(j), weight=(i + 1) + (j + 1) * .1)
                    self.add_edge(str(i + 1) + ',' + str(j), str(i) + ',' + str(j), weight=i + 1 + (j + 1) * .1)
                    r = random.random()
                    self.add_edge(str(i) + ',' + str(j), str(i) + ',' + str(j + 1), weight=(i + 1) + (j + 1) * .1)
                    self.add_edge(str(i) + ',' + str(j + 1), str(i) + ',' + str(j), weight=i + 1 + (j + 1) * .1)

            for i in range(spalten - 1):
                self.add_edge(str(zeilen - 1) + ',' + str(i), str(zeilen - 1) + ',' + str(i + 1), weight=1)
                self.add_edge(str(zeilen - 1) + ',' + str(i + 1), str(zeilen - 1) + ',' + str(i), weight=1)

            for i in range(zeilen - 1):
                self.add_edge(str(i) + ',' + str(spalten - 1), str(i + 1) + ',' + str(spalten - 1), weight=1)
                self.add_edge(str(i + 1) + ',' + str(spalten - 1), str(i) + ',' + str(spalten - 1), weight=1)


class SquareSignalProcessingGraphConstantValues(SignalProcessingGraph):
    def __init__(self, n: int, standardweights=True):
        super().__init__()
        zeilen = n
        spalten = n
        v = 77
        if standardweights:
            for i in range(zeilen):
                for j in range(spalten):
                    self.add_node(str(i) + ',' + str(j), x=60 * i, y=70 * j, value=v)

            for i in range(zeilen - 1):
                for j in range(spalten - 1):
                    r = random.random()
                    self.add_edge(str(i) + ',' + str(j), str(i + 1) + ',' + str(j), weight=1)
                    self.add_edge(str(i + 1) + ',' + str(j), str(i) + ',' + str(j), weight=1)
                    r = random.random()
                    self.add_edge(str(i) + ',' + str(j), str(i) + ',' + str(j + 1), weight=1)
                    self.add_edge(str(i) + ',' + str(j + 1), str(i) + ',' + str(j), weight=1)

            for i in range(spalten - 1):
                self.add_edge(str(zeilen - 1) + ',' + str(i), str(zeilen - 1) + ',' + str(i + 1), weight=1)
                self.add_edge(str(zeilen - 1) + ',' + str(i + 1), str(zeilen - 1) + ',' + str(i), weight=1)

            for i in range(zeilen - 1):
                self.add_edge(str(i) + ',' + str(spalten - 1), str(i + 1) + ',' + str(spalten - 1), weight=1)
                self.add_edge(str(i + 1) + ',' + str(spalten - 1), str(i) + ',' + str(spalten - 1), weight=1)
        else:
            for i in range(zeilen):
                for j in range(spalten):
                    self.add_node(str(i) + ',' + str(j), x=60 * i, y=70 * j, value=v)

            for i in range(zeilen - 1):
                for j in range(spalten - 1):
                    r = random.random()
                    self.add_edge(str(i) + ',' + str(j), str(i + 1) + ',' + str(j), weight=(i + 1) + (j + 1) * .1)
                    self.add_edge(str(i + 1) + ',' + str(j), str(i) + ',' + str(j), weight=i + 1 + (j + 1) * .1)
                    r = random.random()
                    self.add_edge(str(i) + ',' + str(j), str(i) + ',' + str(j + 1), weight=(i + 1) + (j + 1) * .1)
                    self.add_edge(str(i) + ',' + str(j + 1), str(i) + ',' + str(j), weight=i + 1 + (j + 1) * .1)

            for i in range(spalten - 1):
                self.add_edge(str(zeilen - 1) + ',' + str(i), str(zeilen - 1) + ',' + str(i + 1), weight=1)
                self.add_edge(str(zeilen - 1) + ',' + str(i + 1), str(zeilen - 1) + ',' + str(i), weight=1)

            for i in range(zeilen - 1):
                self.add_edge(str(i) + ',' + str(spalten - 1), str(i + 1) + ',' + str(spalten - 1), weight=1)
                self.add_edge(str(i + 1) + ',' + str(spalten - 1), str(i) + ',' + str(spalten - 1), weight=1)
