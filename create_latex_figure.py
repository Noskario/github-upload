def create_rec_without_detail():
    print(r'\begin{figure}')
    print(r'\centering')
    for q_prime in [.01, .06, .2, .3, .5, 1.5, 2.5, 4, 6.5, 9.5, 13, 18, 25, 50, 110, 230, 580, 1400, 6000, 15000]:
        print(r'\begin{subfigure}[b]{.22\textwidth}')
        print(r'\includegraphics[width=\textwidth]{square_graph_rec_without_detail_', end='')
        print(f'{q_prime=}', end='')
        print(r'}')
        print(r"\caption*{$q'=", q_prime, r"$}")
        print(r'\end{subfigure}')
    print(
        r"%\caption{Here you see that given the roots (sampled using parameter $q=12.345$), it is important to choose $q'$ accordingly. Remember what $q'$ is: The analysis operator is the expected value of a Markov chain and stopped at $q'$-exponential time. For smaller $q'$ the stopping time is bigger. So the analyzed signal is more washed out. So in order for a root vertex $x\in\Xq$ to have a analyzed value differing a lot from the analyzed signal}")
    print(
        r"\caption{Here you see the reconstructed signal only using representative vertices (i.e. the detail vertices were set zero before computing the reconstruction). You see that reconstruction is especially bad for small $q'$ on the representative vertices. This can be explained as follows: We search a signal whose reconstruction is the }")
    print(r'\label{fig:rec_without_detail_different_q_primes}')
    print(r'\end{figure}')


def create_analyzed():
    print(r'\begin{figure}')
    print(r'\centering')
    for q_prime in [.01, .06, .2, .3, .5, 1.5, 2.5, 4, 6.5, 9.5, 13, 18, 25, 50, 110, 230, 580, 1400, 6000, 15000]:
        print(r'\begin{subfigure}[b]{.22\textwidth}')
        print(r'\includegraphics[width=\textwidth]{square_graph_analyzed_', end='')
        print(f'{q_prime=}', end='')
        print(r'}')
        print(r"\caption*{$q'=", q_prime, r"$}")
        print(r'\end{subfigure}')
    print(r"\caption{Here you see the analzed signal. }")
    print(r'\label{fig:analyzed_different_q_primes}')
    print(r'\end{figure}')

def create_rec_only_detail():
    print(r'\begin{figure}')
    print(r'\centering')
    for q_prime in [.01, .06, .2, .3, .5, 1.5, 2.5, 4, 6.5, 9.5, 13, 18, 25, 50, 110, 230, 580, 1400, 6000, 15000]:
        print(r'\begin{subfigure}[b]{.22\textwidth}')
        print(r'\includegraphics[width=\textwidth]{square_graph_rec_only_detail_', end='')
        print(f'{q_prime=}', end='')
        print(r'}')
        print(r"\caption*{$q'=", q_prime, r"$}")
        print(r'\end{subfigure}')
    print(r"\caption{Here you see the reconstructed signal only using detail vertices. }")
    print(r'\label{fig:rec_only_detail_different_q_primes}')
    print(r'\end{figure}')

if __name__ == '__main__':
    create_rec_only_detail()
