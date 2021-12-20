import example_graphs,non_reversible_graphs

trajectory=[1,2,3,4,5,3]
print(f'{trajectory=}')
nbr = 3
pos = trajectory.index(nbr)
print(f'{trajectory[pos + 1:]}')
for i in range(pos, len(trajectory) - 1):
    print(f'{(trajectory[i], trajectory[i + 1])=}')
trajectory = trajectory[:pos + 1]
print(f'{trajectory=}')


g=non_reversible_graphs.Halfer(10)
for e in g.edges:
    print(e,g.edges[e])

print(g[1])