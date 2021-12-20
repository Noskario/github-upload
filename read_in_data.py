import pickle

import networkx as nx
import matplotlib.pyplot as plt
import pyvis
#import community

with open('G_fb.pickle', 'rb') as file:
    G_fb = pickle.load(file)
with open('parts.pickle', 'rb') as file:
    parts = pickle.load(file)
with open('spring_pos.pickle', 'rb') as file:
    spring_pos = pickle.load(file)
# G_fb=nx.read_edgelist('facebook_combined.txt',create_using=nx.Graph(),nodetype=int)
#print(nx.info(G_fb))

#parts = community.best_partition(G_fb)

values = [parts.get(node) for node in G_fb.nodes()]
#print(values)
#print(parts)
#print(G_fb)
##print(spring_pos)
#for n in G_fb.nodes:
#    print(spring_pos[n])
#with open('parts.pickle','wb') as file:
#   pickle.dump(parts,file)
# with open('G_fb.pickle','wb') as file:
#    pickle.dump(G_fb,file)
# with open('spring_pos.pickle','wb') as file:
#    pickle.dump(spring_pos,file)




#plt.axis('off')
#nx.draw_networkx(G_fb, pos=spring_pos, with_labels=False, node_size=20, node_color=values)
#plt.show()

