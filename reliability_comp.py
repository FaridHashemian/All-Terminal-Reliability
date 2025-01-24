import logging
import itertools
import time
import networkx as nx

def reliability(G, t):

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

    file_handler = logging.FileHandler('logs.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    #logger.info('graph {} number {}'.format(G,t))
    edges = list(G.edges())
    y = 0
    # the first two loops create all the (n-1) combination of the edges in a graph which n represents number of nodes
    for L in range(len(list(G.nodes))-1, len(edges)+1):
        for edge in itertools.combinations(edges, L):
          # this condition ensures that a subset of a given graph with a set of edge combination is connected or not
            if len(list(G.edge_subgraph(list(edge)).nodes)) == len(list(G.nodes)) and nx.is_connected(G.edge_subgraph(list(edge))) == True:
                x = 1

                # from here, I started calculating the reliability
                for e in G.edges():
                    if e not in list(edge):
                        x = x * (1 - nx.get_edge_attributes(G, "reliability")[e])
                    else:
                        x = x * nx.get_edge_attributes(G, "reliability")[e]
                y += x
    logger.info('reliability of graph {} is {}'.format(t,y))
    time.sleep(0.1)
    return y