import networkx as nx
import random
import os
import pandas as pd
import multiprocessing
import argparse
from reliability_comp import reliability_comp_fbs


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=1871, type=int, help="seed number")
    parser.add_argument("--n_graph", default=10, type=int, help="number of graphs")
    parser.add_argument("--n_node", default=8, type=int, help="number of nodes")
    parser.add_argument("--l_bound", default=0.3, type=float, help="edge addition lower bound")
    parser.add_argument("--u_bound", default=0.6, type=float, help="edge addition upper bound")
    parser.add_argument("--path", default='reliability_tdzdd/', type=str, help="path to the folder")
    args = parser.parse_args()
    return args



def create_graph(args):

    random.seed(args.seed)

    dataset = []
    number_of_graphs_created = 0

    while number_of_graphs_created < args.n_graph:

        N = args.n_node

        #reliabilty = [0.80, 0.85, 0.90, 0.95, 0.99]
        # Create an empty graph object
        g = nx.Graph()


        # Adding nodes
        g.add_nodes_from(range(0, N))


        # Add edges to the graph randomly.
        for i in g.nodes():
            for j in g.nodes():
                if (i < j):

                    # Take random number R.
                    R = random.random()

                    # Check if R is in the range [0.3, 0.56]
                    if (args.l_bound <= R <= args.u_bound):
                        g.add_edge(i, j)
                        nx.set_edge_attributes(g, {(i, j): {"reliability": random.uniform(0.5,1)}})


        if nx.is_connected(g) == True:
            dataset.append(g)
            number_of_graphs_created += 1
        else:
            continue
    print('Graphs created: ', number_of_graphs_created)
    return dataset

def comp_process(dataset, args):
    #os.chdir(os.path.expanduser('~'))
    os.chdir(args.path)
    #results = []
    with open ('../save_files/results_{}.csv'.format(args.n_node), 'w') as f:
        
        for i, t in zip(dataset, range(len(dataset))):
            rel, time = reliability_comp_fbs(i, t, args.n_node)
            #results.append()
            f.write('{};{};{}\n'.format(nx.to_dict_of_dicts(i), rel, time))

    f.close()

# def multi_process(dataset):
#     args = [(i,t) for i,t in zip(dataset, range(len(dataset)))]

#     with multiprocessing.Pool(processes=32) as pool:
#         results = pool.starmap(reliability, args)
#     return results

if __name__ == "__main__":
    args = get_args()
    dataset = create_graph(args)
    #print(multi_process(dataset))
    comp_process(dataset, args)