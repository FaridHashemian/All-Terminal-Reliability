import logging
import itertools
import time
import networkx as nx
import random
import os
import subprocess
import shutil
import time
import numpy as np
from scipy import stats

def reliability_comp_enum(G, t):

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

    file_handler = logging.FileHandler('logs.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    edges = list(G.edges())
    y = 0
    # the first two loops create all the (n-1) combination of the edges in a graph which n represents number of nodes
    for L in range(len(list(G.nodes))-1, len(edges)+1):
        for edge in itertools.combinations(edges, L):
          # this condition ensures that a subset of a given graph with a set of edge combination is connected
            if len(list(G.edge_subgraph(list(edge)).nodes)) == len(list(G.nodes)) and nx.is_connected(G.edge_subgraph(list(edge))) == True:
                x = 1

                # started calculating the reliability
                for e in G.edges():
                    if e not in list(edge):
                        x = x * (1 - nx.get_edge_attributes(G, "reliability")[e])
                    else:
                        x = x * nx.get_edge_attributes(G, "reliability")[e]
                y += x
    logger.info('reliability of graph {} is {}'.format(t,y))
    time.sleep(0.1)
    return y


def reliability_comp_fbs(G:nx.Graph, 
                id: int,
                node_size:int)->float:
    
    #logging configuration
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    file_handler = logging.FileHandler('../log_files/logs_{}.log'.format(node_size))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


    os.mkdir('tmp_files_{}'.format(id)) if not os.path.isdir('tmp_files_{}'.format(id)) else None

    if not os.path.isfile('tmp_files_{}/reliability.dat'.format(id)) and \
        not os.path.isfile('tmp_files_{}/connections.dat'.format(id)) and \
        not os.path.isfile('tmp_files_{}/grid.dat'.format(id)):  
        rels = open('tmp_files_{}/reliability.dat'.format(id), 'w')
        cons = open('tmp_files_{}/connections.dat'.format(id), 'w')
        ter = open('tmp_files_{}/grid.dat'.format(id), 'w')
        nodes = G.number_of_nodes()
        for key , value in nx.get_edge_attributes(G, 'reliability').items():
            rels.write(f'{value} ')
            cons.write(f'{key[0]} {key[1]}\n')

        ter.write(f'{0} {nodes-1}\n')
        rels.close()
        cons.close()
        ter.close()
        #subprocess.run(["g++", "reliability.cpp", "-o", "reliability_{}".format(id)])
        
        cmd = ["./reliability", 
               "-allrel", 
               'tmp_files_{}/connections.dat'.format(id),
               'tmp_files_{}/grid.dat'.format(id),
               'tmp_files_{}/reliability.dat'.format(id)]
        start_time = time.time()
        result = subprocess.run(cmd, text=True, capture_output=True)
        total_time = time.time() - start_time
        #os.remove('./reliability_{}'.format(id))
        shutil.rmtree('tmp_files_{}'.format(id))
        rel = float((str(result).split(',')[-3][str(result).split(',')[-3].index('prob')+7:str(result).split(',')[-3].index('\\n')]))
        logger.info('reliability of graph id {} is {}. Computed in {}.'.format(id, rel, total_time))
    return rel, total_time


def confidence_interval(data, args):

    mean = np.mean(data)
    standard_error = stats.sem(data)
    degrees_of_freedom = len(data) - 1
    t_critical = stats.t.ppf((1 + args.confidence) / 2, degrees_of_freedom)
    margin_of_error = t_critical * standard_error
    
    lower_bound = mean - margin_of_error
    upper_bound = mean + margin_of_error
    
    return lower_bound, upper_bound


def reliability_comp_mc(G, args, id, node_size:int)->list:
  
    #logging configuration
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    file_handler = logging.FileHandler('../log_files/mc/logs_{}.log'.format(node_size))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    num_run = args.mc_run
    num_sample = 0
    samples = []
    edges = [e for e in G.edges]
    random.seed(args.seed)
    start_time = time.time()
    while num_sample < args.mc_scenario:
        u = [random.uniform(0, 1) for i in range(len(edges)*num_run)]
        u_av = [1-elem for elem in u]
        rel_total = 0
        for j in range(num_run):
            g = G.copy()
            for i in edges:
                if u[(j*len(edges))+edges.index(i)] > nx.get_edge_attributes(g, 'reliability')[i]:
                  g.remove_edge(i[0],i[1])
            if nx.is_connected(g):
                rel_total += 1
        samples.append(rel_total/num_run)
        num_sample += 1
    total_time = time.time() - start_time
        
    rel = np.mean(samples)
    rel_std = np.std(samples)
    rel_lower, rel_upper = confidence_interval(samples, args)
    logger.info('reliability of graph id {} is {}, the std is {}, the lb is {}, the ub is {}, and computed in {}.'.format(id, rel, rel_std, rel_lower, rel_upper, total_time))
    return rel, rel_std, rel_lower, rel_upper, total_time