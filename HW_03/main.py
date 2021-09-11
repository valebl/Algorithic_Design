from math import inf
from copy import deepcopy
from binheap import binheap

##================================== AUXILIARY FUNCTIONS REQUIRED BY DIJKSTRA ===========================================================

def init_sssp(G,s):
    try:
        assert s < len(G)
    except:
        AssertionError(f'The specified source s must be a node of the graph.\nNodes are {range(G)} while {s} was given.')
    dist = [None] * len(G)
    pred = [None] * len(G)
    for k in range(len(G)):
        dist[k] = inf
        pred[k] = None
    dist[s] = 0
    pred[s] = s
    return dist, pred

def relax(H, u, v, w, dist, pred):
    if (dist[u] + w) < dist[v]:
        dist[v] = dist[u] + w
        pred[v] = u
        H.decrease_key(v, [v,dist[v]])

##================================== MORE AUXILIARY FUNCTIONS =========================================================================

def compute_infty(G):
    '''
    Function to compute the "inf" value in the given graph G
    '''
    infty = 0
    for i in range(len(G)):
        for j in range(len(G[i])):
            infty += G[i][j][1]
    return infty+1

def total_order_dijk(a,b):
    '''
    Total order used in the binheap implementation of dijkstra, based on the weight value
    '''
    return a[1] <= b[1]


def pretty_print_adj_list(l, name):
    '''
    Simple function that can be used to print the graph in its adjacency list representation
    '''
    s = f'{name}: \n'
    for i in range(len(l)):
        adj = l[i]
        s += f'{i}: ['
        for pair in adj:
            s += f'{pair[0], pair[1]}'
        s += ']\n'
    print(s)


def find_predecessors(G,n):
    '''
    Function that given in input a graph, G, and a node, n, returns in output a list
    containing all the predessors of n.
    '''
    pred = []
    for k in range(len(G)):
        if G[k] is not None:
            for v in G[k]:
                if v[0] == n:
                    pred.append(k)
    return pred


def remove_node(G,n, preds_of_n=None):
    '''
    Function that receives in input a graph, G, and a node, n. Additionally, the list of
    the predecessors of n can also be passed in input if already known, otherwise it is
    retrieved as first step. The algorithm removes the node from G, which means that it
    removes both the edges whith n as source and the ones with n as destination.
    '''
    if preds_of_n is None:
        preds_of_n = find_predecessors(G, n)
    for p in preds_of_n:
        G[p].remove([n,find_weight(G[p], n)])
    G[n] = []
    return(G)


def find_weight(list_of_pairs, value):
    '''
    Function that receives in input a list, corresponding to the adjacent list of a node.
    If there exists an edge with destination in the given node, the corresponding weight is returned
    '''
    for v, w in list_of_pairs:
        if v == value:
            return w
    raise RuntimeError(f"The node {value} is not present in the adjacency list given in input.")


def make_forward_backward_graphs(G):
    '''
    Function that takes in input a graph, G, and returns in output:
    G_up: which contains the edges (u,v) s.t. u < v
    G_down: which contains the edges (u,v) s.t. u > v, reversed
    '''
    G_f = [[] for i in range(len(G))]
    G_b = [[] for i in range(len(G))]
    for k in range(len(G)):
        for v,w in G[k]:
            if k < v:
                G_f[k].append([v,w])
            else:
                G_b[v].append([k,w])
    return G_f, G_b


##================================== DIJKSTRA ================================================================================

def dijkstra(G,s):
    '''
    The algorithm takes in input a graph G and a source node s and computes the sssp from
    s to all nodes in the graph. It returns in input a list containing the shortest distances (v.dist in the slides)
    and a list containing the predecessor of the destination nodes in the shortest path, (v.pred in the slides)
    '''
    dist, pred = init_sssp(G, s)
    queue = [[v,dist[v]] for v in range(len(G))] 
    H = binheap(queue,total_order=total_order_dijk)
    while not H.is_empty():
        u, _ = H.remove_minimum()
        for pair in G[u]: # v is the value and w is the weight
            relax(H, u, pair[0], pair[1], dist, pred)

    return dist, pred

##================================== SHORTCUTS ================================================================================

def shortcuts(G):
    '''
    The algorithm looks for the sssp of the type (i, n, w1) U (n, j, w2), where n is the target node, and adds the
    shortcut (i, j, w1+w2) if and only if removing the node n would result in increasing the shortest distance.
    The process is repeated by considering, in turn, all the nodes in the graph as target nodes.
    '''
    G_updated = deepcopy(G) # will store the graph enriched with the shortcuts
    G_overlay = deepcopy(G) # will store the overlay graphs

    for n in range(len(G)): # we repeat the algorithm for all the nodes of G
        preds_of_n = find_predecessors(G_overlay, n) # find all the predecessors of n
        adj_nodes = G_overlay[n]
        G_overlay = remove_node(G_overlay,n, preds_of_n)
        for p in preds_of_n:
            dist, _ = dijkstra(G_overlay, p) # run dijkstra to find the sssp with the predecessor p as source
            for adj in adj_nodes: # loop on the adjacent nodes of the taget node n
                adj_v = adj[0] # get the vertex from the (vertex, weight) pair
                if adj_v != p:
                    w1 = find_weight(G_updated[p],n) # weight of the incoming edge (node p, node n)
                    w2 = adj[1] # weight of the outcoming edge (node n, adj node)
                    if dist[adj_v] > w1 + w2: # if this conditions holds, it means that the sssp is
                                              # represented by (p, n, w1) U (n, adj_v, w2), so we add the shortcut
                        for i in range(len(G[p])):
                            if G_updated[p][i][0] == adj_v:
                                G_updated[p].remove(G_updated[p][i]) # we cannot have two edges between the same nodes
                                break
                        G_updated[p].append([adj_v,w1+w2]) # add the shortcut
                        G_overlay[p].append([adj_v,w1+w2])
                        print(f'Found shortest path: ({p},{n},{w1}) U ({n},{adj_v},{w2})' +
                        f' and added shortcut: ({p},{adj_v},{w1+w2})')
    return G_updated

##================================== BIDIRECTIONAL DIJKSTRA ================================================================================

def dijkstra_bidir(G,s,d):
    '''
    The algorithm takes in input a graph enriched with all its shortcuts, a source node s and a destination
    node d. It returns in putput the shortest distance between s and d.
    '''

    G_f, G_b = make_forward_backward_graphs(G) # derive the forward and backward graphs from G

    dist_f, pred_f = init_sssp(G_f, s) # distances for forward search
    queue = [[v,dist_f[v]] for v in range(len(G))]
    H_f = binheap(queue,total_order=total_order_dijk) # binheap for forward search

    dist_b, pred_b = init_sssp(G_b, d) # distances for backward search
    queue = [[v,dist_b[v]] for v in range(len(G))]
    H_b = binheap(queue,total_order=total_order_dijk) # binheap for backward search

    discovered_f = [] # list that will store the nodes discovered in the forward search
    discovered_b = [] # list that will store the nodes discovered in the backward search

    min_in_queue_b = inf # will store the minimum distance from s in the forward sense, considering the nodes still in the queue
    min_in_queue_f = inf # will store the minimum distance from d in the backward sense, considering the nodes still in the queue

    distance = inf # will store the minimum distance d(s,v) + d(v,d), where s=source, d=destination, v in G

    frontier_intersect = False # boolean that becomes true, when in forward a node already discovered in backward
                               # passes is discovered, and vice versa

    while not H_f.is_empty() and not H_b.is_empty():

        u_f, _ = H_f.remove_minimum() # remove the minimum from the queue
        u_b, _ = H_b.remove_minimum()

        discovered_f.append(u_f) # add to the list of discovered nodes
        discovered_b.append(u_b)

        for (v, w) in G_f[u_f]: # relax all the adjacent nodes
            relax(H_f, u_f, v, w, dist_f, pred_f)

        for (v, w) in G_b[u_b]:
                relax(H_b, u_b, v, w, dist_b, pred_b)

        if u_f in discovered_b or u_b in discovered_f: # if the condition holds, we still have to look for the shorter distance
            frontier_intersect = True

        if frontier_intersect == True: # this part of the code is to make sure the distance is the shortest one
            for x in H_f:
                x_v = x[0]
                if dist_f[x_v] < min_in_queue_f: # derive the smallest distance among the nodes in the queue
                    min_in_queue_f = dist_f[x_v]

            for x in H_b:
                x_v = x[0]
                if dist_b[x_v] < min_in_queue_b: 
                    min_in_queue_b = dist_f[x_v]      

            for x in range(len(dist_f)):
                if dist_f[x] + dist_b[x] < distance: # derive the smallest distance among the candidate distances
                    distance = dist_f[x] + dist_b[x]
        
            if (distance <= min(min_in_queue_f, min_in_queue_b)):
               return distance

    return distance


##================================== TESTING ======================================================================================

if __name__ == '__main__':

    G0 = [
    [[4,1],[5,1]],  # 0
    [[2,2]],        # 1
    [[1,1],[3,3]],  # 2
    [[6,1],[7,3]],  # 3
    [[0,3],[5,1]],  # 4
    [[7,1]],        # 5
    [[7,1]],        # 6
    [[0,1],[6,1]]   # 7
    ]

    G1 = [
    [],                             # 0
    [[0,4]],                        # 1
    [[1,4],[3,13],[5,1],[6,3]],     # 2
    [[0,3]],                        # 3
    [[1,10],[2,1],[6,2]],           # 4
    [[3,3],[4,5]],                  # 5
    [],                             # 6
    []                              # 7
    ]
    
    ## PARAMETERS

    G = G0  # graph
    s = 0   # source
    d = 6   # destination

    ## DIJKSTRA

    dist, pred = dijkstra(G,s)

    print(f'\n- Dijkstra Algorithm -')
    print(f'Single Source Shortest Paths of (G, {s}):')
    print(f'Distances: {dist}')
    print(f'Predecessors: {pred}')

    ## SHORTCUTS

    print('\n- Shortcuts Algorithm -')
    Gs = shortcuts(G)
    #pretty_print_adj_list(Gs,"\nGraph decorated with shortcuts")

    ## BIDIRECTIONAL DIJKSTRA

    print('\n- Bidirectional Dijkstra Algorithm -')
    # min_dist = [dijkstra_bidir(Gs, s, d) for d in range(len(G))] # cycle over all destination to check the algorithm results
    # print(f"{min_dist}\n")
    min_dist = dijkstra_bidir(Gs, s, d)
    print(f'The shortest distance from {s} to {d} is {min_dist}\n')


