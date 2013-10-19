
import math
import itertools

import numpy as np

from setiptah.basic_graph.mygraph import mygraph
from setiptah.basic_graph.dijkstra import Dijkstra
from setiptah.basic_graph.toposort import toposort


class line :
    def __init__(self, slope ) :
        self.m = slope
    def __call__(self, x ) :
        return self.m * x


""" Utility Algorithms """

"""
each algorithm can either populate an empty data structure, or can incrementally 
update one
"""
    
    
def ResidualGraph( rgraph, flow, capacity, Delta, network, edge=None ) :
    if edge is None :
        iter = network.edges()
    else :
        iter = [ edge ]
        
    for e in iter :
        x = flow.get( e, 0. )
        u = capacity.get( e, np.Inf )
        assert x >= 0. and x <= u
        
        i,j = network.endpoints(e)
        
        ee = (e,+1)
        if rgraph.has_edge( ee ) : rgraph.remove_edge( ee )
        if x + Delta <= u : rgraph.add_edge( ee, i, j )
        
        ee = (e,-1)
        if rgraph.has_edge( ee ) : rgraph.remove_edge( ee )
        if x >= Delta : rgraph.add_edge( ee, j, i )



def LinearizeCost( lincost, cost, flow, Delta, network, edge=None ) :
    if edge is None :
        iter = network.edges()
    else :
        iter = [ edge ]
    
    for e in iter :
        x = flow.get(e, 0. )
        cc = cost.get( e, line(0.) )
        for dir in [ +1, -1 ] :
            lincost[(e,dir)] = float( cc( x + dir * Delta ) - cc(x) ) / Delta
            
            
def ReducedCost( rcost, lincost, potential, network, edge=None ) :
    if edge is None :
        iter = network.edges()
    else :
        iter = [ edge ]
        
    # if node is not in potential, it is assumed to have zero potential
    for e in iter :
        i,j = network.endpoints(e)
        #rcost[(e,+1)] = lincost.get( (e,+1), 0. ) + potential.get(j,0.) - potential.get(i,0.)
        #rcost[(e,-1)] = lincost.get( (e,-1), 0. ) + potential.get(i,0.) - potential.get(j,0.)
        rcost[(e,+1)] = lincost[(e,+1)] + potential.get(j,0.) - potential.get(i,0.)
        rcost[(e,-1)] = lincost[(e,-1)] + potential.get(i,0.) - potential.get(j,0.)
            
            
            
# TODO: Incrementize
def Excess( flow, graph, supply ) :
    excess = {}
    for i in graph.nodes() :
        excess[i] = supply.get(i, 0. )
        
        for e in graph.W[i] :   # edges in
            excess[i] += flow.get(e, 0. )
        for e in graph.V[i] :
            excess[i] -= flow.get(e, 0. )
            
    return excess



            
            

""" Convex Cost Flow Algorithm """

def MinConvexCostFlow( network, capacity, supply, cost, epsilon=None ) :
    """
    see "Fragile" version;
    this wrapper adds robustness:
    pre-processes to ensure strong connectivity of *any* Delta-residual graph;
    edge weights should be prohibitively expensive
    """
    # pre-process
    REGULAR = ':'
    AUGMENTING = 'AUG'
    
    network_aug = mygraph()
    capacity_rename = {}
    cost_aug = {}
    
    for e in network.edges() :
        i,j = network.endpoints(e)
        newedge = (REGULAR,e)
        
        network_aug.add_edge( newedge, i, j )
        if e in capacity : capacity_rename[ newedge ] = capacity[e]
        if e in cost : cost_aug[ newedge ] = cost[e]
        
    # add a directed cycles, with prohibitive cost
    U = sum([ b for b in supply.values() if b > 0. ])
    # since costs are convex, flow cannot have cost greater than M
    M = sum([ c(U) for c in cost.values() ])
    prohibit = line(M)
    
    NODES = network.nodes()
    edgegen = itertools.count()
    for i,j in zip( NODES, NODES[1:] + NODES[:1] ) :
        frwd = (AUGMENTING, edgegen.next() )
        network_aug.add_edge( frwd, i, j )
        cost_aug[frwd] = prohibit
    
    # run the "fragile" version
    flow = FragileMCCF( network_aug, capacity_rename, supply, cost_aug, epsilon )
    
    # prepare output --- perhaps do some feasibility checking in the future
    res = { e : x for (type,e), x in flow.iteritems() if type == REGULAR }
    return res
    #
    
    
    
def FragileMCCF( network, capacity_in, supply, cost, epsilon=None ) :
    """
    network is a mygraph (above)
    capacity is a dictionary from E -> real capacities
    supply is a dictionary from E -> real supplies
    cost is a dictionary from E -> lambda functions of convex cost edge costs
    
    1. Assumes supply is conservative (sum to zero).
    2. Assumes every Delta-residual graph is strongly connected,
    i.e., there exists a path with inf capacity b/w any two nodes;
    """
    if epsilon is None : epsilon = 1
    
    # initialize algorithm data
    rgraph = mygraph()
    lincost = {}
    redcost = {}
    
    """ ALGORITHM """
    U = sum([ b for b in supply.values() if b > 0. ])
    print 'total supply: %f' % U
    
    # trimming infinite capacities to U allows negative initial slopes 
    # the initial flow may not be Delta-optimality at the beginning of Stage One,
    # but achieves Delta-optimality by the end, by saturating any negative cost edges.
    # most treatments fail to consider negative initial slope, which is totally possible... 
    capacity = {}
    for e in network.edges() :
        capacity[e] = min( U, capacity_in.get( e, np.Inf ) )
        
        
    temp = math.floor( math.log(U,2) )
    Delta = 2.**temp
    print 'Delta: %d' % Delta
    
    
    flow = { e : 0. for e in network.edges() }
    potential = { i : 0. for i in network.nodes() }
    
    while Delta >= epsilon :
        print '\nnew phase: Delta=%f' % Delta
        
        # Delta is fresh, so we need to [re-] linearize the costs and compute residual graph 
        LinearizeCost( lincost, cost, flow, Delta, network )
        ReducedCost( redcost, lincost, potential, network )
        ResidualGraph( rgraph, flow, capacity, Delta, network )
        #
        cert = { re : c for (re,c) in redcost.iteritems() if re in rgraph.edges() }
        print 'reduced costs on res. graph, phase init: %s' % repr( cert )
        
        """ Stage 1. """
        # for every arc (i,j) in the residual network G(x)
        for resedge in rgraph.edges() :
            e,dir = resedge
            # theory says, we only need to do this at most once per edge...
            # wouldn't want to question theory
            # ... keep an eye out for a flip-flop; in theory, shouldn't happen
            if redcost[resedge] < 0. :
                print 'correcting negative red. cost on resedge %s: %f' % ( resedge, redcost[resedge] )
                
                # no augment, just saturate!
                flow[e] += dir * Delta
                #print 'flow correction: %s' % repr( flow )
                
                LinearizeCost( lincost, cost, flow, Delta, network, edge=e )
                ResidualGraph( rgraph, flow, capacity, Delta, network, edge=e )
                ReducedCost( redcost, lincost, potential, network, edge=e )
                
        # at end of each stage, verify the optimality certificate (should be empty every time)
        CERT = { re : c for (re,c) in redcost.iteritems() if re in rgraph.edges() and c < 0. }
        print 'certificate, end stage ONE: %s' % repr( CERT )
        if len( CERT ) > 0 : print "STAGE ONE CERTIFICATE CORRUPT!"
        # am considering removing this assertion, but leaving the stage two one
        # could be running into problems where the functional form is defined beyond saturation bounds
        assert len( CERT ) <= 0
        
                
        """ Stage 2. """
        # while there are imbalanced nodes
        while True :
            print 'flow: %s' % repr( flow )
            
            excess = Excess( flow, network, supply )        # last function that needs to be increment-ized
            print 'excess: %s' % repr(excess)
            
            SS = [ i for i,ex in excess.iteritems() if ex >= Delta ]
            TT = [ i for i,ex in excess.iteritems() if ex <= -Delta ]
            print 'surplus nodes: %s' % repr( SS )
            print 'deficit nodes: %s' % repr( TT )
            if len( SS ) <= 0 or len( TT ) <= 0 : break
            
            s = SS[0] ; t = TT[0]
            print 'shall augment %s to %s' % ( repr(s), repr(t) )
            
            print 'potentials: %s' % repr( potential )
            cert = { re : c for (re,c) in redcost.iteritems() if re in rgraph.edges() }
            print 'reduced costs on res. graph, for shortest paths: %s' % repr( cert )
            
            dist, upstream = Dijkstra( rgraph, redcost, s )
            print 'Dijkstra shortest path distances: %s' % repr( dist )
            print 'Dijkstra upstreams: %s' % repr( upstream )
            
            # find shortest path w.r.t. reduced costs (just follow ancestry links to the root)
            PATH = [] ; j = t
            while j is not s :
                e = upstream[j]
                i,_ = rgraph.endpoints(e)
                PATH.insert( 0, e )
                j = i
            print 'using path: %s' % repr( PATH )
            
            # augment Delta flow along the path P
            for e,dir in PATH :
                flow[e] += dir * Delta
                LinearizeCost( lincost, cost, flow, Delta, network, edge=e )    # all edges
                ResidualGraph( rgraph, flow, capacity, Delta, network, edge=e )
                
            # update the potentials; 
            # by connectivity, should touch *every* node
            for i in network.nodes() : potential[i] -= dist[i]
            
            # re-compute the reduced costs... everywhere? (all the potentials have changed)
            ReducedCost( redcost, lincost, potential, network )
            
            
        # at end of each stage, verify the optimality certificate (should be empty every time)
        CERT = { re : c for (re,c) in redcost.iteritems() if re in rgraph.edges() and c < 0. }
        print 'certificate, end stage TWO: %s' % repr( CERT )
        RELAXCERT = { re : c for (re,c) in redcost.iteritems() if re in rgraph.edges() and c < -10**-10 }
        if len( RELAXCERT ) > 0 : print "STAGE TWO CERTIFICATE CORRUPT!"
        assert len( RELAXCERT ) <= 0
                    
        # end the phase
        if Delta <= epsilon : break
        Delta = Delta / 2
    
    return flow






if __name__ == '__main__' :
    import networkx as nx
    import matplotlib.pyplot as plt
    """
    convert linear instances on non-multi graphs to networkx format
    for comparison against nx.min_cost_flow() algorithm
    """
    def mincostflow_nx( network, capacity, supply, weight ) :
        digraph = nx.DiGraph()
        for i in network.nodes() :
            digraph.add_node( i, demand=-supply.get(i, 0. ) )
            
        for e in network.edges() :
            i,j = network.endpoints(e)
            digraph.add_edge( i, j, capacity=capacity.get(e, np.Inf ), weight=weight.get(e, 1. ) )
        return digraph
    
    
    g = mygraph()
    
    if False :
        g.add_edge( 'a', 0, 1 )
        g.add_edge( 'b', 1, 2 )
        g.add_edge( 'c', 2, 3 )
        g.add_edge( 'd', 3, 0 )
        
        u = { e : 10. for e in g.edges() }
        supply = { 0 : 1., 1 : 2., 2 : -3., 3 : 0. }
        c = { 'a' : 10., 'b' : 5., 'c' : 1., 'd' : .5 }
    else :
        u = {}
        c = {}
        s = {}
        
        s[0] = 10.
        
        g.add_edge( 'a', 0, 1 )
        c['a'] = 1.
        #u['a'] = 1.35
        
        #g.add_edge( 'aprime', 0, 1 )
        #c['aprime'] = 1000.
        
        g.add_edge( 'b', 0, 2 )
        c['b'] = 10.
        
        g.add_edge( 'c', 1, 3 )
        g.add_edge( 'd', 2, 3 )
        
        s[3] = -10.
        supply = s
    
    cf = {}
    #for e in c : cf[e] = line( c[e] )
    cf['a'] = lambda x : 5.5 * x + 100.
    cf['b'] = lambda x : np.power( x, 2.0 )
    #cf['c'] = lambda x : 2. * np.exp( .5 * ( x - 1. ) )
    cf['c'] = lambda x : 2. * np.exp( .5 * ( 1. - x ) )
    
    def show( func ) :
        x = np.linspace(0,10,1000)
        y = [ func(xx) for xx in x ]
        plt.figure()
        plt.plot(x,y)
        plt.show()
    
    
    def FLOWCOST( flow, cost ) :
        res = [ cc( flow.get( e, 0. ) ) for e, cc in cost.iteritems() ]    # big difference!
        #res = [ cost.get( e, line(0.) )( flow[e] ) for e in flow ]
        return sum( res )
    
    def FEAS( flow, capacity, network ) :
        for e in network.edges() :
            if flow.get(e, 0. ) > capacity.get(e, np.Inf ) : return False
        return True
    
    flow = MinConvexCostFlow( g, u, supply, cf, epsilon=.001 )
    print ( flow, FLOWCOST( flow, cf ), FEAS( flow, u, g ) )
    
    flowstar = { 'b' : 10., 'd' : 10. }
    print ( flowstar, FLOWCOST( flowstar, cf ), FEAS( flowstar, u, g ) )
    
    
    digraph = mincostflow_nx( g, u, supply, c )
    compare = nx.min_cost_flow( digraph )
    
    
    
    
    
