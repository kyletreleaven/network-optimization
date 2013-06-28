
""" deprecate ASAP """

import itertools

import numpy as np

import networkx as nx
import cvxpy


DEFAULT_DELTA = 0.0001  # there are numerical issues if DELTA == 0.


"""
a flow graph is a DiGraph which may have any of the following properties:
    flow: a cvxpy.variable(), the flow on the edge, in its direction
    minflow: a lower bound for flow, may be -np.inf
    maxflow: an upper bound for flow, may be np.inf
    cost: a disciplined convex function, in terms of flow only
    
the nodes 's' and 't' of a flow graph do not need to obey flow conservation constraints

the flow property is mandatory, the others are not
"""


""" convenience """
def peel_flow( data ) :
    return data.get( 'flow', 0. )

def peel_cost( data, cost='cost' ) :
    return data.get( cost, 0. )
    
def peel_constraints( data ) :
    return data.get('constraints', [] )

def node_data_iter( graph ) :
    return ( data for _,data in graph.nodes_iter( data=True ) )

def edge_data_iter( graph ) :
    return ( data for _,__,data in graph.edges_iter( data=True ) )

def graph_data_iter( graph ) :
    return itertools.chain( node_data_iter( graph ), edge_data_iter( graph ) )




""" exports """

def attach_flownx_constraints( flownx ) :
    """ add flow capacity constraints to edges """
    for _,__, data in flownx.edges_iter( data=True ) :
        flowvar = data.get('flow')
        constr = []
        
        minflow = data.get( 'minflow', -np.inf )
        if minflow > -np.inf : constr.append( cvxpy.geq( flowvar, minflow ) )
        maxflow = data.get( 'maxflow', np.inf )
        if maxflow < np.inf : constr.append( cvxpy.leq( flowvar, maxflow ) )
        
        data['constraints'] = constr
    
    """ add flow conservation constraints to nodes, besides 's' and 't' """
    for u, node_data in flownx.nodes_iter( data=True ) :
        if u in [ 's', 't' ] : continue
        
        in_flow = 0.
        for _,__, data in flownx.in_edges_iter( u, data=True ) :
            in_flow += data.get('flow')
            
        out_flow = 0.
        for _,__, data in flownx.out_edges_iter( u, data=True ) :
            out_flow += data.get('flow')
            
        node_data['constraints'] = [ cvxpy.eq( in_flow, out_flow ) ]
        


""" Synthesis of Flow Network Data """

def compute_totalflow( digraph ) :
    total_flow = 0.
    for _,__, data in digraph.out_edges_iter( 's', data=True ) :
        total_flow += data['flow']
    return total_flow

def compute_totalcost( digraph, cost='cost' ) :
    total_cost = 0.
    for data in graph_data_iter( digraph ) :
        total_cost += peel_cost( data, cost=cost )
    return total_cost

def collect_constraints( digraph ) :
    constraints = []
    for data in graph_data_iter( digraph ) :
        constraints.extend( peel_constraints( data ) )
    return constraints
    
def flownx_to_opt( digraph ) :
    total_flow = compute_totalflow( digraph )
    total_cost = compute_totalcost( digraph )
    constraints = collect_constraints( digraph )
    return total_flow, total_cost, constraints
    
    
""" PROBLEMS """

def maxflow( flownx ) :
    total_flow = compute_totalflow( flownx )
    constraints = collect_constraints( flownx )
    prog = cvxpy.program( cvxpy.maximize( total_flow ), constraints )
    max_flow = prog.solve()
    return max_flow

def mincost_maxflow( flownx, cost='cost', DELTA=None ) :
    if DELTA is None : DELTA = DEFAULT_DELTA
    
    total_flow = compute_totalflow( flownx )
    total_cost = compute_totalcost( flownx, cost=cost )
    constraints = collect_constraints( flownx )
    
    prog1 = cvxpy.program( cvxpy.maximize( total_flow ), constraints )
    max_flow = prog1.solve()
    
    constraints2 = [ c for c in constraints ]
    constraints2.append( cvxpy.geq( total_flow, max_flow - DELTA ) )
    prog2 = cvxpy.program( cvxpy.minimize( total_cost ), constraints2 )
    res = prog2.solve()
    return res
    


