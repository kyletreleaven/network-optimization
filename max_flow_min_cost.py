
import itertools

import numpy as np

import networkx as nx
import cvxpy

DEFAULT_DELTA = 0.0001  # there are numerical issues if DELTA == 0.


class data(object) : pass


""" convenience """
def peel_flow( data, flow='flow' ) : return data.get( flow, 0. )

def peel_weight( data, weight='weight' ) : return data.get( weight, 0. )

def peel_cost( data, cost='cost' ) : return data.get( cost, 0. )

def peel_constraints( data, constraints='constraints' ) :
    return data.get( constraints, [] )

def node_data_iter( graph ) :
    return ( data for _,data in graph.nodes_iter( data=True ) )

def edge_data_iter( graph ) :
    return ( data for _,__,data in graph.edges_iter( data=True ) )

def graph_data_iter( graph ) :
    return itertools.chain( node_data_iter( graph ), edge_data_iter( graph ) )

def compute_totalflow( flowgraph, s, flow='flow' ) :
    flows = [ peel_flow( data, flow=flow ) for _,__,data in flowgraph.out_edges_iter( s, data=True ) ]
    return sum( flows )

def compute_totalcost( flowgraph, flow='flow', cost='cost' ) :
    costs = [ peel_cost( data, cost=cost ) for data in edge_data_iter( flowgraph ) ]
    return sum( costs )

def collect_constraints( flowgraph, constraints='constraints' ) :
    constraints = []
    for data in graph_data_iter( flowgraph ) :
        constraints.extend( peel_constraints( data ) )
    return constraints





""" a nice utility for weighted graphs """
def OptimizationGraph( digraph, s, t, capacity='capacity', weight_in='weight', **kwargs ) :
    if isinstance( digraph, nx.MultiDiGraph ) :
        ITER = digraph.edges_iter( keys=True, data=True )
    elif isinstance( digraph, nx.DiGraph ) :
        ITER = digraph.edges_iter( data=True )
    else :
        raise Exception( 'must be directed graph' )
    
    # these and <weight> are the output attributes
    flow_key = kwargs.get( 'flow', 'flow' )
    constraints_key = kwargs.get( 'constraints', 'constraints' )
    cost_out = kwargs.get( 'cost_out', 'cost' )
    
    # build the graph
    opt = nx.MultiDiGraph()
    
    for it in ITER :
        e = it[:-1] ; data = it[-1]
        cap = data.get( capacity, np.inf )
        weight = data.get( weight_in, 0. )
        
        flow = cvxpy.variable()
        constr = [ cvxpy.geq( flow, 0.) ]
        if cap < np.inf : constr.append( cvxpy.leq( flow, cap ) )
        cost = weight * flow
        #print weight, flow, cost
        
        edge_data = { flow_key : flow, constraints_key : constr, cost_out : cost }
        opt.add_edge( *e, attr_dict=edge_data )
        
    for u, u_data in opt.nodes_iter( data=True ) :
        if u in [ s, t ] : continue
        
        in_flows = [ data.get( flow_key ) for _,__,data in opt.in_edges( u, data=True ) ]
        in_degree = cvxpy.sum( in_flows )
        out_flows = [ data.get( flow_key ) for _,__,data in opt.out_edges( u, data=True ) ]
        out_degree = cvxpy.sum( out_flows )
        
        u_data[ constraints_key ] = [ cvxpy.eq( out_degree, in_degree ) ]
        
    opt.graph['s'] = s
    opt.graph['t'] = t
    return opt
        
        
def FlowGraphStatistics( flowgraph, **kwargs ) :
    s = flowgraph.graph['s']        # used to make a lot of things depend on s; no longer!
    
    flow = kwargs.get( 'flow', 'flow' )
    constraints = kwargs.get( 'constraints', 'constraints' )
    cost_in = kwargs.get( 'cost', 'cost' )
    
    res = data()
    res.total_flow = compute_totalflow( flowgraph, s, flow=flow )
    res.constraints = collect_constraints( flowgraph, constraints=constraints )
    res.total_cost = compute_totalcost( flowgraph, flow=flow, cost=cost_in )
    
    return res


""" compute quantities from the flow graph """
 
def flow_values( flowgraph, flow='flow', flow_in='flow' ) :
    res = nx.MultiDiGraph()
    for u,v,key, data in flowgraph.edges_iter( keys=True, data=True ) :
        flowvar = data.get( flow_in )
        flowval = flowvar.value
        res.add_edge( u,v, key, { flow : flowval } )
        
    return res

def flow_costs( flowgraph, cost='cost', cost_in='cost' ) :
    def get_val( var ) :
        try :
            return var.value
        except AttributeError :
            return var
        
    res = nx.MultiDiGraph()
    for u,v,key, data in flowgraph.edges_iter( keys=True, data=True ) :
        cost_expr = data.get( cost_in, 0. )
        cost_val = get_val( cost_expr )
        res.add_edge( u,v, key, { cost : cost_val } )
        
    return res

def flow_cost( flowgraph, cost='cost' ) :
    digraph = flow_costs( flowgraph, 'cost', cost_in=cost )
    costs = [ data.get('cost') for _,__,data in digraph.edges_iter( data=True ) ]
    return sum( costs )


def flow_weighted_cost( flowgraph, digraph, flow='flow', weight='weight' ) :
    if isinstance( digraph, nx.MultiDiGraph ) :
        LOOKUP = lambda *e : digraph.get_edge_data( *e )
    elif isinstance( digraph, nx.DiGraph ) :
        LOOKUP = lambda u,v,key : digraph.get_edge_data( u, v )
    else :
        raise Exception( 'digraph must be a digraph' )
    
    res = 0.
    for u,v,key, flow_data in flowgraph.edges_iter( keys=True, data=True ) :
        graph_data = LOOKUP( u, v, key )
        
        curr_flow = flow_data.get( flow, 0. )
        curr_weight = graph_data.get( weight, 0. )
        #print ( curr_flow, curr_weight )
        res += curr_flow * curr_weight
        
    return res




""" PROBLEMS """

def compute_max_flow( flowgraph, flow_out='flow', **kwargs ) :
    flow_in = kwargs.get( 'flow', 'flow' )
    constraints_in = kwargs.get( 'constraints', 'constraints' )
    #weight_in = kwargs.get( 'weight', 'weight' )
    
    stats = FlowGraphStatistics( flowgraph, flow=flow_in, constraints=constraints_in )
    program = cvxpy.program( cvxpy.maximize( stats.total_flow ), stats.constraints )
    quiet = kwargs.get( 'quiet', False )
    program.solve( quiet )
    
def max_flow( digraph, s, t, capacity='capacity', weight='weight', flow='flow' ) :
    opt = OptimizationGraph( digraph, s, t, capacity=capacity, weight=weight )
    compute_max_flow( opt )
    return opt





def compute_max_flow_min_cost( flowgraph, flow_out='flow', **kwargs ) :
    flow_in = kwargs.get( 'flow', 'flow' )
    constraints_in = kwargs.get( 'constraints', 'constraints' )
    
    stats = FlowGraphStatistics( flowgraph, flow=flow_in, constraints=constraints_in )
    
    program1 = cvxpy.program( cvxpy.maximize( stats.total_flow ), stats.constraints )
    max_flow = program1.solve()
    
    delta = kwargs.get( 'delta', DEFAULT_DELTA )
    constraints2 = [ c for c in stats.constraints ] + [ cvxpy.geq( stats.total_flow, max_flow - delta ) ]
    
    program2 = cvxpy.program( cvxpy.minimize( stats.total_cost ), constraints2 )
    quiet = kwargs.get( 'quiet', False )
    program2.solve( quiet )


def max_flow_min_cost( digraph, s, t, capacity='capacity', weight='weight', flow='flow' ) :
    opt = OptimizationGraph( digraph, s, t, capacity=capacity, weight=weight )
    compute_max_flow_min_cost( opt )
    return opt










if __name__ == '__main__' :
    
    g = nx.MultiDiGraph()
    
    if False :
        g.add_edge( 0, 't', capacity=2.5 )
        g.add_edge( 's', 0, weight=1., capacity=2. )
        g.add_edge( 's', 0, weight=1000. )
        
    elif True :
        g.add_edge( 's', 0, capacity=10. )
        g.add_edge( 0, 1, capacity=5. )
        g.add_edge( 1, 't', capacity= 7.5 )
        
        g.add_edge( 's', 1, capacity=5., weight=1. )
        g.add_edge( 0, 't', capacity=2., weight=1. )
    
    #opt = OptimizationGraph( g, 's', 't' )
    #res = max_flow_on_flowgraph( opt )
    res = max_flow_min_cost( g, 's', 't' )
    
    print flow_cost( res )
    #print flow_cost( res, g )






