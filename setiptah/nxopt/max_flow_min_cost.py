
import itertools

import numpy as np

import networkx as nx

import cvxpy

# fix!
class mycvxpy :
    """ adapter for any future cvxpy APIs """
    
    # variables
    @classmethod
    def variable(cls) : return cvxpy.Variable()
        
    # constraints
    @classmethod
    def eq(cls, lhs, rhs ) : return cvxpy.constraints.EqConstraint( lhs, rhs )
    @classmethod
    def leq(cls, lhs, rhs ) : return cvxpy.constraints.LeqConstraint( lhs, rhs )
    @classmethod
    def geq(self, lhs, rhs ) : return cvxpy.constraints.LeqConstraint( rhs, lhs )
    
    # math
    #@classmethod
    #def sum(cls, x ) : return cvxpy.sum( x )
    
    # objective
    @classmethod
    def minimize(cls, x ) : return cvxpy.Minimize( x )
    @classmethod
    def maximize(cls, x ) : return cvxpy.Maximize( x )
    
    # program
    @classmethod
    def program(cls, *args, **kwargs ) : return cvxpy.Problem( *args, **kwargs )
    



DEFAULT_DELTA = 0.00001  # there are numerical issues if DELTA == 0.


class data(object) : pass

class node(object) : pass


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

def totalflow( flowgraph, flow='flow' ) :
    source_node = flowgraph.graph['s']
    flows = [ peel_flow( data, flow=flow ) for _,__,data in flowgraph.out_edges_iter( source_node, data=True ) ]
    return sum( flows )

def collect_constraints( flowgraph, constraints='constraints' ) :
    constraints = []
    for data in graph_data_iter( flowgraph ) :
        constraints.extend( peel_constraints( data ) )
    return constraints

def totalcost( costgraph, cost='cost' ) :
    costs = [ peel_cost( data, cost=cost ) for data in edge_data_iter( costgraph ) ]
    return sum( costs )







""" network transformation utilities """ 


def obtainCapacityNetwork( lengraph, supplygraph, length='length', weight1='weight1', weight2='weight2' ) :
    res = nx.MultiDiGraph()
    s = node() ; res.add_node( s )      # just in case; needs to be there
    t = node() ; res.add_node( t )
    
    # add all the edges from lengraph
    res.add_edges_from( lengraph.edges_iter() )      # I think that works
    
    # add the supply/demand capacity edges
    for u, data in supplygraph.nodes_iter( data=True ) :
        supply = data.get( weight1, 0. )
        if supply > 0. : res.add_edge( s, u, capacity=supply )
        
        demand = data.get( weight2, 0. )
        if demand > 0. : res.add_edge( u, t, capacity=demand )
        
    return res, s, t




""" construct an optimization graph with flows and constraints (no costs) """
def obtainFlowNetwork( digraph, s, t, capacity='capacity', **kwargs ) :
    if isinstance( digraph, nx.MultiDiGraph ) :
        ITER = digraph.edges_iter( keys=True, data=True )
    elif isinstance( digraph, nx.DiGraph ) :
        ITER = digraph.edges_iter( data=True )
    else :
        raise Exception( 'must be directed graph' )
    
    # these and <weight> are the output attributes
    flow_out = kwargs.get( 'flow', 'flow' )
    constraints_out = kwargs.get( 'constraints', 'constraints' )
    
    # build the graph
    flowgraph = nx.MultiDiGraph()
    
    for it in ITER :
        e = it[:-1] ; data = it[-1]
        cap = data.get( capacity, np.inf )
        
        flow = mycvxpy.variable()
        constr = [ mycvxpy.geq( flow, 0.) ]
        if cap < np.inf : constr.append( mycvxpy.leq( flow, cap ) )
        #print weight, flow, cost
        
        edge_data = { flow_out : flow, constraints_out : constr }
        flowgraph.add_edge( *e, attr_dict=edge_data )
        
    for u, u_data in flowgraph.nodes_iter( data=True ) :
        if u in [ s, t ] : continue
        
        in_flows = [ data.get( flow_out ) for _,__,data in flowgraph.in_edges( u, data=True ) ]
        in_degree = sum( in_flows )
        out_flows = [ data.get( flow_out ) for _,__,data in flowgraph.out_edges( u, data=True ) ]
        out_degree = sum( out_flows )
        
        u_data[ constraints_out ] = [ mycvxpy.eq( out_degree, in_degree ) ]
        
    flowgraph.graph['s'] = s
    flowgraph.graph['t'] = t
    return flowgraph


def obtainWeightedCosts( flowgraph, wgraph, weight='weight', **kwargs ) :
    if isinstance( wgraph, nx.MultiDiGraph ) :
        LOOKUP = lambda *e : wgraph.get_edge_data( *e )
    elif isinstance( wgraph, nx.DiGraph ) :
        LOOKUP = lambda u,v,key : wgraph.get_edge_data( u, v )
    else :
        raise Exception( 'digraph must be a digraph' )
    
    res = nx.MultiDiGraph()
    flow_in = kwargs.get( 'flow', 'flow' )
    cost_out = kwargs.get( 'cost', 'cost' )
    
    for u,v,key, flow_data in flowgraph.edges_iter( keys=True, data=True ) :
        wgraph_data = LOOKUP( u, v, key )
        
        if wgraph_data is None :
            edge_data = None
        else :
            curr_flow = flow_data.get( flow_in, 0. )
            curr_weight = wgraph_data.get( weight, 0. )
            edge_data = { cost_out : curr_weight * curr_flow }
        
        res.add_edge( u,v,key, attr_dict=edge_data )
        
    return res



""" compute quantities from the flow graph """

def flow_values( flowgraph, **kwargs ) :
    flow_in = kwargs.get( 'flow_in', 'flow' )
    flow_out = kwargs.get( 'flow_out', 'flow' )
    
    res = nx.MultiDiGraph()
    for u,v,key, data in flowgraph.edges_iter( keys=True, data=True ) :
        flowvar = data.get( flow_in )
        flowval = flowvar.value
        res.add_edge( u,v, key, { flow_out : flowval } )
    return res

def cost_values( costgraph, **kwargs ) :
    cost_in = kwargs.get( 'cost_in', 'cost' )
    cost_out = kwargs.get( 'cost_out', 'cost' )
    
    def get_val( var ) :
        try :
            return var.value
        except AttributeError :
            return var
        
    res = nx.MultiDiGraph()
    for u,v,key, data in costgraph.edges_iter( keys=True, data=True ) :
        cost_expr = data.get( cost_in, 0. )
        cost_val = get_val( cost_expr )
        res.add_edge( u,v, key, { cost_out : cost_val } )
    return res



""" PROBLEMS """

def max_flow( flowgraph, **kwargs ) :
    flow_in = kwargs.get( 'flow', 'flow' )
    constraints_in = kwargs.get( 'constraints', 'constraints' )
    
    total_flow = totalflow( flowgraph, flow=flow_in )
    constraints = collect_constraints( flowgraph, constraints=constraints_in )
    program = mycvxpy.program( mycvxpy.maximize( total_flow ), constraints )
    #
    quiet = kwargs.get( 'quiet', False )
    program.solve( quiet )
    
def max_flow_min_cost( flowgraph, costgraph, **kwargs ) :
    flow_in = kwargs.get( 'flow', 'flow' )
    constraints_in = kwargs.get( 'constraints', 'constraints' )
    cost_in = kwargs.get( 'cost', 'cost' )
    quiet = kwargs.get( 'quiet', False )
    delta = kwargs.get( 'delta', DEFAULT_DELTA )
    
    total_flow = totalflow( flowgraph, flow=flow_in )
    constraints = collect_constraints( flowgraph, constraints=constraints_in )
    
    program1 = mycvxpy.program( mycvxpy.maximize( total_flow ), constraints )
    max_flow = program1.solve( quiet )
    
    constraints2 = [ c for c in constraints ] + [ mycvxpy.geq( total_flow, max_flow - delta ) ]
    total_cost = totalcost( costgraph, cost=cost_in )
    
    program2 = mycvxpy.program( mycvxpy.minimize( total_cost ), constraints2 )
    program2.solve( quiet )





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
    flowgraph = obtainFlowNetwork( g, 's', 't' )
    costgraph = obtainWeightedCosts( flowgraph, g )
    
    max_flow_min_cost( flowgraph, costgraph )
    print totalcost( costgraph ).value
    






