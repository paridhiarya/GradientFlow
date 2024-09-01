from graphviz import Digraph

def trace(root):
    nodes, edges = set(), set()
    
    #Adding nodes and edges
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._children:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

def draw(root):
    dot = Digraph(format='svg', graph_attr={'rankdir':'LR'})
    
    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        dot.node(name = uid, label = "{ %s | Data : %.4f | grad %.4f}" % (n.label, n.data, n.grad), shape='record')
        if n._operator:
            dot.node(name=uid + n._operator, label = n._operator)
            dot.edge(uid + n._operator, uid)
        
    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._operator)
    return dot
    
    
