from jhu_primitives.core.JHUGraph import JHUGraph
from jhu_primitives.utils.util import gen_graph_r

gen_graph_r() # path is always /tmp/graph.gml
g = JHUGraph()
g.read_graph(fname="/tmp/graph.gml")
g.summary()
print(("\nweighted: {}\n".format(g.is_weighted())))
print(("num_vertices: {}\n".format(g.get_num_vertices())))
print(("num_edges: {}\n".format(g.get_num_edges())))
print(("dangling nodes: {}\n".format(g.get_dangling_nodes())))
g.get_adjacency_matrix()
