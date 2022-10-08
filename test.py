import graphviz



if  __name__=="__main__":

    digraph = graphviz.Digraph(comment="example classification tree")

    node_names = ["A","B","C","D","E"]
    names_to_class = dict({})

    node_names = ["A","B","C"]
    name_to_class = dict({"A":"1", "B":"0","C":"1"})
    for node_name in node_names:
        digraph.node(node_name, name_to_class[node_name])

    digraph.edges(["AB", "AC"])
    digraph.render(directory='doctest-output', view=True)  



