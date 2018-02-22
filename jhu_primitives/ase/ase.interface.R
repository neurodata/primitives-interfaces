if(!require(igraph)){
    install.packages("igraph")
    suppressMessages(require(igraph))
}

ase.interface <- function(g, dim)
{

    ## embedding into "dim"
    if (class(g) == "dgCMatrix")
        g = igraph::graph_from_adjacency_matrix(g)

    X <- embed_adjacency_matrix(g, dim)$X

    return(X)
}
