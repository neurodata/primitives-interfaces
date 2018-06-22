if(!require(igraph)){
    install.packages("igraph")
    suppressMessages(require(igraph))
}

ase.interface <- function(g, dim)
{
    ## embedding into "dim"
    if (class(g) == "dgCMatrix" || class(g) == "matrix") {
        g = igraph::graph_from_adjacency_matrix(g)
    }
    SVD <- embed_adjacency_matrix(g, dim)
    X <-SVD$X
    D <- SVD$D

    return(list(X,D))
}
