#if(!require(gmmase)){
#    require(devtools)
#    suppressMessages(devtools::install_github("youngser/gmmase"))
#    suppressMessages(library(gmmase))
#}

if(!require(igraph)){
    install.packages("igraph")
    suppressMessages(library(igraph))
}

passtoranks <- function(g)
{
    if (class(g) != "igraph") {
        if (!is.matrix(g)) stop("the input has to be either an igraph object or a matrix!")
        else {
            if (ncol(g)==2) g <- graph_from_edgelist(g)
            else if (nrow(g)==ncol(g)) g <- graph_from_adjacency_matrix(g, weighted = TRUE)
            else stop("the input matrix is not a graph format!")
        }
    }

    if (is.weighted(g)) {
        W <- E(g)$weight
    } else { # no-op!
        W <- rep(1,ecount(g))
    }

    E(g)$weight <- rank(W)*2 / (ecount(g)+1)
    return(g)
}

ptr.interface <- function(g)
{
#    X <- as.matrix(read.table(input))
    if (class(g) == "dgCMatrix" || class(g) == 'matrix') {
        g = igraph::graph_from_adjacency_matrix(g, weighted = TRUE)
        # if we always assume weighted, are there any issues if it is unweighted?
    }
    new_g <- passtoranks(g)
    out <- as.matrix(new_g[])

    return(out)

#    cat("The output file, a new n x n adjacency matrix, is saved in '../DATA/out.txt'.\n")
#    write.table(out,"../DATA/out.txt", row.names=FALSE, col.names=FALSE)
}
