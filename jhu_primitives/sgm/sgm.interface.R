if(!require(VN)) {
    suppressMessages(require(devtools))
    devtools::install_github("youngser/VN")
    suppressMessages(library(VN))
}
if(!require(igraph)) {
    install.packages("igraph")
    suppressMessages(library(igraph))
}

sgm.interface <- function(g1, g2, S)
{
    if (class(g1) == "dgCMatrix" || class(g1) == 'matrix') {
        g1 = igraph::graph_from_adjacency_matrix(g1)
    }
    if (class(g2) == "dgCMatrix" || class(g2) == 'matrix') {
        g2 = igraph::graph_from_adjacency_matrix(g2)
    }


    A1 <- as.matrix(g1[]); n <- nrow(A1)
    A2 <- as.matrix(g2[]); m <- nrow(A2)

    gamma <- 1
#    niter <- 30
    s <- nrow(S)

    if(is.null(s) || s == 1){
        s <- 0
        S <- NULL
    }else{
        S <- S[,c(2,1)]
    }

    out <- sgm(A2, A1, S)$P
    return(out)
}
