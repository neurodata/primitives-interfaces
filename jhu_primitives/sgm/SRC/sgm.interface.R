if(!require(VN)) {
    install.packages("http://www.cis.jhu.edu/~parky/D3M/VN_0.3.0.tar.gz",type="source")
    suppressMessages(library(VN))
}
if(!require(igraph)) {
    install.packages("igraph")
    suppressMessages(library(igraph))
}

sgm.interface <- function(g1, g2, s)
{
    ## X <- as.matrix(read.table(input1))
    ## if (ncol(X)==2) {
    ##     g1 <- graph.edgelist(X)
    ## } else {
    ##     g1 <- graph.adjacency(X)
    ## }
    ## X <- as.matrix(read.table(input2))
    ## if (ncol(X)==2) {
    ##     g2 <- graph.edgelist(X)
    ## } else {
    ##     g2 <- graph.adjacency(X)
    ## }

#    g1 <- read.graph(input1, "gml")
#    g2 <- read.graph(input2, "gml")

    A1 <- as.matrix(g1[]); n <- nrow(A1)
    A2 <- as.matrix(g2[]); m <- nrow(A2)

    gamma <- 1
    niter <- 30
    M <- rsp(n-s,gamma)
    S <- diag(n);
    S[(s+1):n,(s+1):n] <- M
    out <- sgm(A2, A1, 0, start=S, pad=0, iteration=niter)$P

    return(out)
#    cat("The output file is saved in '../DATA/out.txt'.\n")
#    write.table(out,"../DATA/out.txt", row.names=F, col.names=F)
}
