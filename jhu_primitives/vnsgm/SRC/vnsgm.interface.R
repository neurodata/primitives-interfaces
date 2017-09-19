if(!suppressMessages(require(VN))) {
    install.packages("http://www.cis.jhu.edu/~parky/D3M/VN_0.3.0.tar.gz",type="source")
    suppressMessages(library(VN))
}
if(!require(igraph)) {
    install.packages("igraph")
    suppressMessages(library(igraph))
}

vnsgm.interface <- function(g1, g2, voi, s)
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

    W <- intersect(V(g1),V(g2)) # shared vertices
    W <- setdiff(W,voi) # exclude x from W
    maxseed <- min(length(W),s)
    S <- sort(sample(W,maxseed))

    R <- 100
    gamma <- 1
    h <- ell <- 1
    out <- vnsgm(voi,S,g1,g2,h,ell,R,gamma,sim=FALSE,plotF=FALSE)$P

    return(out)

#    cat("The output file is saved in '../DATA/out.txt'.\n")
#    write.table(out,"../DATA/out.txt", row.names=F, col.names=F)
}
