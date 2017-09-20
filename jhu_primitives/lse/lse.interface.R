if(!require(igraph)){
    install.packages("igraph")
    suppressMessages(library(igraph))
}

lse.interface <- function(g, dim)
{
    ## X <- as.matrix(read.table(input))
    ## if (nrow(X) == ncol(X)) {
    ##     ## pass to rank
    ##     tmp <- X[X!=0]
    ##     nnz <- length(tmp)
    ##     rk <- rank(tmp) / nnz
    ##     X[X!=0] <- rk * 2 / nnz

    ##     ## form igraph object: directed, weighted, hollow
    ##     suppressMessages(require(igraph))
    ##     g <- graph_from_adjacency_matrix(X,weighted=TRUE,mode="directed",diag=FALSE)
    ## } else {
    ##     stop("The input matrix must be a square matrix!")
    ## }

    ## embedding into "dim"
    X <- embed_laplacian_matrix(g, dim)$X

    return(X)

    ## cat("The output files are saved in '../DATA/out_vectors.txt', '../DATA/in_vectors.txt', and '../DATA/eigenvalues'.\n")
    ## write.table(embed$X,"../DATA/out_vectors.txt", col.names=F, row.names=F)
    ## write.table(embed$Y,"../DATA/in_vectors.txt", col.names=F, row.names=F)
    ## write(embed$D,"../DATA/eigenvalues.txt", ncol=1)
}
