if(!require(mclust)){
    install.packages("mclust")
    suppressMessages(library(mclust))
}

gclust.interface <- function(X, K=2) {
    return(Mclust(X, 1:K)$G)
}
