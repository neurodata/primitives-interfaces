if(!require(gmmase)){
    require(devtools)
    suppressMessages(devtools::install_github("youngser/gmmase"))
    suppressMessages(library(gmmase))
}
if(!require(igraph)){
    install.packages("igraph")
    suppressMessages(library(igraph))
}

ptr.interface <- function(g)
{
#    X <- as.matrix(read.table(input))
    g <- ptr(g)
    out <- as.matrix(g[])

    return(out)

#    cat("The output file, a new n x n adjacency matrix, is saved in '../DATA/out.txt'.\n")
#    write.table(out,"../DATA/out.txt", row.names=FALSE, col.names=FALSE)
}
