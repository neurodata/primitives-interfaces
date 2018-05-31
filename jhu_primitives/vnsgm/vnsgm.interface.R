if(!suppressMessages(require(VN))) {
    suppressMessages(require(devtools))
    devtools::install_github("youngser/VN")
    suppressMessages(library(VN))
}
if(!require(igraph)) {
    install.packages("igraph")
    suppressMessages(library(igraph))
}

vnsgm.interface <- function(g1, g2, voi, S)
{

    if (class(g1) == "dgCMatrix" || class(g1) == 'matrix') {
        g1 = igraph::graph_from_adjacency_matrix(g1)
    }
    if (class(g2) == "dgCMatrix" || class(g2) == 'matrix') {
        g2 = igraph::graph_from_adjacency_matrix(g2)
    }

    R <- 100
    gamma <- 1
    h <- ell <- 1
    out <- vnsgm(voi,S,g1,g2,h,ell,R,gamma)$P

    return(out)

#    cat("The output file is saved in '../DATA/out.txt'.\n")
#    write.table(out,"../DATA/out.txt", row.names=F, col.names=F)
}
