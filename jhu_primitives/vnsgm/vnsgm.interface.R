if(!suppressMessages(require(VN))) {
    install.packages("http://www.cis.jhu.edu/~parky/D3M/VN_0.3.0.tar.gz",type="source")
    suppressMessages(library(VN))
}
if(!require(igraph)) {
    install.packages("igraph")
    suppressMessages(library(igraph))
}

vnsgm.interface <- function(g1, g2, voi, S)
{

    R <- 100
    gamma <- 1
    h <- ell <- 1
    out <- vnsgm(voi,S,g1,g2,h,ell,R,gamma)$P

    return(out)

#    cat("The output file is saved in '../DATA/out.txt'.\n")
#    write.table(out,"../DATA/out.txt", row.names=F, col.names=F)
}
