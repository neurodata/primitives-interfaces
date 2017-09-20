source("vnsgm.interface.R")

args <- commandArgs(trailingOnly=TRUE)
if (length(args)<3) {
    stop("At least three argument must be supplied, which are either a gml or a n x 2 edgelist, and x, the vertex of interest..., and s, the number of seeds, which takes 0 as default.\n", call.=FALSE)
} else if (length(args)==3) {
    input1 <- args[1]
    input2 <- args[2]
    voi <- arg[3]
    s <- 0
} else {
    input1 <- args[1]
    input2 <- args[2]
    voi <- arg[3]
    s <- args[4]
}


cat("working with ", input1, ",", input2, "\n")

vnsgm.interface(input1, input2, s)
