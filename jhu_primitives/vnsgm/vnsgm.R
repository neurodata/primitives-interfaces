source("vnsgm.interface.R")

args <- commandArgs(trailingOnly=TRUE)
if (length(args)<4) {
    stop("All arguments must be supplied, either a gml or a n x 2 edgelist (2), and x, the vertex of interest, and S, the matrix of seeds -- first column is seed indices for first graph, second for second graph.\n", call.=FALSE)
} else {
    input1 <- args[1]
    input2 <- args[2]
    voi <- args[3]
    S <- args[4]
}


cat("working with ", input1, ",", input2, "\n")

vnsgm.interface(input1, input2, voi, S)
