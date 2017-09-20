source("sgm.interface.R")

args <- commandArgs(trailingOnly=TRUE)
if (length(args)<2) {
    stop("At least two argument must be supplied, which are either a n x n adjacency matrix or a n x 2 edgelist, and an optional s, the number of seeds, with 0 as default..\n", call.=FALSE)
} else if (length(args)==2) {
    input1 <- args[1]
    input2 <- args[2]
    S <- NULL
} else {
    input1 <- args[1]
    input2 <- args[2]
    S <- args[3]
}


cat("working with ", input1, ",", input2, "\n")

sgm.interface(input1, input2, S)
