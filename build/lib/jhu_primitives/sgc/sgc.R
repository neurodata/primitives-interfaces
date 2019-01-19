source("sgc.interface.R")

args <- commandArgs(trailingOnly=TRUE)
if (length(args)<1) {
    stop("At least one argument must be supplied, which is either a n x n adjacency matrix or a n x 2 edgelist.\n", call.=FALSE)
} else {
    input <- args[1]
} 

cat("working with ", input, "\n")

sgc.interface(input)
