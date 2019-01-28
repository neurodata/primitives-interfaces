source("oocase.interface.R")

args <- commandArgs(trailingOnly=TRUE)
if (length(args)==0) {
    stop("At least one argument must be supplied (input file, which has to be n x 2 edgelist in the text format!). The second argument can be dmax, the maximum embedding dimension, which is optional with a default value of 2.\n", call.=FALSE)
} else if (length(args)==1) {
    ## default embedding dimension
    input <- args[1]
    dmax <- 2
} else {
    input <- args[1]
    dmax <- as.integer(args[2])
}

cat("working with ", input, ", embedding into ", dmax, "dimension (2 is default).\n")

oocase.interface(input, dmax)
