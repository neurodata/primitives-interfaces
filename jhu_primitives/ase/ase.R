source("ase.interface.R")

args <- commandArgs(trailingOnly=TRUE)
if (length(args)==0) {
    stop("At least one argument must be supplied (input file).\n", call.=FALSE)
} else if (length(args)==1) {
    ## default embedding dimension
    input <- args[1]
    dim <- 2
} else {
    input <- args[1]
    dim <- args[2]
}

cat("working with ", input, ", embedding into ", dim, "dimension (2 is default).\n")

ase.interface(input, dim)
