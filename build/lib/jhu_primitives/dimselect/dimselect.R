source("dimselect.interface.R")

args <- commandArgs(trailingOnly=TRUE)
if (length(args)==0) {
    stop("At least one argument must be supplied (input file, a numeric vector.\n", call.=FALSE)
} else if (length(args)==1) {
    X <- args[1]
    n <- 3 # default number of elbows = 3
} else if (length(args)==2){
	X <- args[1]
	n <- args[2]
}

dimselect.interface(X, n)
