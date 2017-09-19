source("dimselect.interface.R")

args <- commandArgs(trailingOnly=TRUE)
if (length(args)==0) {
    stop("At least one argument must be supplied (input file, a numeric vector.\n", call.=FALSE)
} else if (length(args)==1) {
    input <- args[1]
}

dimselect.interface(input)
