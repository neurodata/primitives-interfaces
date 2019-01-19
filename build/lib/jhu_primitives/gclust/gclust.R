source("gclust.interface.R")

args <- commandArgs(trailingOnly=TRUE)
if (length(args)==0) {
    stop("At least two argument must be supplied (input file, a numeric matrix, and the largest number of clusters.\n", call.=FALSE)
} else if (length(args)==2) {
    input <- args[1]
    K <- args[2]
}

cat("working with ", input, ", clustering up to ", K, " clusters...\n")

X <- as.matrix(read.table(input))
gclust.interface(X, K)

