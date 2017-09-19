source("numclust.interface.R")

args <- commandArgs(trailingOnly=TRUE)
if (length(args)<1) {
    stop("One argument must be supplied, which is a numeric vector containing clustering criteria, e.g., BIC or Average Silhouette numbers.\n", call.=FALSE)
} else {
    input <- args[1]
}

cat("working with ", input, "\n")

X <- scan(input)
numclust.interface(X)
