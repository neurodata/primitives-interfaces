if(!require(gmmase)){
    suppressMessages(require(devtools))
    devtools::install_github("youngser/gmmase")
    suppressMessages(library(gmmase))
}

nonpar.interface <- function(Xhat1, Xhat2, sigma=0.5)
{
    return(nonpar(Xhat1,Xhat2,sigma))

    #cat("The output file is saved in '../DATA/out.txt'.\n")
    #write(out,"../DATA/out.txt", ncol=1)
}
