library(FlashGraphR)

#' @param fm The FlashR object
#' @param nev The number of eigenvalues/vectors required.
#' @param which The type of the embedding.
#' @param c The constant used in the Aug variant of embedding.
#' @param ncv The number of vectors in the vector subspace.
#' @param tol Numeric scalar. Stopping criterion: the relative accuracy
#' of the Ritz value is considered acceptable if its error is less than
#' tol times its estimated value. If this is set to zero then machine
#' precision is used.
#' @return A named list with the following members:
#' \item{values}{Numeric vector, the desired eigenvalues.}
#' \item{vectors}{Numeric matrix, the desired eigenvectors as columns. If complex=TRUE
#'   (the default for non-symmetric problems), then the matrix is complex.}
#' \item{options}{A named list with the supplied options and some information
#'   about the performed calculation, including an ARPACK exit code.}
#' @name fm.oocase
#' @author Da Zheng <dzheng5@@jhu.edu>
fm.spectral.embedding <- function(fm, nev, which=c("A, Aug, L, nL"), tol=1.0e-12)
{
    ptm <- proc.time()

	stopifnot(!is.null(fm))
	stopifnot(class(fm) == "fm")

	# multiply function for eigen on the adjacency matrix
	# this is the default setting.
	multiply <- function(x, extra) fm %*% x
	multiply.right <- function(m) t(fm) %*% m
	multiply.left.diag <- function(v, m) fm.mapply.col(m, v, fm.bo.mul)
	get.degree <- function(fm) drop(fm %*% fm.rep.int(1, ncol(fm)))

    c <- 1/nrow(fm)
    ncv <- 2*nev

	directed = !fm.is.sym(fm)
	if (which == "L" || which == "nL") {
		if (directed) {
			print("Don't support embedding on a directed (normalized) Laplacian matrix")
			return(NULL)
		}
	}

	# Compute the opposite of the spectrum.
	comp.oppo <- FALSE
	if (which == "A") {
		if (directed) multiply <- function(x, extra) fm %*% (t(fm) %*% x)
		else multiply <- function(x, extra) fm %*% x
	}
	else if (which == "Aug") {
		cd <- get.degree(fm) * c
		if (directed) {
			multiply <- function(x, extra) {
				x <- fm.as.matrix(x)
				t <- t(fm) %*% x + multiply.left.diag(cd, x)
				fm %*% t + multiply.left.diag(cd, t)
			}
			multiply.right <- function(m) {
				m <- fm.as.matrix(m)
				t(fm) %*% m + multiply.left.diag(cd, m)
			}
		}
		else
			multiply <- function(x, extra) {
				x <- fm.as.matrix(x)
				fm %*% x + multiply.left.diag(cd, x)
			}
	}
	else if (which == "L") {
		d <- get.degree(fm)
		# We compute the largest eigenvalues and then convert them to
		# the smallest eigenvalues. It's easier to compute the largest eigenvalues.
		multiply <- function(x, extra) fm %*% x
		comp.oppo <- TRUE
	}
	else if (which == "nL") {
		d <- 1/sqrt(get.degree(fm))
		# We compute the largest eigenvalues and then convert them to
		# the smallest eigenvalues. It's easier to compute the largest eigenvalues.
		multiply <- function(x, extra) {
			x <- fm.as.matrix(x)
			multiply.left.diag(d, fm %*% multiply.left.diag(d, x))
		}
		comp.oppo <- TRUE
	}
	else {
		print("wrong option")
		stopifnot(FALSE)
	}

	ret <- fm.eigen(multiply, k=nev, n=nrow(fm), which="LM", sym=TRUE,
					options=list(block_size=1, num_blocks=ncv, tol=tol))
	rescale <- function(x) {
		scal <- sqrt(colSums(x * x))
		x <- fm.mapply.row(x, scal, fm.bo.div)
		x <- fm.materialize(x)
	}
	if (directed) {
		ret$values <- sqrt(ret$values)
		ret[["left"]] <- ret$vectors
		ret[["right"]] <- rescale(multiply.right(ret$vectors))
		ret$vectors <- NULL
	}
	else if (comp.oppo) {
		if (which == "nL")
			ret$values <- 1 - ret$values
		# We can't compute the eigenvalues of the Laplacian matrix.
		else
			ret$values[1:length(ret$values)] <- 0
	}

    tmp <- proc.time() - ptm
    cat("It takes ", as.numeric(tmp[3]), " seconds to embed the graph.\n")

	ret
}

oocase.interface <- function(fg, dmax=2) {
#    fg <- fg.load.graph(input, directed=FALSE)
    fg

    cc <- fg.clusters(fg, mode="weak")
    tcc <- fm.table(cc)
    max.idx <- which(as.vector(tcc@Freq == max(tcc@Freq)))
    lccV <- which(as.vector(cc == tcc@val[max.idx]))

    lcc <- fg.fetch.subgraph(fg, vertices=lccV, compress=TRUE)
    lcc

    m <- fg.get.sparse.matrix(lcc)
    res <- fm.spectral.embedding(m, dmax, which="Aug", tol=1e-8)
#    res$values
    return(res)

    ## cat("The output files are saved in 'DATA/out_vectors.txt' and 'DATA/out_values.txt'.\n")
    ## write.table(as.matrix(res$vectors),"/home/user/D3M/DATA/out_vectors.txt", col.names=F, row.names=F)
    ## write(as.vector(res$values),"/home/user/D3M/DATA/eigenvalues.txt", ncol=1)
}
