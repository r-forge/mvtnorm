# $Id$

rmvnorm <- function(n, mean = rep(0, nrow(sigma)), sigma = diag(length(mean)),
                    method=c("eigen", "svd", "chol"), pre0.9_9994 = FALSE, checkSymmetry = TRUE)
{

    if (checkSymmetry && !isSymmetric(sigma, tol = sqrt(.Machine$double.eps),
                     check.attributes = FALSE)) {
        stop("sigma must be a symmetric matrix")
    }
    if (length(mean) != nrow(sigma))
        stop("mean and sigma have non-conforming size")

    method <- match.arg(method)

    R <- if(method == "eigen") {
        ev <- eigen(sigma, symmetric = TRUE)
        if (!all(ev$values >= -sqrt(.Machine$double.eps) * abs(ev$values[1]))){
            warning("sigma is numerically not positive semidefinite")
        }
        ## ev$vectors %*% diag(sqrt(ev$values), length(ev$values)) %*% t(ev$vectors)
        ## faster for large  nrow(sigma):
        t(ev$vectors %*% (t(ev$vectors) * sqrt(pmax(ev$values, 0))))
    }
    else if(method == "svd"){
        s. <- svd(sigma)
        if (!all(s.$d >= -sqrt(.Machine$double.eps) * abs(s.$d[1]))){
            warning("sigma is numerically not positive semidefinite")
        }
        t(s.$v %*% (t(s.$u) * sqrt(pmax(s.$d, 0))))
    }
    else if(method == "chol"){
        R <- chol(sigma, pivot = TRUE)
        R[, order(attr(R, "pivot"))]
    }

    retval <- matrix(rnorm(n * ncol(sigma)), nrow = n, byrow = !pre0.9_9994) %*%  R
    retval <- sweep(retval, 2, mean, "+")
    colnames(retval) <- names(mean)
    retval
}

### allow matrix arguments x and also mean
.xm <- function(x, mean, p, n) {

    if (!is.matrix(x)) x <- matrix(x, ncol = length(x))
    nr <- nrow(x)
    nc <- ncol(x)
    if (nc != n)
        stop("x and (inv)chol have non-conforming size")
    if (nr != p)
        stop("x and (inv)chol have non-conforming size")
    if (identical(unique(mean), 0)) return(x)
    if (length(mean) == p) 
        return(x - mean)
    if (!is.matrix(mean))
        stop("x and mean have non-conforming size")
    if (nrow(mean) != nr)
        stop("x and mean have non-conforming size")
    if (ncol(mean) != nc)
        stop("x and mean have non-conforming size")
    return(x - mean)
}


dmvnorm <- function (x, mean = rep(0, p), sigma = diag(p), log = FALSE, checkSymmetry = TRUE, 
                     chol, invchol) ### allow multiple covariance matrices specified as
                                    ### chol = chol(sigma) or
                                    ### invchol = solve(chol(sigma))
{

    if (is.vector(x))
        x <- matrix(x, ncol = length(x))
    p <- ncol(x)

    if (missing(chol) && missing(invchol)) {
        ### "old" code
        if(!missing(mean)) {
            if(!is.null(dim(mean))) dim(mean) <- NULL
            if (length(mean) != p)
                stop("x and mean have non-conforming size")
        }
        if(!missing(sigma)) {
            if (p != ncol(sigma))
                stop("x and sigma have non-conforming size")
            if (checkSymmetry && !isSymmetric(sigma, tol = sqrt(.Machine$double.eps),
                             check.attributes = FALSE))
                stop("sigma must be a symmetric matrix")
        }

        ## <faster code contributed by Matteo Fasiolo mf364 at bath.ac.uk
        dec <- tryCatch(base::chol(sigma), error=function(e)e)
        if (inherits(dec, "error")) {
            ## warning("cannot compute chol(sigma)"); return(NaN)
            ## behave the same as dnorm(): return Inf or 0
            x.is.mu <- colSums(t(x) != mean) == 0
            logretval <- rep.int(-Inf, nrow(x))
            logretval[x.is.mu] <- Inf # and all other f(.) == 0
        } else {
            tmp <- backsolve(dec, t(x) - mean, transpose = TRUE)
            rss <- colSums(tmp ^ 2)
            logretval <- - sum(log(diag(dec))) - 0.5 * p * log(2 * pi) - 0.5 * rss
        }
    } else {
        ## potentially one covariance per row of x
        ## mean might be a matrix

        if (missing(chol)) {
            if (missing(invchol))
                stop("either chol or invchol must be given")
            ## invchol is given
            if (!inherits(invchol, "ltMatrices"))
                stop("invchol is not an object of class ltMatrices")
            N <- dim(invchol)[1L]
            N <- ifelse(N == 1, p, N)
            J <- dim(invchol)[2L]
            x <- .xm(x = x, mean = mean, p = J, n = N)
            ## use dnorm (gets the normalizing factors right)
            ## NOTE: x is (p x N) when (inv)chol are given
            logretval <- colSums(dnorm(Mult(invchol, x), log = TRUE))
            ## note that the second summand gets recycled the correct number
            ## of times in case dim(invchol)[1L] == 1 but ncol(x) > 1
            if (attr(invchol, "diag"))
                logretval <- logretval + colSums(log(diagonals(invchol)))
        } else {
            if (missing(chol))
                stop("either chol or invchol must be given")
            ## chol is given
            if (!inherits(chol, "ltMatrices"))
                stop("chol is not an object of class ltMatrices")
            N <- dim(chol)[1L]
            N <- ifelse(N == 1, p, N)
            J <- dim(chol)[2L]
            x <- .xm(x = x, mean = mean, p = J, n = N)
            logretval <- colSums(dnorm(solve(chol, x), log = TRUE))
            if (attr(chol, "diag"))
                logretval <- logretval - colSums(log(diagonals(chol)))
        }
    }
    names(logretval) <- rownames(x)
    if(log) logretval else exp(logretval)
}

sldmvnorm <- function(x, mean = 0, chol, invchol) {

    stopifnot(xor(missing(chol), missing(invchol)))

    if (!missing(invchol)) {

        N <- dim(invchol)[1L]
        N <- ifelse(N == 1, ncol(x), N)
        J <- dim(invchol)[2L]
        x <- .xm(x = x, mean = mean, p = J, n = N)

        Mix <- Mult(invchol, x)
        sx <- - Mult(invchol, Mix, transpose = TRUE)

        Y <- matrix(x, byrow = TRUE, nrow = J, ncol = N * J)
        ret <- - matrix(Mix[, rep(1:N, each = J)] * Y, ncol = N)

        M <- matrix(1:(J^2), nrow = J, byrow = FALSE)
        ret <- ltMatrices(ret[M[lower.tri(M, diag = attr(invchol, "diag"))],,drop = FALSE], 
                          diag = attr(invchol, "diag"), byrow = FALSE)
        ret <- ltMatrices(ret, 
                          diag = attr(invchol, "diag"), byrow = attr(invchol, "byrow"))
        if (attr(invchol, "diag")) {
            ### recycle properly
            diagonals(ret) <- diagonals(ret) + c(1 / diagonals(invchol))
        } else {
            diagonals(ret) <- 0
        }
        return(list(x = sx, invchol = ret))
    }

    invchol <- solve(chol)
    ret <- sldmvnorm(x = x, mean = mean, invchol = invchol)
    ### this means: ret$chol <- - vectrick(invchol, ret$invchol, invchol)
    ret$chol <- - vectrick(invchol, ret$invchol)
    ret$invchol <- NULL
    return(ret)
}
