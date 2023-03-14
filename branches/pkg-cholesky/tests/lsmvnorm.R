
library("mvtnorm")
library("numDeriv")

set.seed(290875)

chk <- function(...) stopifnot(isTRUE(all.equal(...)))

J <- 5

chks <- function(dg, tol = .Machine$double.eps^(1 / 4)) {

    prm <- runif(J * (J + c(-1, 1)[dg + 1L]) / 2)
    L <- ltMatrices(prm, diag = dg)
    a <- matrix(-2, nrow = J, ncol = 1)
    b <- matrix( 2, nrow = J, ncol = 1)
    M <- 10000L

    l <- function(x) {
        x <- ltMatrices(x, diag = dg)
        lmvnorm(a, b, chol = x, M = M, seed = 29)
    }

    s <- function(x)
        smvnorm(a, b, chol = x, M = M, seed = 29)

    rl <- l(L)
    rs <- s(L)
    chk(rl, rs$logLik)
    chk(rs$chol, ltMatrices(grad(l, unclass(L)), diag = dg), tol = tol)

    l <- function(x) {
        x <- ltMatrices(x, diag = dg)
        lmvnorm(a, b, invchol = x, M = M, seed = 29)
    }

    s <- function(x)
        smvnorm(a, b, invchol = x, M = M, seed = 29)

    rl <- l(L)
    rs <- s(L)
    chk(rl, rs$logLik)
    chk(rs$invchol, ltMatrices(grad(l, unclass(L)), diag = dg), tol = tol)

    l <- function(x)
        lmvnorm(a, b, mean = x, chol = L, M = M, seed = 29)
    s <- function(x)
        smvnorm(a, b, mean = x, chol = L, M = M, seed = 29)

    x <- numeric(J)
    rl <- l(x)
    rs <- s(x)
    chk(rl, rs$logLik)
    chk(grad(l, x), c(rs$mean), tol = tol)
    x <- 1:J
    rl <- l(x)
    rs <- s(x)
    chk(rl, rs$logLik)
    chk(grad(l, x), c(rs$mean), tol = tol)
}

chks(TRUE)
chks(FALSE)


.cmvnorm <- function(invchol, which_given, given) {
    L <- invchol
    J <- dim(L)[2L]
    tmp <- matrix(0, ncol = ncol(given), nrow = J - length(which_given))
    center <- Mult(L, rbind(given, tmp))
    center <- center[-which_given,,drop = FALSE]
    L <- L[,-which_given]
    return(list(center = center, invchol = L))
}

J <- (cJ <- 5) + (dJ <- 5)
N <- 3
M <- 10
ltM <- function(x) ltMatrices(x, diag = FALSE, byrow = TRUE, trans = TRUE)
prm <- matrix(runif(J * (J - 1) / 2 * N), ncol = N)
L <- ltM(prm)

obs <- matrix(rnorm(J * N), ncol = N)
lwr <- -abs(obs)
upr <- abs(obs)

w <- matrix(runif((dJ - 1) * N), ncol = N)

j <- 1:cJ
ll <- function(x) {
    LD <- ltMatrices(x, diag = FALSE, byrow = TRUE, trans = TRUE)
    cd <- .cmvnorm(invchol = LD, which = j, given = obs[j,,drop = FALSE])
    lmvnorm(lwr[-j,], upr[-j,], center = cd$center, 
            invchol = cd$invchol, w = w)
}

ll(L)

a <- ltMatrices(matrix(grad(ll, unclass(L)), ncol = N), diag = FALSE, byrow =
TRUE, trans = TRUE)

cd <- .cmvnorm(invchol = L, which = j, given = obs[j,,drop = FALSE])
b <- smvnorm(lwr[-j,], upr[-j,], center = cd$center, invchol = cd$invchol, 
        w = w)$invchol

all.equal(a[,-j], b, check.attributes = FALSE)

ll <- function(x) {
    LD <- ltMatrices(x, diag = TRUE, byrow = TRUE, trans = TRUE)
    cd <- .cmvnorm(invchol = LD, which = j, given = obs[j,,drop = FALSE])
    lmvnorm(lwr[-j,], upr[-j,], center = cd$center, 
            invchol = cd$invchol, w = w)
}

LD <- invcholD(L)
ll(LD)

a <- ltMatrices(matrix(grad(ll, unclass(LD)), ncol = N), diag = TRUE, byrow =
TRUE, trans = TRUE)

cd <- .cmvnorm(invchol = LD, which = j, given = obs[j,,drop = FALSE])
b <- smvnorm(lwr[-j,], upr[-j,], center = cd$center, invchol = cd$invchol,
        w = w)$invchol

all.equal(a[,-j], b, check.attributes = FALSE)

a[,-j] - b

