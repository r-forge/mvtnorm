
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
