
library("mvtnorm")

set.seed(29)

chk <- function(...) stopifnot(isTRUE(all.equal(..., check.attributes = FALSE)))

### N samples with N different covariance matrices
N <- 10
J <- 5
dg <- TRUE
br <- FALSE
lt <- ltMatrices(matrix(runif(N * J * (J + c(-1, 1)[dg + 1L]) / 2) + 1, ncol = N), 
                 diag = dg)
lt <- ltMatrices(lt, diag = dg, byrow = br)
Z <- matrix(rnorm(N * J), ncol = N)
Y <- solve(lt, Z)
ll1 <- sum(dnorm(Mult(lt, Y), log = TRUE)) + sum(log(diagonals(lt)))

S <- as.array(Tcrossprod(solve(lt)))
ll2 <- sum(l2 <- sapply(1:N, function(i) mvtnorm:::dmvnorm(x = Y[,i], sigma = S[,,i], log = TRUE)))
chk(ll1, ll2)

l3 <- dmvnorm(x = Y, invchol = lt, log = TRUE)
l4 <- dmvnorm(x = Y, chol = solve(lt), log = TRUE)

chk(l2, l3)
chk(l2, l4)

ll1 <- sum(dnorm(Mult(lt[1,], Y), log = TRUE)) + N * sum(log(diagonals(lt[1,])))

S <- as.array(Tcrossprod(solve(lt)))
ll2 <- sum(l2 <- sapply(1:N, function(i) mvtnorm:::dmvnorm(x = Y[,i], sigma = S[,,1], log = TRUE)))
chk(ll1, ll2)

l3 <- dmvnorm(x = Y, invchol = lt[1,], log = TRUE)
l4 <- dmvnorm(x = Y, chol = solve(lt[1,]), log = TRUE)

chk(l2, l3)
chk(l2, l4)

### check scores
if (require("numDeriv")) {

  f <- function(L) {
      L <- ltMatrices(L, diag = dg, byrow = br)
      sum(dmvnorm(x = Y, invchol = L, log = TRUE))
  }

  s0 <- grad(f, unclass(lt))
  s1 <- sldmvnorm(x = Y, invchol = lt)

  chk(Lower_tri(ltMatrices(matrix(s0, ncol = N), diag = dg, byrow = br), diag = dg), 
      Lower_tri(s1$invchol, diag = dg))

  f <- function(L) {
      L <- ltMatrices(L, diag = dg, byrow = br)
      sum(dmvnorm(x = Y, chol = L, log = TRUE))
  }

  s0 <- grad(f, unclass(lt))
  s1 <- sldmvnorm(x = Y, chol = lt)

  chk(Lower_tri(ltMatrices(matrix(s0, ncol = N), diag = dg, byrow = br), diag = dg), 
      Lower_tri(s1$chol, diag = dg))

  f <- function(x)
    sum(dmvnorm(x = x, invchol = lt, log = TRUE))

  s0 <- grad(f, Y)
  s1 <- sldmvnorm(x = Y, invchol = lt)

  chk(matrix(s0, ncol = N), s1$x)
}

dg <- FALSE
br <- FALSE
lt <- ltMatrices(matrix(runif(N * J * (J + c(-1, 1)[dg + 1L]) / 2) + 1, ncol = N), 
                 diag = dg)
lt <- ltMatrices(lt, diag = dg, byrow = br)
Z <- matrix(rnorm(N * J), ncol = N)
Y <- solve(lt, Z)
ll1 <- sum(dnorm(Mult(lt, Y), log = TRUE)) + sum(log(diagonals(lt)))

S <- as.array(Tcrossprod(solve(lt)))
ll2 <- sum(l2 <- sapply(1:N, function(i) mvtnorm:::dmvnorm(x = Y[,i], sigma = S[,,i], log = TRUE)))
chk(ll1, ll2)

l3 <- dmvnorm(x = Y, invchol = lt, log = TRUE)
l4 <- dmvnorm(x = Y, chol = solve(lt), log = TRUE)

chk(l2, l3)
chk(l2, l4)

ll1 <- sum(dnorm(Mult(lt[1,], Y), log = TRUE)) + N * sum(log(diagonals(lt[1,])))

S <- as.array(Tcrossprod(solve(lt)))
ll2 <- sum(l2 <- sapply(1:N, function(i) mvtnorm:::dmvnorm(x = Y[,i], sigma = S[,,1], log = TRUE)))
chk(ll1, ll2)

l3 <- dmvnorm(x = Y, invchol = lt[1,], log = TRUE)
l4 <- dmvnorm(x = Y, chol = solve(lt[1,]), log = TRUE)

chk(l2, l3)
chk(l2, l4)

### check scores
if (require("numDeriv")) {

  f <- function(L) {
      L <- ltMatrices(L, diag = dg, byrow = br)
      sum(dmvnorm(x = Y, invchol = L, log = TRUE))
  }

  s0 <- grad(f, unclass(lt))
  s1 <- sldmvnorm(x = Y, invchol = lt)

  chk(Lower_tri(ltMatrices(matrix(s0, ncol = N), diag = dg, byrow = br), diag = dg), 
      Lower_tri(s1$invchol, diag = dg))

  f <- function(L) {
      L <- ltMatrices(L, diag = dg, byrow = br)
      sum(dmvnorm(x = Y, chol = L, log = TRUE))
  }

  s0 <- grad(f, unclass(lt))
  s1 <- sldmvnorm(x = Y, chol = lt)

  chk(Lower_tri(ltMatrices(matrix(s0, ncol = N), diag = dg, byrow = br), diag = dg), 
      Lower_tri(s1$chol, diag = dg))

  f <- function(x)
    sum(dmvnorm(x = x, invchol = lt, log = TRUE))

  s0 <- grad(f, Y)
  s1 <- sldmvnorm(x = Y, invchol = lt)

  chk(matrix(s0, ncol = N), s1$x)
}

dg <- FALSE
br <- TRUE
lt <- ltMatrices(matrix(runif(N * J * (J + c(-1, 1)[dg + 1L]) / 2) + 1, ncol = N), 
                 diag = dg)
lt <- ltMatrices(lt, diag = dg, byrow = br)
Z <- matrix(rnorm(N * J), ncol = N)
Y <- solve(lt, Z)
ll1 <- sum(dnorm(Mult(lt, Y), log = TRUE)) + sum(log(diagonals(lt)))

S <- as.array(Tcrossprod(solve(lt)))
ll2 <- sum(l2 <- sapply(1:N, function(i) mvtnorm:::dmvnorm(x = Y[,i], sigma = S[,,i], log = TRUE)))
chk(ll1, ll2)

l3 <- dmvnorm(x = Y, invchol = lt, log = TRUE)
l4 <- dmvnorm(x = Y, chol = solve(lt), log = TRUE)

chk(l2, l3)
chk(l2, l4)

ll1 <- sum(dnorm(Mult(lt[1,], Y), log = TRUE)) + N * sum(log(diagonals(lt[1,])))

S <- as.array(Tcrossprod(solve(lt)))
ll2 <- sum(l2 <- sapply(1:N, function(i) mvtnorm:::dmvnorm(x = Y[,i], sigma = S[,,1], log = TRUE)))
chk(ll1, ll2)

l3 <- dmvnorm(x = Y, invchol = lt[1,], log = TRUE)
l4 <- dmvnorm(x = Y, chol = solve(lt[1,]), log = TRUE)

chk(l2, l3)
chk(l2, l4)

### check scores
if (require("numDeriv")) {

  f <- function(L) {
      L <- ltMatrices(L, diag = dg, byrow = br)
      sum(dmvnorm(x = Y, invchol = L, log = TRUE))
  }

  s0 <- grad(f, unclass(lt))
  s1 <- sldmvnorm(x = Y, invchol = lt)

  chk(Lower_tri(ltMatrices(matrix(s0, ncol = N), diag = dg, byrow = br), diag = dg), 
      Lower_tri(s1$invchol, diag = dg))

  f <- function(L) {
      L <- ltMatrices(L, diag = dg, byrow = br)
      sum(dmvnorm(x = Y, chol = L, log = TRUE))
  }

  s0 <- grad(f, unclass(lt))
  s1 <- sldmvnorm(x = Y, chol = lt)

  chk(Lower_tri(ltMatrices(matrix(s0, ncol = N), diag = dg, byrow = br), diag = dg), 
      Lower_tri(s1$chol, diag = dg))

  f <- function(x)
    sum(dmvnorm(x = x, invchol = lt, log = TRUE))

  s0 <- grad(f, Y)
  s1 <- sldmvnorm(x = Y, invchol = lt)

  chk(matrix(s0, ncol = N), s1$x)
}

dg <- FALSE
br <- FALSE
lt <- ltMatrices(matrix(runif(N * J * (J + c(-1, 1)[dg + 1L]) / 2) + 1, ncol = N), 
                 diag = dg)
lt <- ltMatrices(lt, diag = dg, byrow = br)
Z <- matrix(rnorm(N * J), ncol = N)
Y <- solve(lt, Z)
ll1 <- sum(dnorm(Mult(lt, Y), log = TRUE)) + sum(log(diagonals(lt)))

S <- as.array(Tcrossprod(solve(lt)))
ll2 <- sum(l2 <- sapply(1:N, function(i) mvtnorm:::dmvnorm(x = Y[,i], sigma = S[,,i], log = TRUE)))
chk(ll1, ll2)

l3 <- dmvnorm(x = Y, invchol = lt, log = TRUE)
l4 <- dmvnorm(x = Y, chol = solve(lt), log = TRUE)

chk(l2, l3)
chk(l2, l4)

ll1 <- sum(dnorm(Mult(lt[1,], Y), log = TRUE)) + N * sum(log(diagonals(lt[1,])))

S <- as.array(Tcrossprod(solve(lt)))
ll2 <- sum(l2 <- sapply(1:N, function(i) mvtnorm:::dmvnorm(x = Y[,i], sigma = S[,,1], log = TRUE)))
chk(ll1, ll2)

l3 <- dmvnorm(x = Y, invchol = lt[1,], log = TRUE)
l4 <- dmvnorm(x = Y, chol = solve(lt[1,]), log = TRUE)

chk(l2, l3)
chk(l2, l4)

### check scores
if (require("numDeriv")) {

  f <- function(L) {
      L <- ltMatrices(L, diag = dg, byrow = br)
      sum(dmvnorm(x = Y, invchol = L, log = TRUE))
  }

  s0 <- grad(f, unclass(lt))
  s1 <- sldmvnorm(x = Y, invchol = lt)

  chk(Lower_tri(ltMatrices(matrix(s0, ncol = N), diag = dg, byrow = br), diag = dg), 
      Lower_tri(s1$invchol, diag = dg))

  f <- function(L) {
      L <- ltMatrices(L, diag = dg, byrow = br)
      sum(dmvnorm(x = Y, chol = L, log = TRUE))
  }

  s0 <- grad(f, unclass(lt))
  s1 <- sldmvnorm(x = Y, chol = lt)

  chk(Lower_tri(ltMatrices(matrix(s0, ncol = N), diag = dg, byrow = br), diag = dg), 
      Lower_tri(s1$chol, diag = dg))

  f <- function(x)
    sum(dmvnorm(x = x, invchol = lt, log = TRUE))

  s0 <- grad(f, Y)
  s1 <- sldmvnorm(x = Y, invchol = lt)

  chk(matrix(s0, ncol = N), s1$x)
}

