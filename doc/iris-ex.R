### R code from vignette source 'lmvnorm_src.Rnw'

###################################################
### code chunk number 1: mvtnorm-citation
###################################################
year <- substr(packageDescription("mvtnorm")$Date, 1, 4)
version <- packageDescription("mvtnorm")$Version


###################################################
### code chunk number 2: digits
###################################################
options(digits = 4)


###################################################
### code chunk number 3: chk
###################################################
chk <- function(...) stopifnot(isTRUE(all.equal(...)))


###################################################
### code chunk number 4: example
###################################################
library("mvtnorm")
library("numDeriv")
library("tram")
set.seed(290875)

###################################################
### code chunk number 88: gc-classical
###################################################
data("iris", package = "datasets")
N <- nrow(iris)
J <- 4
Jd <- J * (J + 1) / 2
Jo <- J * (J - 1) / 2
j <- seq_len(J)
(vn <- colnames(iris)[j])
Y <- as.matrix(iris[, vn])

## Y ~ N(mu, Sigma)
-sum(dmvnorm(Y, mean = colMeans(Y), sigma = var(Y) * (N - 1) / N, log = TRUE))

nll <- function(parm, object = FALSE, ...) {
    L <- ltMatrices(parm[-j], names = vn, diag = TRUE)
    obj <- mvnorm(invcholmean = parm[j], invchol = L)
    if (object) return(obj)
    - logLik(obj, ...)
}

nsc <- function(parm, ...) {
    obj <- nll(parm, object = TRUE)
    ret <- lLgrad(obj, ...)
    - c(rowSums(ret$invcholmean),
        rowSums(Lower_tri(ret$scale, diag = TRUE)))
}
    
X <- cbind(1, Y)
XtX <- crossprod(X)    

cf <- lapply(j, function(i) {
    si <- seq_len(i)
    cf <- solve(XtX[si,si], XtX[si,i+1])
    sde <- mean((X[,i+1] - X %*% c(cf, rep(0, J + 1 - i)))^2)
    return(c(-cf, 1) / sqrt(sde))
})

nu <- sapply(cf, function(x) - x[1L])
L <- ltMatrices(do.call("c", lapply(cf, function(x) x[-1L])), diag = TRUE, 
                names = vn, byrow = TRUE)
start <- c(nu, Lower_tri(ltMatrices(L, byrow = FALSE), diag = TRUE))

nll(start, obs = t(Y))
grad(nll, start, obs = t(Y))
nsc(start, obs = t(Y))

lower <- numeric(J + Jd)
lower[-(J + diagonals(ltMatrices(seq_len(Jd), diag = TRUE)))] <- -Inf

(op0 <- optim(par = start, fn = nll, gr = nsc, lower = lower, 
             method = "L-BFGS-B", obs = t(Y), control = list(maxit = 1000)))
nll(op0$par, object = TRUE)

Z <- t(d <- qnorm(do.call("cbind", lapply(iris[1:J], rank, ties.method = "max")) / 
       (N + 1)))
d <- as.data.frame(d)

(op1 <- optim(par = start, fn = nll, gr = nsc, lower = lower, 
             method = "L-BFGS-B", obs = Z, control = list(maxit = 1000)))
obj <- nll(op1$par, object = TRUE)
obj$invcholmean
invchol2cov(obj$scale)


nll <- function(parm, object = FALSE, ...) {
    L <- ltMatrices(parm, names = vn)
    obj <- mvnorm(invchol = L)
    if (object) return(obj)
    - logLik(obj, standardize = TRUE, ...)
}

nsc <- function(parm, ...) {
    obj <- nll(parm, object = TRUE)
    ret <- lLgrad(obj, standardize = TRUE, ...)
    - rowSums(Lower_tri(ret$scale, diag = FALSE))
}

start <- Lower_tri(obj$scale, diag = FALSE)
x1 <- cbind(grad(nll, start, obs = Z), nsc(start, obs = Z))
cor(x1) 

(op1 <- optim(start, fn = nll, gr = nsc, method = "BFGS", obs = Z))

invchol2cov(standardize(invchol = nll(op1$par, object = TRUE)$scale))



lwr <- do.call("cbind", lapply(iris[1:J], rank, ties.method = "min")) - 1L
upr <- do.call("cbind", lapply(iris[1:J], rank, ties.method = "max"))
lwr <- t(qnorm(lwr / N))
upr <- t(qnorm(upr / N))

M <- 500 
if (require("qrng", quietly = TRUE)) {
    ### quasi-Monte-Carlo
    W <- t(ghalton(M, d = J - 1))
} else {
    ### Monte-Carlo
    W <- matrix(runif(M * (J - 1)), nrow = J - 1, byrow = TRUE)
}

x2 <- cbind(grad(nll, start, lower = lwr, upper = upr, M = M, w = W), 
            nsc(start, lower = lwr, upper = upr, M = M, w = W))
cor(x2)

(op2 <- optim(op1$par, 
             fn = function(parm) nll(parm, lower = lwr, upper = upr, M = M, w = W), 
             gr = function(parm) nsc(parm, lower = lwr, upper = upr, M = M, w = W), 
             method = "BFGS"))

invchol2cov(standardize(invchol = nll(op2$par, object = TRUE)$scale))

### mixed likelihoods

v1 <- vn[1:2]
v2 <- vn[-(1:2)]
(op2a <- optim(op1$par, 
             fn = function(parm) nll(parm, obs = Z[v1,], lower = lwr[v2,], upper = upr[v2,], M = M, w = W[1,,drop = FALSE]), 
             gr = function(parm) nsc(parm, obs = Z[v1,], lower = lwr[v2,], upper = upr[v2,], M = M, w = W[1,,drop = FALSE]), 
             method = "BFGS"))

invchol2cov(standardize(invchol = nll(op2a$par, object = TRUE)$scale))



idx <- lapply(iris[1:J], function(x)
    factor(rank(x, ties.method = "max")))

h <- function(x)
    qnorm(cumsum(prop.table(table(x))[-nlevels(x)]))

start0 <- do.call("c", lapply(idx, h))
start <- c(start0, op2$par)

i <- rep(gl(J + 1, 1, labels = c(vn, "Lambda")), times = c(sapply(idx, nlevels) - 1L, Jo))

pp <- function(parm) {
    sparm <- split(parm, i)
    chk <- sapply(sparm[vn], function(x) 
        any(diff(x) < 0))
    if (any(chk)) return(Inf)
    lwr <- do.call("rbind", lapply(seq_len(J), function(j)
        c(-Inf, sparm[[j]])[idx[[j]]]
    ))
    upr <- do.call("rbind", lapply(seq_len(J), function(j)
        c(sparm[[j]], Inf)[idx[[j]]]
    ))
    rownames(upr) <- rownames(lwr) <- vn
    c(sparm, list(lower = lwr, upper = upr))
}


nll2 <- function(parm, ...) {
    p <- pp(parm)
    if (length(p) == 1L) return(p)
    nll(p$Lambda, lower = p$lower, upper = p$upper, ...)
}

nsc2 <- function(parm, ...) {
    p <- pp(parm)
    obj <- nll(p$Lambda, object = TRUE)
    ret <- lLgrad(obj, standardize = TRUE, lower = p$lower, upper = p$upper, ...)
    slwr <- do.call("c", lapply(seq_len(J), function(j) {
        ret <- tapply(ret$lower[j, ], idx[[j]], sum)
        return(ret[-1L])
    }))
    supr <- do.call("c", lapply(seq_len(J), function(j) {
        ret <- tapply(ret$upper[j, ], idx[[j]], sum)
        return(ret[-length(ret)])
    }))
    - c(slwr + supr, 
        rowSums(Lower_tri(ret$scale, diag = FALSE))[seq_along(p$Lambda)])
}

nll2(start, M = M, w = W)
x3 <- cbind(grad(nll2, start, M = M, w = W), nsc2(start, M = M, w = W))
cor(x3)

op3 <- optim(start, 
             fn = function(parm) nll2(parm, M = M, w = W), 
             gr = function(parm) nsc2(parm, M = M, w = W),
             method = "BFGS")

### with marginal effects
i <- rep(gl(J + 2L, 1, labels = c(vn, "shift", "Lambda")), 
         times = c(sapply(idx, nlevels) - 1L, 2L * J, Jo))

pp <- function(parm) {
    sparm <- split(parm, i)
    chk <- sapply(sparm[vn], function(x) 
        any(diff(x) < 0))
    if (any(chk)) return(Inf)
    lwr <- do.call("rbind", lapply(seq_len(J), function(j)
        c(-Inf, sparm[[j]])[idx[[j]]] - c(0, matrix(sparm$shift, ncol = 2, byrow = TRUE)[j,])[iris$Species])
    )
    upr <- do.call("rbind", lapply(seq_len(J), function(j)
        c(sparm[[j]], Inf)[idx[[j]]] - c(0, matrix(sparm$shift, ncol = 2, byrow = TRUE)[j,])[iris$Species])
    )
    rownames(upr) <- rownames(lwr) <- vn
    c(sparm, list(lower = lwr, upper = upr))
}


nsc3 <- function(parm, ...) {
    p <- pp(parm)
    obj <- nll(p$Lambda, object = TRUE)
    ret <- lLgrad(obj, standardize = TRUE, lower = p$lower, upper = p$upper, ...)
    slwr <- do.call("c", lapply(seq_len(J), function(j) {
        ret <- tapply(ret$lower[j, ], idx[[j]], sum)
        return(ret[-1L])
    }))
    supr <- do.call("c", lapply(seq_len(J), function(j) {
        ret <- tapply(ret$upper[j, ], idx[[j]], sum)
        return(ret[-length(ret)])
    }))
    blwr <- do.call("c", lapply(seq_len(J), function(j) {
        ret <- tapply(ret$lower[j, ], iris$Species, sum)
        return(ret[-1L])
    }))
    bupr <- do.call("c", lapply(seq_len(J), function(j) {
        ret <- tapply(ret$upper[j, ], iris$Species, sum)
        return(ret[-1L])
    }))
    - c(slwr + supr, - (blwr + bupr),
        rowSums(Lower_tri(ret$scale, diag = FALSE))[seq_along(p$Lambda)])
}


start <- c(start0, numeric(2 * J), op2$par)


nll2(start, M = M, w = W)
x4 <- cbind(grad(nll2, start, M = M, w = W),
      nsc3(start, M = M, w = W))
cor(x4)


op4 <- optim(start, 
             fn = function(parm) nll2(parm, M = M, w = W), 
             gr = function(parm) nsc3(parm, M = M, w = W),
             method = "BFGS")


### SEM
n0 <- rep(0, J * (J - 1) / 2)
nll <- function(parm, object = FALSE, ...) {
    L <- ltMatrices(c(parm, n0), names = c("Latent", vn))
    obj <- mvnorm(invchol = L)
    if (object) return(obj)
    - logLik(obj, standardize = TRUE, ...)
}

i <- rep(gl(J + 2L, 1, labels = c(vn, "shift", "Lambda")), times = c(sapply(idx, nlevels) - 1L, 2 * J, J))
start <- c(start0, numeric(2 * J), numeric(J))


nll2(start, M = M, w = W)
x5 <- cbind(grad(nll2, start, M = M, w = W),
      nsc3(start, M = M, w = W))
cor(x5)


op5 <- optim(start, 
             fn = function(parm) nll2(parm, M = M, w = W), 
             gr = function(parm) nsc3(parm, M = M, w = W),
             method = "BFGS")

pchisq(-2 * (op4$value - op3$value), df = length(n0), lower.tail = FALSE)

