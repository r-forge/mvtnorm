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
N <- table(iris$Species)

### unconditional models for one species 
setosa <- subset(iris, Species == "setosa")

J <- 4
Jd <- J * (J + 1) / 2
Jo <- J * (J - 1) / 2
j <- seq_len(J)
(vn <- colnames(iris)[j])
Y <- as.matrix(setosa[, vn])
mY <- colMeans(Y)
sY <- var(Y) * (N["setosa"] - 1) / N["setosa"]

## Y ~ N(mu, Sigma)
-sum(dmvnorm(Y, mean = mY, sigma = sY, log = TRUE))

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

### construct starting values from conditional distributions    
X <- cbind(1, Y)
XtX <- crossprod(X)    

cf <- lapply(j, function(i) {
    si <- seq_len(i)
    ### Y_j = cf[1] * Y_1 + ... + cf[j-1] * Y_{j - 1} + sigma eps
    cf <- solve(XtX[si,si], XtX[si,i+1])
    ### sigma
    sde <- mean((X[,i+1] - X %*% c(cf, rep(0, J + 1 - i)))^2)
    ### negative standardized coefficients
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

### theory is correct
chk(start, op0$par)

### interval censoring
idx <- lapply(setosa[j], factor)
suy <- lapply(setosa[j], function(y) sort(unique(y)))

l <- lapply(j, function(i) c(-Inf, suy[[i]])[idx[[i]]])
u <- lapply(j, function(i) c(suy[[i]], Inf)[unclass(idx[[i]]) + 1L])

lm <- t(do.call("cbind", l))
um <- t(do.call("cbind", u))
rownames(lm) <- rownames(um) <- vn


M <- 1000 
if (require("qrng", quietly = TRUE)) {
    ### quasi-Monte-Carlo
    W <- t(ghalton(M, d = J - 1))
} else {
    ### Monte-Carlo
    W <- matrix(runif(M * (J - 1)), nrow = J - 1, byrow = TRUE)
}

system.time(lp1 <- - nll(start, lower = lm, upper = um, M = M, w = W, logLik = FALSE))

a <- GenzBretz(maxpts = M, abseps = 0, releps = 0)
system.time(prb <- sapply(seq_len(ncol(lm)), function(i)
    pmvnorm(lower = lm[,i], upper = um[,i], mean = mY, sigma = sY, algorithm = a)))

lp2 <- log(prb)
all.equal(lp1, lp2)

-sum(lp1)
-sum(lp2)

# however
nll(start, lower = lm, upper = um, M = M, w = W)
system.time(gr1 <- grad(nll, start, lower = lm, upper = um, M = M, w = W))
system.time(gr2 <- nsc(start, lower = lm, upper = um, M = M, w = W))

all.equal(gr1, gr2)


### two-step copula via log-density
Z <- t(d <- qnorm(do.call("cbind", lapply(setosa[j], rank, ties.method = "max")) / 
       (N["setosa"] + 1)))
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

### two-step copula via log-probabilities
upr <- do.call("cbind", lapply(setosa[j], rank, ties.method = "max"))
lwr <- upr - 1L
lwr <- t(qnorm(lwr / N["setosa"]))
upr <- t(qnorm(upr / N["setosa"]))

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



### simultaneous copula
idx <- lapply(setosa[j], factor)

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
idx <- lapply(iris[j], factor)

h <- function(x)
    qnorm(cumsum(prop.table(table(x))[-nlevels(x)]))

start0 <- do.call("c", lapply(idx, h))
start <- c(start0, op2$par)

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

