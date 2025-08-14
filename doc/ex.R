
### Diagnostic tests
pkgs <- c("openxlsx", "mvtnorm", "lattice", "MASS", "numDeriv")

ip <- rownames(installed.packages())
if (any(!pkgs %in% ip))
    install.packages(pkgs[!pkgs %in% ip], repos = "https://stat.ethz.ch/CRAN/")

OK <- sapply(pkgs, require, character.only = TRUE)
if (!all(OK)) 
    stop("package(s) ", paste(pkgs[!OK], collapse = ", "), " not available")

### Load data
if (file.exists("HCC.rda")) {
    load("HCC.rda")
} else {
    dat <- read.xlsx("https://datadryad.org/api/v2/files/44697/download", sheet = 1)
    HCC <- with(dat, data.frame(id = 1:nrow(dat),
                                x = factor(HCC_studyGr),
                                AFP = log(AFP_ng_per_ml),
                                PIV = log(PIVKA_delete_range),
                                OPN = log(OPN),
                                DKK = log(DKK)))
    save(HCC, file = "HCC.rda")
}

vars <- c("AFP", "PIV", "OPN", "DKK")

splom(~ HCC[, vars], groups = x, data = HCC)

### LDA
lda(x ~ AFP + PIV + OPN + DKK, data = HCC)


obs <- t(HCC[, vars])
J <- dim(obs)[1L]
pidx <- rep(gl(4, 1, labels = c("Lm", "Lm1", "Ld", "Lo")), 
            times = c(J, J, J, J * (J - 1) / 2))
plim <- rep(-Inf, length.out = length(pidx))
plim[pidx == "Ld"] <- 1e-4
start <- runif(length(pidx))

ll <- function(parm) {
    plist <- split(parm, pidx)
    im <- cbind(plist$Lm, plist$Lm + plist$Lm1)[,HCC$x]
    L <- ltMatrices(plist$Lo, diag = FALSE, names = vars)
    diagonals(L) <- plist$Ld
    object <- mvnorm(invcholmean = im, invchol = L)
    -logLik(object, obs = obs)
}

sc <- function(parm) {
    plist <- split(parm, pidx)
    im <- cbind(plist$Lm, plist$Lm + plist$Lm1)[,HCC$x]
    L <- ltMatrices(plist$Lo, diag = FALSE, names = vars)
    diagonals(L) <- plist$Ld
    object <- mvnorm(invcholmean = im, invchol = L)
    ret <- lLgrad(object, obs = obs)
    -c(rowSums(ret$invcholmean),
       rowSums(ret$invcholmean[,HCC$x == "1"]),
       rowSums(diagonals(ret$scale)),
       rowSums(Lower_tri(ret$scale, diag = FALSE)))
}

ll(start)
sc(start)
grad(ll, start)

(opLDA <- optim(par = start, fn = ll, gr = sc, method = "L-BFGS-B", lower =
plim))

### QDA
pidx <- rep(gl(6, 1, labels = c("Lm", "Lm1", "Ld0", "Ld1", "Lo0", "Lo1")), 
            times = c(rep(J, 4), rep(J * (J - 1) / 2, 2)))
plim <- rep(-Inf, length.out = length(pidx))
plim[pidx %in% c("Ld0", "Ld1")] <- 1e-4
start <- runif(length(pidx))

ll <- function(parm) {
    plist <- split(parm, pidx)
    im <- cbind(plist$Lm, plist$Lm + plist$Lm1)[,HCC$x]
    L <- ltMatrices(cbind(plist$Lo0, plist$Lo1), diag = FALSE, names = vars)
    diagonals(L) <- cbind(plist$Ld0, plist$Ld1)
    L <- L[HCC$x,]
    object <- mvnorm(invcholmean = im, invchol = L)
    -logLik(object, obs = obs)
}

sc <- function(parm) {
    plist <- split(parm, pidx)
    im <- cbind(plist$Lm, plist$Lm + plist$Lm1)[,HCC$x]
    L <- ltMatrices(cbind(plist$Lo0, plist$Lo1), diag = FALSE, names = vars)
    diagonals(L) <- cbind(plist$Ld0, plist$Ld1)
    L <- L[HCC$x,]
    object <- mvnorm(invcholmean = im, invchol = L)
    ret <- lLgrad(object, obs = obs)
    -c(rowSums(ret$invcholmean),
       rowSums(ret$invcholmean[,HCC$x == "1"]),
       rowSums(diagonals(ret$scale)[, HCC$x == "0"]),
       rowSums(diagonals(ret$scale)[, HCC$x == "1"]),
       rowSums(Lower_tri(ret$scale, diag = FALSE)[, HCC$x == "0"]),
       rowSums(Lower_tri(ret$scale, diag = FALSE)[, HCC$x == "1"]))
}

ll(start)
sc(start)
grad(ll, start)

(opQDA <- optim(par = start, fn = ll, gr = sc, method = "L-BFGS-B", lower = plim))

plist <- split(opQDA$par, pidx)
im0 <- plist$Lm
im1 <- plist$Lm + plist$Lm1 
L0 <- ltMatrices(plist$Lo0, diag = FALSE, names = vars)
diagonals(L0) <- plist$Ld0
L1 <- ltMatrices(plist$Lo1, diag = FALSE, names = vars)
diagonals(L1) <- plist$Ld1

m0 <- mvnorm(invcholmean = im0, invchol = L0)
m1 <- mvnorm(invcholmean = im1, invchol = L1)
HCC$llR <- logLik(m0, obs = obs, logLik = FALSE) - logLik(m1, obs = obs, logLik = FALSE)
boxplot(llR ~ x, data = HCC)
abline(h = 0, col = "red")

### discrete QDA
iHCC <- HCC
qOPN <- quantile(HCC$OPN, probs = 1:4 / 5)
iHCC$OPN <- cut(iHCC$OPN, breaks = c(-Inf, qOPN, Inf))
qDKK <- quantile(HCC$DKK, probs = 1:4 / 5)
iHCC$DKK <- cut(iHCC$DKK, breaks = c(-Inf, qDKK, Inf))

lwr <- rbind(OPN = c(-Inf, qOPN, Inf)[iHCC$OPN],
             DKK = c(-Inf, qDKK, Inf)[iHCC$DKK])
upr <- rbind(OPN = c(-Inf, qOPN, Inf)[unclass(iHCC$OPN) + 1L],
             DKK = c(-Inf, qDKK, Inf)[unclass(iHCC$DKK) + 1L])
obs <- obs[c("AFP", "PIV"),]


ll <- function(parm) {
    plist <- split(parm, pidx)
    im <- cbind(plist$Lm, plist$Lm + plist$Lm1)[,HCC$x]
    L <- ltMatrices(cbind(plist$Lo0, plist$Lo1), diag = FALSE, names = vars)
    diagonals(L) <- cbind(plist$Ld0, plist$Ld1)
    L <- L[HCC$x,]
    object <- mvnorm(invcholmean = im, invchol = L)
    -logLik(object, obs = obs, lower = lwr, upper = upr, seed = 2908, M = 1000)
}

sc <- function(parm) {
    plist <- split(parm, pidx)
    im <- cbind(plist$Lm, plist$Lm + plist$Lm1)[,HCC$x]
    L <- ltMatrices(cbind(plist$Lo0, plist$Lo1), diag = FALSE, names = vars)
    diagonals(L) <- cbind(plist$Ld0, plist$Ld1)
    L <- L[HCC$x,]
    object <- mvnorm(invcholmean = im, invchol = L)
    ret <- lLgrad(object, obs = obs, lower = lwr, upper = upr, seed = 2908, M = 1000)
    ret$invcholmean[!is.finite(ret$invcholmean)] <- NA
    lt <- Lower_tri(ret$scale, diag = FALSE)
    lt[!is.finite(lt)] <- NA
    -c(rowSums(ret$invcholmean, na.rm = TRUE),
       rowSums(ret$invcholmean[,HCC$x == "1"], na.rm = TRUE),
       rowSums(diagonals(ret$scale)[, HCC$x == "0"], na.rm = TRUE),
       rowSums(diagonals(ret$scale)[, HCC$x == "1"], na.rm = TRUE),
       rowSums(lt[, HCC$x == "0"], na.rm = TRUE),
       rowSums(lt[, HCC$x == "1"], na.rm = TRUE))
}

start <- opQDA$par
ll(start)
sc(start)
grad(ll, start)

(iopQDA <- optim(par = start, fn = ll, gr = sc, method = "L-BFGS-B", lower =
plim))

plist <- split(iopQDA$par, pidx)
im0 <- plist$Lm
im1 <- plist$Lm + plist$Lm1 
L0 <- ltMatrices(plist$Lo0, diag = FALSE, names = vars)
diagonals(L0) <- plist$Ld0
L1 <- ltMatrices(plist$Lo1, diag = FALSE, names = vars)
diagonals(L1) <- plist$Ld1

m0 <- mvnorm(invcholmean = im0, invchol = L0)
m1 <- mvnorm(invcholmean = im1, invchol = L1)

AFPmiss <- sample.int(ncol(obs), floor(ncol(obs) / 4))

HCC$illR <- 0
HCC$illR[-AFPmiss] <- 
    logLik(m0, 
           obs = obs[, -AFPmiss], 
           lower = lwr[, -AFPmiss], 
           upper = upr[, -AFPmiss], 
           logLik = FALSE, seed = 2908, M = 1000) - 
    logLik(m1, 
           obs = obs[, -AFPmiss], 
           lower = lwr[, -AFPmiss], 
           upper = upr[, -AFPmiss], 
           logLik = FALSE, seed = 2908, M = 1000)
HCC$illR[AFPmiss] <- 
    logLik(m0, 
           obs = obs["PIV",AFPmiss,drop = FALSE], 
           lower = lwr[, AFPmiss], 
           upper = upr[, AFPmiss], 
           logLik = FALSE, seed = 2908, M = 1000) - 
    logLik(m1, 
           obs = obs["PIV",AFPmiss,drop = FALSE], 
           lower = lwr[, AFPmiss], 
           upper = upr[, AFPmiss], 
           logLik = FALSE, seed = 2908, M = 1000)

boxplot(illR ~ x, data = HCC)
abline(h = 0, col = "red")

plot(illR ~ llR, data = HCC, col = 1 + is.na(obs["AFP",]), pch = 19)

### Regression
data("bodyfat", package = "TH.data")
bodyfat <- bodyfat[, c((1:ncol(bodyfat))[-2], 2)]
vars <- colnames(bodyfat)
ct <- lapply(bodyfat, function(x) cut(x, breaks = c(-Inf, sort(unique(x)))))

J <- length(ct)

pidx <- rep(gl(J + 1, 1, labels = c(vars, "Lo")), c(sapply(ct, nlevels) - 1, J * (J - 1) / 2))

start <- c(unlist(lapply(ct, function(x) {
    ret <- qnorm(cumsum(table(x)[-nlevels(x)]) / length(x))
    c(ret[1], diff(ret))
})), rep(0, length.out = J * (J - 1) / 2))

llim <- rep(1e-4, length.out = length(start))
llim[pidx == "Lo"] <- -Inf
llim[c(1, cumsum(sapply(ct, nlevels) - 1) + 1)] <- -Inf
                
cbind(pidx, start, llim)

ll <- function(parm) {
    plist <- split(parm, pidx)
    plist[vars] <- lapply(plist[vars], cumsum)
    lwr <- do.call("rbind", lapply(vars, function(x) c(-Inf, plist[[x]])[ct[[x]]]))
    upr <- do.call("rbind", lapply(vars, function(x) c(plist[[x]], Inf)[ct[[x]]]))
    rownames(lwr) <- rownames(upr) <- vars
    L <- ltMatrices(plist$Lo, diag = FALSE, names = vars)
    object <- mvnorm(invchol = L)
    -logLik(object, lower = lwr, upper = upr, standardize = TRUE, seed = 2908, M = 250)
}

ll(start)
