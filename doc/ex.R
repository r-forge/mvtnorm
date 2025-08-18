
pkgs <- c("openxlsx", "mvtnorm", "lattice", "MASS", "numDeriv", "lavaan")

ip <- rownames(installed.packages())
if (any(!pkgs %in% ip))
    install.packages(pkgs[!pkgs %in% ip], repos = "https://stat.ethz.ch/CRAN/")

OK <- sapply(pkgs, require, character.only = TRUE)
if (!all(OK)) 
    stop("package(s) ", paste(pkgs[!OK], collapse = ", "), " not available")

### Diagnostic tests
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

### mixed discrete-continuous QDA
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
bodyfat <- bodyfat[,c("age", "hipcirc", "DEXfat")]
bf <- bodyfat[, colnames(bodyfat) != "age"]
J <- length(vars <- colnames(bf))

obs <- t(qnorm(do.call("cbind", lapply(bf, rank, ties.method = "max")) / (nrow(bf) + 1)))
(CR <- cor(t(obs)))

ct <- lapply(bf, function(x) cut(x, breaks = c(-Inf, sort(unique(x)))))

ll <- function(parm) {
    L <- ltMatrices(parm, names = vars)
    object <- mvnorm(invchol = L)
    -logLik(object, obs = obs, standardize = TRUE)
}

sc <- function(parm) {
    L <- ltMatrices(parm, names = vars)
    object <- mvnorm(invchol = L)
    -rowSums(Lower_tri(lLgrad(object, obs = obs, standardize = TRUE)$scale, diag = FALSE))
}

start <- rep(0, J * (J - 1) / 2)
ll(start)

op <- optim(par = start, fn = ll, gr = sc, method = "BFGS", hessian = TRUE)

invchol2cov(Ls <- standardize(invchol = ltMatrices(op$par, names = vars)))
as.array(Ls)["DEXfat",,]


### with margins

pidx <- rep(gl(J + 1, 1, labels = c(vars, "Lo")), c(sapply(ct, nlevels) - 1, J * (J - 1) / 2))

start <- c(unlist(lapply(ct, function(x) {
    ret <- qnorm(cumsum(table(x)[-nlevels(x)]) / length(x))
    c(ret[1], diff(ret))
})), op$par)

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

X <- lapply(ct, function(x) {
    X <- diag(nlevels(x))
    X[lower.tri(X)] <- 1
    X <- X[x,]
})

sc <- function(parm) {
    plist <- split(parm, pidx)
    plist[vars] <- lapply(plist[vars], cumsum)
    lwr <- do.call("rbind", lapply(vars, function(x) c(-Inf, plist[[x]])[ct[[x]]]))
    upr <- do.call("rbind", lapply(vars, function(x) c(plist[[x]], Inf)[ct[[x]]]))
    rownames(lwr) <- rownames(upr) <- vars
    L <- ltMatrices(plist$Lo, diag = FALSE, names = vars)
    object <- mvnorm(invchol = L)
    ret <- lLgrad(object, lower = lwr, upper = upr, standardize = TRUE, seed = 2908, M = 250)
    -c(do.call("c", lapply(vars, 
                           function(v) 
                               colSums(X[[v]][,-ncol(X[[v]])] * c(ret$upper[v,]) + 
                                       X[[v]][,-2] * c(ret$lower[v,]), na.rm = TRUE))),
       rowSums(Lower_tri(ret$scale, diag = FALSE)))
}

ll(start)
cbind(sc(start), grad(ll, start))

opM <- optim(par = start, fn = ll, gr = sc, 
             lower = llim, method = "L-BFGS-B", hessian = FALSE, control = list(trace = TRUE))

### with age-dependent correlation
pidx <- rep(gl(J + 2, 1, labels = c(vars, "Lo", "Loage")), c(sapply(ct, nlevels) - 1, rep(J * (J - 1) / 2, 2)))
llim <- c(llim, rep(-Inf, J * (J - 1) / 2))

AGE <- matrix(bodyfat$age, ncol = nrow(bf), nrow = J * (J - 1) / 2, byrow = TRUE)

ll <- function(parm) {
    plist <- split(parm, pidx)
    plist[vars] <- lapply(plist[vars], cumsum)
    lwr <- do.call("rbind", lapply(vars, function(x) c(-Inf, plist[[x]])[ct[[x]]]))
    upr <- do.call("rbind", lapply(vars, function(x) c(plist[[x]], Inf)[ct[[x]]]))
    rownames(lwr) <- rownames(upr) <- vars
    L <- ltMatrices(plist$Lo + AGE * plist$Loage , diag = FALSE, names = vars)
    object <- mvnorm(invchol = L)
    -logLik(object, lower = lwr, upper = upr, standardize = TRUE, seed = 2908, M = 250)
}

sc <- function(parm) {
    plist <- split(parm, pidx)
    plist[vars] <- lapply(plist[vars], cumsum)
    lwr <- do.call("rbind", lapply(vars, function(x) c(-Inf, plist[[x]])[ct[[x]]]))
    upr <- do.call("rbind", lapply(vars, function(x) c(plist[[x]], Inf)[ct[[x]]]))
    rownames(lwr) <- rownames(upr) <- vars
    L <- ltMatrices(plist$Lo + AGE * plist$Loage , diag = FALSE, names = vars)
    object <- mvnorm(invchol = L)
    ret <- lLgrad(object, lower = lwr, upper = upr, standardize = TRUE, seed = 2908, M = 250)
    ret$lower[!is.finite(ret$lower)] <- NA
    ret$upper[!is.finite(ret$upper)] <- NA
    -c(do.call("c", lapply(vars, 
                           function(v) 
                               colSums(X[[v]][,-ncol(X[[v]])] * c(ret$upper[v,]) + 
                                       X[[v]][,-2] * c(ret$lower[v,]), na.rm = TRUE))),
       rowSums(Lower_tri(ret$scale, diag = FALSE), na.rm = TRUE), 
       rowSums(AGE * Lower_tri(ret$scale, diag = FALSE), na.rm = TRUE))
}

start <- c(start, 0)

ll(start)
cbind(sc(start), grad(ll, start))

opA <- optim(par = start, fn = ll, gr = sc, 
             lower = llim, method = "L-BFGS-B", hessian = FALSE, control = list(trace = TRUE))

### SEM: CFA
data("HolzingerSwineford1939", package = "lavaan")
## The famous Holzinger and Swineford (1939) example
HS.model <- ' visual  =~ x1 + x2 + x3
              textual =~ x4 + x5 + x6
              speed   =~ x7 + x8 + x9 '
     
fit <- cfa(HS.model, data = HolzingerSwineford1939)
### meanstructure = TRUE for additional means
summary(fit, fit.measures = TRUE)

latJ <- 3
lat <- c("visual", "textual", "speed")
man <- paste0("x", 1:9)
manJ <- length(man)
# obs <- t(as.matrix(HolzingerSwineford1939[, man]))
obs <- t(scale(HolzingerSwineford1939[,man], center = TRUE, scale = FALSE))

z6 <- rep(0, 6)
z3 <- rep(0, 3)
z9 <- rep(0, 9)

ll <- function(parm) {
    Lo <- parm[1:9]
    Ld <- parm[10:21]
    od <- c(Lo[1:2], 1, Lo[4:5], z6, Lo[3], z3, 1, Lo[6:7], z9, 1, Lo[8:9], rep(0, manJ * (manJ - 1) / 2))
    L <- ltMatrices(od, diag = FALSE, names = c(lat, man))
    diagonals(L) <- Ld
#    nu <- parm[-(1:21)]
    object <- mvnorm(invchol = L)#, invcholmean = c(z3, nu))
    -logLik(object, obs = obs)
}

sc <- function(parm) {
    Lo <- parm[1:9]
    Ld <- parm[10:21]
    od <- c(Lo[1:2], NA, Lo[4:5], z6, Lo[3], z3, NA, Lo[6:7], z9, NA, Lo[8:9], rep(0, manJ * (manJ - 1) / 2))
    nn <- which(abs(od) > 0)
    od[is.na(od)] <- 1.0
    L <- ltMatrices(od, diag = FALSE, names = c(lat, man))
    diagonals(L) <- Ld
#    nu <- parm[-(1:21)]
    object <- mvnorm(invchol = L)#, invcholmean = c(z3, nu))
    ret <- lLgrad(object, obs = obs)
    so <- rowSums(Lower_tri(ret$scale, diag = FALSE)[nn,])
    so <- so[c(1, 2, 5, 3:4, 6:9)]
    sd <- rowSums(diagonals(ret$scale))
#    sn <- rowSums(ret$invcholmean[-(1:3),])
    - c(so, sd)#, sn)		
}

llim <- rep(-Inf, 21)# + 9)
llim[10:21] <- 1e-4

start <- rep(.1, 21)# + 9)

ll(start)
sc(start)
grad(ll, start)
    
opCFA <- optim(par = start, fn = ll, gr = sc, 
               lower = llim, method = "L-BFGS-B", hessian = FALSE, control = list(trace = TRUE))

-opCFA$value
length(opCFA$value)
logLik(fit)

Lo <- opCFA$par[1:9]
Ld <- opCFA$par[10:21]
od <- c(Lo[1:2], NA, Lo[4:5], z6, Lo[3], z3, NA, Lo[6:7], z9, NA, Lo[8:9], rep(0, manJ * (manJ - 1) / 2))
od[is.na(od)] <- 1.0
L <- ltMatrices(od, diag = FALSE, names = c(lat, man))
diagonals(L) <- Ld
L

round(as.array(invchol2cov(L))[man,man,1], 3)
fitted(fit)

### competing risks
data("follic", package = "randomForestSRC")
library("survival")
  
## Therapy:
### Radiotherapy alone (RT) or Chemotherapy + Radiotherapy (CMTRT)
follic$ch <- factor(as.character(follic$ch),
  levels = c("N", "Y"), labels = c("RT", "CMTRT")) 
  
## Set-up `Surv' object for "Compris":
### "status" needs to be given in with levels:
### 1: for independent censoring, 
### 2: for the event of interest,
### 3: dependent censoring 
follic$status <- factor(follic$status, levels = 0:2,
  labels = c("admin_cens", "relapse", "death"))
  
cmpr <- data.frame(time = with(follic, Surv(time = time, event = status)), 
                   trt = follic$ch)

start <- c(0, 1, 0, 1, 0)

etab <- table(cmpr$time[,2])
erelap <- cmpr$time[,2] == 1
edeath <- cmpr$time[,2] == 2
ecens <- cmpr$time[,2] == 0

ll <- function(parm) {
    prelap <- parm[1:2]
    pdeath <- parm[3:4]
    lambda <- parm[5]
    hrelap <- prelap[1] + prelap[2] * log(cmpr$time[,1])
    hdeath <- pdeath[1] + pdeath[2] * log(cmpr$time[,1])
    L <- ltMatrices(lambda, names = c("relap", "death"))
    object <- mvnorm(invchol = L)
    ret <- 0
    ret <- ret + logLik(object, 
                        obs = rbind(relap = hrelap[erelap]),
                        lower = rbind(death = hdeath[erelap]),
                        upper = rbind(death = rep(Inf, etab[2])),
                        standardize = TRUE) + etab[2] * log(prelap[2])
    ret <- ret +  logLik(object, 
                         obs = rbind(death = hdeath[edeath]),
                         lower = rbind(relap = hrelap[edeath]),
                         upper = rbind(relap = rep(Inf, etab[3])),
                         standardize = TRUE) + etab[3] * log(pdeath[2])
    ret <- ret +  logLik(object, 
                         lower = rbind(relap = hrelap[ecens], 
                                       death = hdeath[ecens]), 
                         upper = rbind(relap = rep(Inf, etab[1]),
                                       death = rep(Inf, etab[1])),
                         standardize = TRUE, seed = 2908, M = 1000)
    - ret
}

ll(start)

llim <- c(-Inf, 1e-4, -Inf, 1e-4, -Inf)
optim(start, fn = ll, lower = llim, method = "L-BFGS-B", hessian = FALSE, control = list(trace = TRUE))
