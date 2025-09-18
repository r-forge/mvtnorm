
pkgs <- c("openxlsx", "mvtnorm", "lattice", "MASS", "numDeriv", "lavaan", "tram", 
          "lava", "survival", "randomForestSRC")

ip <- rownames(installed.packages())
if (any(!pkgs %in% ip))
    install.packages(pkgs[!pkgs %in% ip], repos = "https://stat.ethz.ch/CRAN/")

OK <- sapply(pkgs, require, character.only = TRUE)
if (!all(OK)) 
    stop("package(s) ", paste(pkgs[!OK], collapse = ", "), " not available")

CHKsc <- FALSE

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

#splom(~ HCC[, vars], groups = x, data = HCC)

### LDA
#lda(x ~ AFP + PIV + OPN + DKK, data = HCC)

N <- nrow(HCC)
m <- matrix(sapply(y <- split(HCC[, vars,drop = FALSE], HCC$x), colMeans), ncol = 2)
S <- var(HCC[, vars,drop = FALSE] - t(m)[HCC$x,]) * (N - 1) / N
Sg <- lapply(y, function(x) var(x) * (nrow(x) - 1) / nrow(x))

sum(dmvnorm(y[[1]], mean = m[,1,drop = FALSE], sigma = S, log = TRUE)) + 
sum(dmvnorm(y[[2]], mean = m[,2,drop = FALSE], sigma = S, log = TRUE))

HCC$llR <- dmvnorm(HCC[,vars,drop = FALSE], mean = m[,1, drop = FALSE], sigma = S, log = TRUE) - 
           dmvnorm(HCC[,vars,drop = FALSE], mean = m[,2, drop = FALSE], sigma = S, log = TRUE) 

obs <- t(HCC[, vars, drop = FALSE])
J <- dim(obs)[1L]
pidx <- rep(gl(4, 1, labels = c("Lm", "Lm1", "Ld", "Lo")), 
            times = c(J, J, J, J * (J - 1) / 2))
plim <- rep_len(-Inf, length.out = length(pidx))
plim[pidx == "Ld"] <- 1e-4
start <- rep_len(0, length.out = length(pidx))
start[pidx == "Ld"] <- 1

nllLDA <- function(parm, group = HCC$x, object = FALSE) {
    plist <- split(parm, pidx)
    im <- cbind(plist$Lm, plist$Lm + plist$Lm1)[, group, drop = FALSE]
    L <- ltMatrices(plist$Lo, diag = FALSE, names = vars)
    diagonals(L) <- plist$Ld
    obj <- mvnorm(invcholmean = im, invchol = L)
    if (object) return(obj)
    - logLik(obj, obs = obs)
}

nscLDA <- function(parm) {
    obj <- nllLDA(parm, object = TRUE)
    ret <- lLgrad(obj, obs = obs)
    -c(rowSums(ret$invcholmean),
       rowSums(ret$invcholmean[,HCC$x == "1",drop = FALSE]),
       rowSums(diagonals(ret$scale)),
       rowSums(Lower_tri(ret$scale, diag = FALSE)))
}

if (CHKsc)
  print(all.equal(unname(nscLDA(start)), grad(nllLDA, start)))

ctrl <- list(trace = TRUE, maxit = 1000)
(opLDA <- optim(par = start, fn = nllLDA, gr = nscLDA, 
                method = "L-BFGS-B", lower = plim, control = ctrl))

grp <- sort(unique(HCC$x))
grp0 <- rep(grp[1], length.out = nrow(HCC))
grp1 <- rep(grp[2], length.out = nrow(HCC))
HCC$llR_LDA <- logLik(nllLDA(opLDA$par, group = grp0, object = TRUE), 
                      obs = obs, logLik = FALSE) -
               logLik(nllLDA(opLDA$par, group = grp1, object = TRUE), 
                      obs = obs, logLik = FALSE)

boxplot(llR_LDA ~ x, data = HCC)
abline(h = 0, col = "red")

nllLDA(opLDA$par, object = TRUE)$mean[,c(1, 300)]
m
invchol2cov(nllLDA(opLDA$par, object = TRUE)$scale)
S

plot(llR ~ llR_LDA, data = HCC)
abline(a = 0, b = 1)

AO <- c("AFP", "OPN")
HCC$llR_LDA_AO <- logLik(nllLDA(opLDA$par, group = grp0, object = TRUE), 
                         obs = obs[AO,], logLik = FALSE) -
                  logLik(nllLDA(opLDA$par, group = grp1, object = TRUE), 
                         obs = obs[AO,], logLik = FALSE)

plot(llR_LDA ~ llR_LDA_AO, data = HCC, col = (1:2)[HCC$x])
abline(a = 0, b = 1)
abline(h = 0, v = 0)

### QDA
pidx <- rep(gl(6, 1, labels = c("Lm", "Lm1", "Ld0", "Ld1", "Lo0", "Lo1")), 
            times = c(rep(J, 4), rep(J * (J - 1) / 2, 2)))
plim <- rep(-Inf, length.out = length(pidx))
plim[pidx %in% c("Ld0", "Ld1")] <- 1e-4
start <- rep_len(0, length.out = length(pidx))
start[pidx %in% c("Ld0", "Ld1")] <- 1

nllQDA <- function(parm, group = HCC$x, object = FALSE) {
    plist <- split(parm, pidx)
    im <- cbind(plist$Lm, plist$Lm + plist$Lm1)[,group]
    L <- ltMatrices(cbind(plist$Lo0, plist$Lo1), diag = FALSE, names = vars)
    diagonals(L) <- cbind(plist$Ld0, plist$Ld1)
    L <- L[group,]
    obj <- mvnorm(invcholmean = im, invchol = L)
    if (object) return(obj)
    - logLik(obj, obs = obs)
}

nscQDA <- function(parm) {
    obj <- nllQDA(parm, object = TRUE)
    ret <- lLgrad(obj, obs = obs)
    - c(rowSums(ret$invcholmean),
        rowSums(ret$invcholmean[,HCC$x == "1"]),
        rowSums(diagonals(ret$scale)[, HCC$x == "0"]),
        rowSums(diagonals(ret$scale)[, HCC$x == "1"]),
        rowSums(Lower_tri(ret$scale, diag = FALSE)[, HCC$x == "0"]),
        rowSums(Lower_tri(ret$scale, diag = FALSE)[, HCC$x == "1"]))
}

if (CHKsc)
  print(all.equal(nsc(start), grad(nll, start)))

(opQDA <- optim(par = start, fn = nllQDA, gr = nscQDA, method = "L-BFGS-B", lower = plim, control = ctrl))

HCC$llR_QDA <- logLik(nllQDA(opQDA$par, group = grp0, object = TRUE), 
                      obs = obs, logLik = FALSE) - 
               logLik(nllQDA(opQDA$par, group = grp1, object = TRUE), 
                      obs = obs, logLik = FALSE)

boxplot(llR_QDA ~ x, data = HCC)
abline(h = 0, col = "red")

nllQDA(opQDA$par, object = TRUE)$mean[,c(1, 300)]
m
invchol2cov(nllQDA(opQDA$par, object = TRUE)$scale[c(1, 300),])
Sg

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

nllQDAi <- function(parm, group = HCC$x, object = FALSE) {
    obj <- nllQDA(parm, group = group, object = TRUE)
    obj <- list(object = obj, 
                obs = obs[c("AFP", "PIV"),], lower = lwr, upper = upr, seed = 2908, M = 1000)
    if (object) return(obj)
    - do.call("logLik", obj)
}

nscQDAi <- function(parm) {
    args <- nllQDAi(parm, object = TRUE)
    ret <- do.call("lLgrad", args)
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
if (CHKsc)
  print(all.equal(nscQDAi(start), gradQDAi(nll, start)))

(opQDAi <- optim(par = start, fn = nllQDAi, gr = nscQDAi, method = "L-BFGS-B", lower = plim, control = ctrl))

cbind(opQDA$par, opQDAi$par)
nllQDA(opQDA$par)
nllQDA(opQDAi$par)

HCC$llR_QDAi_AO <- logLik(nllQDAi(opQDAi$par, group = grp0, object = TRUE)$object, 
                          obs = obs["AFP",,drop = FALSE], 
                          lower = lwr["OPN",,drop = FALSE],
                          upper = upr["OPN",,drop = FALSE], 
                          logLik = FALSE) -
                   logLik(nllQDAi(opQDAi$par, group = grp1, object = TRUE)$object, 
                          obs = obs["AFP",,drop = FALSE], 
                          lower = lwr["OPN",,drop = FALSE],
                          upper = upr["OPN",,drop = FALSE], 
                          logLik = FALSE)

plot(llR_QDAi_AO ~ llR_LDA_AO, data = HCC, col = (1:2)[HCC$x])
abline(a = 0, b = 1)
abline(h = 0, v = 0)

splom(~ HCC[, grep("llR", colnames(HCC))], groups = x, data = HCC)

### Regression
data("bodyfat", package = "TH.data")
bodyfat <- bodyfat[, c("age", "waistcirc", "hipcirc", "DEXfat")]
bf <- bodyfat[, colnames(bodyfat) != "age"]
J <- length(vars <- colnames(bf))

obs <- t(qnorm(do.call("cbind", lapply(bf, rank, ties.method = "max")) / (nrow(bf) + 1)))
(CR <- cor(t(obs)))

ct <- as.data.frame(lapply(bf, function(x) cut(x, breaks = c(-Inf, sort(unique(x))))))

nll <- function(parm, logLik = TRUE) {
    L <- ltMatrices(parm, names = vars)
    object <- list(object = mvnorm(invchol = L),
                   obs = obs, standardize = TRUE)
    if (!logLik) return(object)
    - do.call("logLik", object)
}

nsc <- function(parm) {
    ret <- do.call("lLgrad", nll(parm, logLik = FALSE))
    -rowSums(Lower_tri(ret$scale, diag = FALSE))
}

start <- rep(0, J * (J - 1) / 2)
nll(start)

op <- optim(par = start, fn = nll, gr = nsc, method = "BFGS", hessian = TRUE)

invchol2cov(Ls <- standardize(invchol = ltMatrices(op$par, names = vars)))
- as.array(Ls)["DEXfat",vars[vars != "DEXfat"],] / as.array(Ls)["DEXfat","DEXfat",]

coef(lm(DEXfat ~ 0 + ., data = as.data.frame(t(obs))))

### Standard errors
Lower_tri(ltMatrices(sqrt(diag(solve(op$hessian))), names = vars))

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

nll <- function(parm, logLik = TRUE) {
    plist <- split(parm, pidx)
    plist[vars] <- lapply(plist[vars], cumsum)
    lwr <- do.call("rbind", lapply(vars, function(x) c(-Inf, plist[[x]])[ct[[x]]]))
    upr <- do.call("rbind", lapply(vars, function(x) c(plist[[x]], Inf)[ct[[x]]]))
    rownames(lwr) <- rownames(upr) <- vars
    L <- ltMatrices(plist$Lo, diag = FALSE, names = vars)
    object <- list(object = mvnorm(invchol = L),
                   lower = lwr, upper = upr, standardize = TRUE, seed = 2908, M = 250)
    if (!logLik) return(object)
    - do.call("logLik", object)
}

X <- lapply(ct, function(x) {
    X <- diag(nlevels(x))
    X[lower.tri(X)] <- 1
    X <- X[x,]
})

nsc <- function(parm) {
    ret <- do.call("lLgrad", nll(parm, logLik = FALSE))
    ret$upper[!is.finite(ret$upper)] <- NA
    ret$lower[!is.finite(ret$lower)] <- NA
    -c(do.call("c", lapply(vars, 
                           function(v) 
                               colSums(X[[v]][,-ncol(X[[v]])] * c(ret$upper[v,]) + 
                                       X[[v]][,-2] * c(ret$lower[v,]), na.rm = TRUE))),
       rowSums(Lower_tri(ret$scale, diag = FALSE), na.rm = TRUE))
}

nll(start)
nsc(start)
if (CHKsc)
  print(all.equal(nsc(start), grad(nll, start)))

opM <- optim(par = start, fn = nll, gr = nsc, 
             lower = llim, method = "L-BFGS-B", hessian = TRUE, control = list(trace = TRUE, maxit = 1000))

invchol2cov(Ls <- standardize(invchol = ltMatrices(split(opM$par, pidx)[["Lo"]], names = vars)))
- as.array(Ls)["DEXfat",vars[vars != "DEXfat"],] / as.array(Ls)["DEXfat","DEXfat",]

coef(lm(DEXfat ~ 0 + ., data = as.data.frame(t(obs))))

### Standard errors
as.array(ltMatrices(sqrt(diag(solve(op$hessian))), names = vars))["DEXfat",,1]
as.array(ltMatrices(split(sqrt(diag(solve(opM$hessian))), pidx)[["Lo"]], names =
vars))["DEXfat",,1]

### same with lava
fm <- as.formula(paste(paste(colnames(ct), collapse = "+"), "~1"))
m <- lvm(fm)
m <- covariance(m, var1 = colnames(ct), pairwise = TRUE)
plot(m)
#system.time(mf <- estimate(m, data = ct))
### not converged
#mf$opt

# logLik(mf)
-opM$value
invchol2cor(nll(opM$par, logLik = FALSE)$object$scale)

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
    object <- list(object = mvnorm(invchol = L),
                   lower = lwr, upper = upr, standardize = TRUE, seed = 2908, M = 250)
    if (!logLik) return(object)
    - do.call("logLik", object)
}

nsc <- function(parm) {
    ret <- do.call("lLgrad", nll(parm, logLik = FALSE))
    ret$lower[!is.finite(ret$lower)] <- NA
    ret$upper[!is.finite(ret$upper)] <- NA
    -c(do.call("c", lapply(vars, 
                           function(v) 
                               colSums(X[[v]][,-ncol(X[[v]])] * c(ret$upper[v,]) + 
                                       X[[v]][,-2] * c(ret$lower[v,]), na.rm = TRUE))),
       rowSums(Lower_tri(ret$scale, diag = FALSE), na.rm = TRUE), 
       rowSums(AGE * Lower_tri(ret$scale, diag = FALSE), na.rm = TRUE))
}

start <- c(start, rep(0, J * (J - 1) / 2))

nll(start)
nsc(start)
if (CHKsc)
  print(all.equal(nsc(start), grad(nll, start)))

opA <- optim(par = start, fn = nll, gr = nsc, 
             lower = llim, method = "L-BFGS-B", hessian = TRUE, control = list(trace = TRUE))

pchisq(2 * (opA$value - opM$value), df = length(opA$par) - length(opM$par), lower.tail = FALSE)

split(opA$par, pidx)[["Loage"]]
# split(sqrt(diag(solve(opA$hessian))), pidx)[["Loage"]]

### SEM: CFA
data("HolzingerSwineford1939", package = "lavaan")
## The famous Holzinger and Swineford (1939) example
HS.model <- ' visual  =~ x1 + x2 + x3
              textual =~ x4 + x5 + x6
              speed   =~ x7 + x8 + x9 '
     
fit <- cfa(HS.model, data = HolzingerSwineford1939, meanstructure = TRUE)
summary(fit, fit.measures = TRUE)

lat <- c("visual", "textual", "speed")
man <- paste0("x", 1:9)

## with lava
mlvm <- lvm(list(x1 + x2 + x3 ~ visual,
              x4 + x5 + x6 ~ textual,
              x7 + x8 + x9 ~ speed))
latent(mlvm) <- ~ visual + textual + speed
covariance(mlvm) <- visual ~ textual
covariance(mlvm) <- textual ~ speed
covariance(mlvm) <- visual ~ speed
intercept(mlvm, ~ visual + textual + speed) <- 0
plot(mlvm)

mf <- estimate(mlvm, data = HolzingerSwineford1939)

logLik(fit)
logLik(mf)

nm <- c(lat, man)
J <- length(nm)
tmp <- unclass(ltMatrices(rep.int(0, J * (J + 1) / 2), diag = TRUE, names = nm))

fix <- paste(man[1 + 0:2 * 3], lat, sep = ".")
meas <- paste(man[-(1 + 0:2 * 3)], rep(lat, each = 2), sep = ".")
cova <- c("textual.visual", "speed.visual", "speed.textual")
tmp[fix,] <- 1

m <- matrix(0, ncol = 1, nrow = J)
rownames(m) <- c(lat, man)

obs <- t(as.matrix(HolzingerSwineford1939[, man]))

lower <- upper <- NULL

nll <- function(parm, logLik = TRUE, obs, lower = NULL, upper = NULL) {
    Ld <- parm[seq_len(J)]
    im <- parm[J + seq_len(J - 3)]
    Lo <- parm[-seq_len(2 * J - 3)]
    tmp[c(cova, meas),] <- Lo
    L <- ltMatrices(tmp, diag = TRUE, names = nm)
    diagonals(L) <- Ld
    m[man,] <- im
    object <- list(object = mvnorm(mean = m, invchol = L), 
                   obs = obs, lower = lower, upper = upper, 
                   seed = 2908, M = 1000)
    if (!logLik) return(object)
    - do.call("logLik", object)
}

nsc <- function(parm, ...) {
    ret <- do.call("lLgrad", nll(parm, logLik = FALSE, ...))
    - c(rowSums(diagonals(ret$scale)),
        rowSums(ret$mean)[man],
        rowSums(Lower_tri(ret$scale))[c(cova, meas)])
}

start <- runif(J + J - 3 + length(cova) + length(meas))
nll(start, logLik = FALSE, obs = obs)$object

nll(start, obs = obs)
nsc(start, obs = obs)
if (CHKsc)
  print(all.equal(nsc(start, obs = obs), grad(nll, start, obs = obs)))

lwr <- start
lwr[] <- -Inf
lwr[seq_len(J)] <- 0

op <- optim(start, fn = function(parm) nll(parm, obs = obs), 
            gr = function(parm) nsc(parm, obs = obs), lower = lwr, method = "L-BFGS-B",
      control = list(trace = TRUE, maxit = 1000))

logLik(mf)
-op$value
   
r <- nll(op$par, logLik = FALSE, obs = obs)$object
 
margDist(r, which = lat)

cdstr <- condDist(r, which_given = lat, given = diag(length(lat)))

r$mean[man,]
1 / diagonals(cdstr$scale)^2
- (cdstr$mean - r$mean[man,])

### with censoring
HS <- HolzingerSwineford1939
HS$x9 <- with(HS, Surv(pmin(x9, 6), x9 < 6))

mfi <- estimate(mlvm, data = HS) # control = list(start = coef(mf))

lower <- upper <- obs
lower["x9",] <- pmin(lower["x9", ], 6)
upper["x9", upper["x9",] > 6] <- Inf
lower <- lower["x9",,drop = FALSE]
upper <- upper["x9",,drop = FALSE]

start <- op$par

sll <- function(parm) {
    nll(parm, obs = obs[,is.finite(upper)]) + 
    nll(parm, obs = obs[1:8,!is.finite(upper)],
              lower = lower[, !is.finite(upper),drop = FALSE],
              upper = upper[, !is.finite(upper), drop = FALSE])
}

ssc <- function(parm) {
    nsc(parm, obs = obs[,is.finite(upper)]) + 
    nsc(parm, obs = obs[1:8,!is.finite(upper)],
              lower = lower[, !is.finite(upper),drop = FALSE],
              upper = upper[, !is.finite(upper), drop = FALSE])
}

sll(start)
ssc(start)
if (CHKsc)
  print(all.equal(ssc(start), grad(sll, start)))

opi <- optim(start, fn = sll, gr = ssc, lower = lwr, method = "L-BFGS-B",
              control = list(trace = TRUE, maxit = 1000))

nll(opi$par, logLik = FALSE, obs = obs)$object

logLik(mfi)
-opi$value

### with ordinal
HS <- HolzingerSwineford1939
HS$x1 <- as.ordered(HS$x1)
HS$x2 <- as.ordered(HS$x2)

mfo <- estimate(mlvm, data = HS) # control = list(start = coef(mf))

### competing risks
data("follic", package = "randomForestSRC")
  
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
ecens <- cmpr$time[,2] == 0
erelap <- cmpr$time[,2] == 1
edeath <- cmpr$time[,2] == 2

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
                        standardize = TRUE) + sum(log(prelap[2] / cmpr$time[erelap,1]))
    ret <- ret + logLik(object, 
                        obs = rbind(death = hdeath[edeath]),
                        lower = rbind(relap = hrelap[edeath]),
                        upper = rbind(relap = rep(Inf, etab[3])),
                        standardize = TRUE) + sum(log(pdeath[2] / cmpr$time[edeath,1]))
    ret <- ret + logLik(object, 
                        lower = rbind(relap = hrelap[ecens], 
                                      death = hdeath[ecens]), 
                        upper = rbind(relap = rep(Inf, etab[1]),
                                      death = rep(Inf, etab[1])),
                        standardize = TRUE, seed = 2908, M = 1000)
    - ret
}

ll(start)

llim <- c(-Inf, 1e-4, -Inf, 1e-4, -Inf)
opCMP <- optim(start, fn = ll, lower = llim, method = "L-BFGS-B", hessian = TRUE, control = list(trace = TRUE))

opCMP$par
sqrt(diag(solve(opCMP$hessian)))

m <- Compris(time ~ 1, data = cmpr, primary = "BoxCox", 
             log_first = TRUE, order = 1, 
             args = list(seed = 2908, type = "MC", M = 1000))

### note: negative lambda
mC <- Compris(time ~ 1, data = cmpr, primary = "Coxph", 
             log_first = TRUE, order = 1, 
args = list(seed = 2908, type = "MC", M = 1000))

### lambda extremely variable
sqrt(diag(vcov(m)))
sqrt(diag(vcov(mC)))

### lava
set.seed(290875)
N <- 100
J <- 3
Yn <- paste0("Y", seq_len(J))
nm <- c("Z", Yn)

m <- lvm(list(Y1 ~ Z, Y2 ~ Z, Y3 ~ Z))
latent(m) <- ~ Z
intercept(m, ~ Z) <- 0
covariance(m, ~ Z) <- 1
parm <- 2 + runif(9)
d <- sim(m, n = N, p = parm)
mh <- estimate(m, data = d)
logLik(mh)

summary(mh)

obs <- t(as.matrix(d[, Yn]))

nll <- function(parm, logLik = TRUE) {
    dg <- c(1, parm[seq_len(J)])
    im <- c(0, parm[J + seq_len(J)])
    bZ <- c(parm[2 * J + seq_len(J)], rep(0, J * (J - 1) / 2))
    L <- ltMatrices(bZ, byrow = FALSE, diag = FALSE, names = nm)
    diagonals(L) <- dg
    obj <- mvnorm(invcholmean = im, invchol = L)
    if (!logLik) return(obj)
    - logLik(obj, obs = obs)
}

nsc <- function(parm) {
    obj <- nll(parm, logLik = FALSE)
    ret <- lLgrad(obj, obs = obs)
    - c(rowSums(diagonals(ret$scale))[-1L],
        rowSums(ret$invcholmean)[-1L],
        rowSums(Lower_tri(ret$scale))[1:3])
}

nll(parm)
all.equal(nsc(parm), grad(nll, parm), check.attributes = FALSE)

op <- optim(parm, fn = nll, gr = nsc, 
            lower = c(rep(0, 3), rep(-Inf, 6)),
            method = "L-BFGS-B", hessian = TRUE)
-op$value
logLik(mh)
r <- nll(op$par, logLik = FALSE)
cdstr <- condDist(r, which_given = "Z", given = matrix(1))

cf <- c(r$mean[-1L], -(cdstr$mean - r$mean[Yn,]), 1 / diagonals(cdstr$scale)^2)
cbind(cf, coef(mh))

### saturated model
logLik(margDist(r, which = Yn), obs = obs)

