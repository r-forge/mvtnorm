attach(NULL, name = "CheckExEnv")
assign(".CheckExEnv", pos.to.env(2), pos = length(search()))
assign("ptime", proc.time(), env = .CheckExEnv)
postscript("mvtnorm-Examples.ps")
assign("par.postscript", par(no.readonly = TRUE), env = .CheckExEnv)
options(contrasts = c(unordered = "contr.treatment", ordered = "contr.poly"))
library('mvtnorm')
rm(list = ls(all = TRUE)); .Random.seed <- c(0,rep(7654,3))
###--- >>> `pmvnorm' <<<----- Multivariate Normal Distribution

	## alias	 help(pmvnorm)

##___ Examples ___:


n <- 5
mean <- rep(0, 5)
lower <- rep(-1, 5)
upper <- rep(3, 5)
corr <- diag(5)
corr[lower.tri(corr)] <- 0.5
prob <- pmvnorm(mean, corr, lower, upper)
print(prob)


## Keywords: 'multivariate normal distribution'.


rm(list = ls(all = TRUE)); .Random.seed <- c(0,rep(7654,3))
###--- >>> `pmvt' <<<----- Multivariate t Distribution

	## alias	 help(pmvt)

##___ Examples ___:


n <- 5
lower <- rep(-1, 5)
upper <- rep(3, 5)
df <- 4
corr <- diag(5)
corr[lower.tri(corr)] <- 0.5
delta <- rep(0, 5)
prob <- pmvt(lower, upper, df, corr , delta)
print(prob)


## Keywords: 'distribution'.


cat("Time elapsed: ", proc.time() - get("ptime", env = .CheckExEnv),"\n")
dev.off(); quit('no')
