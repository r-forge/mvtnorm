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
lower <- -1
upper <- 3
corr <- diag(5)
corr[lower.tri(corr)] <- 0.5
corr[upper.tri(corr)] <- 0.5
prob <- pmvnorm(lower, upper, mean, corr)
print(prob)

stopifnot(pmvnorm(lower=-Inf, upper=3, mean=0, sigma=1) == pnorm(3))

a <- pmvnorm(lower=-Inf,upper=c(.3,.5),mean=c(2,4),diag(2))

stopifnot(round(a,16) == round(prod(pnorm(c(.3,.5),c(2,4))),16))

a <- pmvnorm(lower=-Inf,upper=c(.3,.5,1),mean=c(2,4,1),diag(3))

stopifnot(round(a,16) == round(prod(pnorm(c(.3,.5,1),c(2,4,1))),16))

# Example from R News paper (original by Genz, 1992):

m <- 3
sigma <- diag(3)
sigma[2,1] <- 3/5
sigma[3,1] <- 1/3
sigma[3,2] <- 11/15
pmvnorm(lower=-Inf, upper=c(1,4,2), mean=rep(0, m), sigma)


## Keywords: 'distribution'.


rm(list = ls(all = TRUE)); .Random.seed <- c(0,rep(7654,3))
###--- >>> `pmvt' <<<----- Multivariate t Distribution

	## alias	 help(pmvt)

##___ Examples ___:


n <- 5
lower <- -1
upper <- 3
df <- 4
corr <- diag(5)
corr[lower.tri(corr)] <- 0.5
delta <- rep(0, 5)
prob <- pmvt(lower=lower, upper=upper, delta=delta,df=df, corr=corr)
print(prob)

pmvt(-Inf, 3, df = 3, sigma = 0) == pt(3, 3)

# Example from R News paper (original by Edwards and Berry, 1987)

n <- c(26, 24, 20, 33, 32)
V <- diag(1/n)
df <- 130
C <- c(1,1,1,0,0,-1,0,0,1,0,0,-1,0,0,1,0,0,0,-1,-1,0,0,-1,0,0)
C <- matrix(C, ncol=5)
cv <- C %*% V %*% t(C)
cr <- matrix(rep(0, ncol(cv)^2), ncol=ncol(cv))
for (i in 1:5) {
  for (j in 1:5) {
    cr[i,j] <- cv[i,j]/sqrt(cv[i,i]*cv[j,j] )
  }
}
delta <- rep(0,5)

myfct <- function(q, alpha) {
  lower <- rep(-q, ncol(cv))
  upper <- rep(q, ncol(cv))
  pmvt(lower, upper, delta, df, cr, abseps=0.0001) - alpha
}

round(uniroot(myfct, lower=1, upper=5, alpha=0.95)$root, 3)

# compare pmvt and pmvnorm for large df:

a <- pmvnorm(-Inf, 1, mean=rep(0, 5), corr=diag(5))
b <- pmvt(-Inf, 1, df=rep(300,5), corr=diag(5), delta=rep(0, 5))
a
b

stopifnot(round(a, 2) == round(b, 2))


## Keywords: 'distribution'.


cat("Time elapsed: ", proc.time() - get("ptime", env = .CheckExEnv),"\n")
dev.off(); quit('no')
