
library(mvtnorm)

n <- c(26, 24, 20, 33, 32)
V <- diag(1/n)
df <- 130
C <- c(1,1,1,0,0,-1,0,0,1,0,0,-1,0,0,1,0,0,0,-1,-1,0,0,-1,0,0)
C <- matrix(C, ncol=5)
cv <- C%*%V%*%t(C)
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
  pmvt(lower, upper, df, cr, delta, abseps=0.0001)$value - alpha
}

uniroot(myfct, lower=1, upper=5, alpha=0.95)
