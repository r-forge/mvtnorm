# $Id$ 
checknormArgs <- function(lower, upper, mean, corr, sigma) 
{
    UNI <- FALSE
    if (is.null(lower) || any(is.na(lower)))
        stop("lower not specified or contains NA")
    if (is.null(upper) || any(is.na(lower)))
        stop("upper not specified or contains NA")
    if (length(lower) != length(upper))
        stop("lower and upper are of different length")
    if (is.null(mean)) {
        mean <- rep(0, length(lower))
        warning("mean not specified: using rep(0, length(lower))")
    }
    if (any(is.na(mean)))
        stop("mean contains NA")
    if (length(mean) != length(lower))
        stop("mean and lower are of different lenght")
    if (is.null(corr) && is.null(sigma)) {
        corr <- diag(length(lower))
        warning("both corr and sigma not specified: using diag(length(lower))")
    }
    if (!is.null(corr) && !is.null(sigma)) {
        sigma <- NULL
        warning("both corr and sigma specified: ignoring sigma")
    }
    if (!is.null(corr)) {
         if (!is.matrix(corr)) {
             if (length(sigma) == 1)
                UNI <- TRUE
             if (length(corr) != length(lower))
               stop("diag(corr) and lower are of diffenent length")
         } else {
             if (length(diag(corr)) != length(lower))        
               stop("diag(corr) and lower are of diffenent length")
         }
    }
    if (!is.null(sigma)) {
         if (!is.matrix(sigma)) {
            if (length(sigma) == 1)
                UNI <- TRUE
            if (length(sigma) != length(lower))        
               stop("diag(sigma) and lower are of diffenent length")
         } else {
            if (length(diag(sigma)) != length(lower))                     
               stop("diag(sigma) and lower are of diffenent length")
         }
    }
    list(lower=lower, upper=upper, mean=mean, corr=corr, sigma=sigma, uni=UNI)
}


pmvnorm <- function(lower=-Inf, upper=Inf, mean=rep(0, length(lower)), corr=NULL, sigma=NULL,
                    maxpts = 25000, abseps = 0.001, releps = 0)
{
    carg <- checknormArgs(lower=lower, upper=upper, mean=mean, corr=corr,
                      sigma=sigma)
    if (!is.null(corr)) {
      if (carg$uni) {
          stop("sigma not specified: cannot compute pnorm")
      } else {
          lower <- carg$lower - carg$mean
          upper <- carg$upper - carg$mean
          mean <- rep(0, length(lower))
          RET <- mvt(lower=lower, upper=upper, df=0, corr=corr, delta=mean,
                     maxpts=maxpts, abseps=abseps,releps=releps)
      }
    } else {
      if (carg$uni) {
        RET <- list(value = pnorm(upper, mean=mean, sd=sqrt(sigma)) -
                            pnorm(lower, mean=mean, sd=sqrt(sigma)),
                    error = 0, msg="univariate: using pnorm")
      } else {
          lower <- (carg$lower - carg$mean)/sqrt(diag(sigma))
          upper <- (carg$upper - carg$mean)/sqrt(diag(sigma))
          mean <- rep(0, length(lower))
          corr <- t(t(sigma)/sqrt(diag(sigma)))
          RET <- mvt(lower=lower, upper=upper, df=0, corr=corr, delta=mean,
                     maxpts=maxpts, abseps=abseps,releps=releps)
      }
    }
    attr(RET$value, "error") <- RET$error
    attr(RET$value, "msg") <- RET$msg
    return(RET$value)
}


checktArgs <- function(lower, upper, df, corr, sigma, delta) 
{
    UNI <- FALSE
    if (is.null(df))
        stop("df not specified")
    if (df < 1)
        stop("cannot compute multivariate t distribution with df < 1")	
    if (is.null(lower) || any(is.na(lower)))
        stop("lower not specified or contains NA")
    if (is.null(upper) || any(is.na(lower)))
        stop("upper not specified or contains NA")
    if (length(lower) != length(upper))
        stop("lower and upper are of different length")
    if (is.null(delta)) {
        delta <- rep(0, length(lower))
        warning("delta not specified: using rep(0, length(lower))")
    }
    if (any(is.na(delta)))
        stop("delta contains NA")
    if (length(delta) != length(lower))
        stop("delta and lower are of different lenght")
    if (is.null(corr) && is.null(sigma)) {
        corr <- diag(length(lower))
        warning("both corr and sigma not specified: using diag(length(lower))")
    }
    if (!is.null(corr) && !is.null(sigma)) {
        sigma <- NULL
        warning("both corr and sigma specified: ignoring sigma")
    }
    if (!is.null(corr)) {
         if (!is.matrix(corr)) {
             if (length(sigma) == 1)
                UNI <- TRUE
             if (length(corr) != length(lower))
               stop("diag(corr) and lower are of diffenent length")
         } else {
             if (length(diag(corr)) != length(lower))        
               stop("diag(corr) and lower are of diffenent length")
         }
    }
    if (!is.null(sigma)) {
         if (!is.matrix(sigma)) {
            if (length(sigma) == 1)
                UNI <- TRUE
            if (length(sigma) != length(lower))        
               stop("diag(sigma) and lower are of diffenent length")
         } else {
            if (length(diag(sigma)) != length(lower))                     
               stop("diag(sigma) and lower are of diffenent length")
         }
    }
    list(lower=lower, upper=upper, df=df, corr=corr, sigma=sigma,
         delta=delta, uni=UNI)
}

pmvt <- function(lower=-Inf, upper=Inf, df=1, corr=NULL, sigma=NULL,
                 delta=rep(0, length(lower)), maxpts = 25000, abseps = 0.001, releps = 0)
{
    carg <- checktArgs(lower=lower, upper=upper, df=df, corr=corr,
                       sigma=sigma, delta=delta)
    if (!is.null(corr)) {
      if (carg$uni) {
          stop("sigma not specified: cannot compute pt")
      } else {
          RET <- mvt(lower=lower, upper=upper, df=df, corr=corr, delta=delta,
                     maxpts=maxpts, abseps=abseps,releps=releps)
      }
    } else {
      if (carg$uni) {
          RET <- list(value = pt(upper, df=df, ncp=delta) -
                            pt(lower, df=df, ncp=delta),
                    error = 0, msg="univariate: using pt")
      } else {
          lower <- carg$lower/sqrt(diag(sigma))
          upper <- carg$upper/sqrt(diag(sigma))
          corr <- t(t(sigma)/sqrt(diag(sigma)))
          RET <- mvt(lower=lower, upper=upper, df=df, corr=corr, delta=delta,
                     maxpts=maxpts, abseps=abseps,releps=releps)
      }
    }
    attr(RET$value, "error") <- RET$error
    attr(RET$value, "msg") <- RET$msg
    return(RET$value)
}


mvt <- function(lower, upper, df, corr, delta, maxpts = 25000,
                abseps = 0.001, releps = 0)
{
    n <- ncol(corr)
    if (is.null(n) || n < 2) stop("dimension less then n = 2")

    if (length(lower) != n) stop("wrong dimensions")
    if (length(upper) != n) stop("wrong dimensions")

    if (n > 1000) stop("only dimensions 1 <= n <= 100 allowed") 

    infin <- rep(2, n)
    infin[upper == Inf] <- 1
    infin[lower == -Inf] <- 0
    infin[lower == -Inf & upper == Inf] <- -1
    
    if (n > 1) {
        corrF <- matrix(as.vector(corr), ncol=n, byrow=T)
        corrF <- corrF[upper.tri(corrF)]
    } else corrF <- corr 

    lower[lower == -Inf] <- 0
    upper[upper == Inf] <- 0

    error <- 0; value <- 0; inform <- 0

    ret <- .Fortran("mvtdst", as.integer(n), as.integer(df),
                        as.double(lower), as.double(upper), as.integer(infin),
                        as.double(corrF), as.double(delta), as.integer(maxpts),
                        as.double(abseps), as.double(releps),
                        error = as.double(error), value = as.double(value),
                        inform = as.integer(inform))
    
    error <- ret$error; value <- ret$value; inform <- ret$inform

    msg <- NULL
    if (inform == 0) msg <- "Normal Completion"
    if (inform == 1) msg <- "Completion with error > abseps"
    if (inform == 3) msg <- "Covariance matrix not positive semidefinite"
    if (is.null(msg)) msg <- inform
    
    RET <- list(value = value, error = error, msg = msg)
    return(RET)
}
