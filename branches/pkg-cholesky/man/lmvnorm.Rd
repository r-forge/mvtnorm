\name{lmvnorm}
\alias{lmvnorm}
\title{
    Multivariate Normal Log-likelihood
}
\description{
    Computes the log-likelihood (contributions) of multiple
    interval-censored observations from a multivariate normal.
}
\usage{
lmvnorm(lower, upper, mean = 0, chol, logLik = TRUE, M = NULL, w = NULL, seed = NULL)
}
\arguments{
  \item{lower}{matrix of lower limits (one column for each observation).
}
  \item{upper}{matrix of upper limits (one column for each observation).
}
  \item{mean}{matrix of means (one column for each observation, length is
recycled to length of \code{lower} and \code{upper}).
}
  \item{chol}{Cholesky factors of covariance matrices as
    \code{\link{ltMatrices}} object, length is recylced to length of \code{lower} and \code{upper}
}
  \item{logLik}{logical, if \code{TRUE}, the log-likelihood is returned,
otherwise the individual contributions to the sum are returned.
}
  \item{M}{number of iterations, early stopping based on
estimated errors is NOT implemented.
}
  \item{w}{an optional matrix of weights. This allows to replace the default
Monte-Carlo procedure (Genz, 1992) with a quasi-Monte-Carlo approach (Genz &
Bretz, 2002). Note that the same weights for evaluating the
multivariate normal probability are used for all observations when \code{w}
is specified. If \code{w} is \code{NULL}, different uniform numbers are
drawn for each observation.
}
  \item{seed}{an object specifying if and how the random number generator
          should be initialized, see \code{\link[stats]{simulate}}. Only 
          applied when \code{w} is \code{NULL}.
}
}
\details{
  Evaluates the multivariate normal law defined by \code{means} and
  \code{chol} over boxes defined by \code{lower} and \code{upper}. By
  default, the log-likelihood is returned.

  Monte-Carlo (Genz, 1992, the default) and quasi-Monte-Carlo (Genz & Bretz, 2002)
  integration is implemented, the latter with weights obtained from package
  \pkg{randtoolbox}.

  Details can be found in the \code{lmvnorm_src} package vignette.

}
\value{
The log-likelihood (\code{likLik = TRUE}) or the individual contributions to the log-likelihood.
}
\references{

  Genz, A. (1992). Numerical computation of multivariate normal probabilities.
  \emph{Journal of Computational and Graphical Statistics}, \bold{1}, 141--150.

  Genz, A. and Bretz, F. (2002), Methods for the computation of multivariate
  t-probabilities. \emph{Journal of Computational and Graphical Statistics},
  \bold{11}, 950--971.

}
\examples{

  ### five observations
  N <- 5L
  ### dimension
  J <- 4L

  ### lower and upper bounds, ie interval-censoring
  lwr <- matrix(-runif(N * J), nrow = J)
  upr <- matrix(runif(N * J), nrow = J)

  ### Cholesky factor
  (C <- ltMatrices(runif(J * (J + 1) / 2), diag = TRUE))
  ### corresponding covariance matrix
  (S <- as.array(Tcrossprod(C))[,,1])

  ### plain Monte-Carlo (Genz, 1992)
  w <- NULL
  ### quasi-Monte-Carlo (Genz & Bretz, 2002, but with different weights)
  if (require("randtoolbox")) w <- t(halton(25000, J - 1))

  ### log-likelihood
  lmvnorm(lower = lwr, upper = upr, chol = C, w = w)
  
  ### compare with pmvnorm
  exp(lmvnorm(lower = lwr, upper = upr, chol = C, logLik = FALSE, w = w))
  sapply(1:N, function(i) pmvnorm(lower = lwr[,i], upper = upr[,i], sigma = S))

}
\keyword{distribution}
