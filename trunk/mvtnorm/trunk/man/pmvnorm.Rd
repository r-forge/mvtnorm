\name{pmvnorm}
\alias{pmvnorm}
\title{ Multivariate Normal Distribution }
\description{

Computes the distribution function of the multivariate normal 
distribution for arbitary limits and correlation matrices 
based on algorithms by Genz and Bretz.

}
\usage{
pmvnorm <- function(mean, corr, lower, upper, maxpts = 25000, abseps =0,
releps = 0.001)
}
\arguments{
  \item{mean}{ the mean vector.}
  \item{corr}{ the correlation matrix.}
  \item{lower}{ the vector of lower limits.}
  \item{upper}{ the vector of upper limits.}
  \item{maxpts}{ maximum number of function values. }
  \item{abseps}{ absolute error tolerance. }
  \item{releps}{ relative error tolerance. }
}
\details{

This program involves the computation of 
multivariate normal-probabilities with arbitrary correlation matrices.
It involves both the computation of singular and nonsingular 
probabilities. The methodology is described in
Genz and Bretz (1999) and Genz and Bretz (2000).

Note that both \code{-Inf} and \code{+Inf} may be specified in \code{lower} and
\code{upper}. For more details see \link{pmvt}. 

Multivariate normal density and random numbers are available using
\link[e1071]{dmvnorm} and \link[MASS]{mvrnorm}.  
 
}
\value{
A list with the following components:
  \item{value}{estimated integral value.}
  \item{error}{estimated absolute error.}
  \item{msg}{status messages.}
}
\references{ 

Genz, A. and Bretz, F. (1999), Numerical computation of multivariate
t-probabilities with application to power calculation of multiple
contrasts. \emph{Journal of Statistical Computation and Simulation},
\bold{63}, 361--378.

Genz, A. and Bretz, F. (2001), Methods for the computation of multivariate
t-probabilities. \emph{(submitted)}

}

\author{ Fortran Code by Alan Genz <AlanGenz@wsu.edu> and Frank Bretz
<bretz@ifgb.uni-hannover.de>, R port by Torsten Hothorn
<Torsten.Hothorn@rzmail.uni-erlangen.de> }

\examples{

n <- 5
mean <- rep(0, 5)
lower <- rep(-1, 5)
upper <- rep(3, 5)
corr <- diag(5)
corr[lower.tri(corr)] <- 0.5
prob <- pmvnorm(mean, corr, lower, upper)
print(prob)

}

\keyword{multivariate normal distribution}
