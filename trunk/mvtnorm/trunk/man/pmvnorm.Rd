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
releps = 0.005)
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
To add. 
}
\value{
A list with the following components:
  \item{value}{estimated integral value.}
  \item{error}{estimated absolute error.}
  \item{msg}{status messages.}
}
\references{ Add the articles. }

\author{ Fortran Code by Alan Genz <AlanGenz@wsu.edu>, Frank Bretz
<bretz@ifgb.uni-hannover.de>, R port by Torsten Hothorn
<Torsten.Hothorn@rzmail.uni-erlangen.de> }

\examples{

# ok, first of all a little univariate test: 

prob <- pmvnorm(mean = 3, corr = 1, lower = -Inf, upper = 3)
print(prob$value - pnorm(3,3))

# now something more meaningful

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
