\name{pmvt}
\alias{pmvt}
\title{ Multivariate t Distribution }
\description{
Computes the the distribution function of the multivariate t distribution 
for arbitary limits, degrees of freedom and correlation matrices 
based on algorithms by Genz and Bretz.
}
\usage{
pmvt(lower, upper, df, corr, delta, maxpts=25000, abseps=0, releps=0.005)
}
\arguments{
  \item{lower}{ the vector of lower limits.}
  \item{upper}{ the vector of upper limits.}
  \item{df}{ degree of freedom as integer.}
  \item{corr}{ the correlation matrix.}
  \item{delta}{ the vector of noncentrality parameters.}
  \item{maxpts}{ maximum number of function values }
  \item{abseps}{ absolute error tolerance }
  \item{releps}{ relative error tolerance }
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

prob <- pmvt(lower = -Inf, upper = 3, df = 3, corr = 1, delta = 0)
print(prob$value - pt(3,3))

# now something more meaningful

n <- 5
lower <- rep(-1, 5)
upper <- rep(3, 5)
df <- 4
corr <- diag(5)
corr[lower.tri(corr)] <- 0.5
delta <- rep(0, 5)
prob <- pmvt(lower, upper, df, corr , delta)
print(prob)

}

\keyword{mutivariate t distribution}