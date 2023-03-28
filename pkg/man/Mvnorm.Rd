\name{Mvnorm}
\alias{dmvnorm}
\alias{sldmvnorm}
\alias{rmvnorm}
\title{Multivariate Normal Density and Random Deviates}
\description{
  These functions provide the density function and a random number
  generator for the multivariate normal
  distribution with mean equal to \code{mean} and covariance matrix
  \code{sigma}.
}
\usage{
dmvnorm(x, mean = rep(0, p), sigma = diag(p), log = FALSE, checkSymmetry = TRUE, chol, invchol)
sldmvnorm(x, mean = 0, chol, invchol) 
rmvnorm(n, mean = rep(0, nrow(sigma)), sigma = diag(length(mean)),
        method=c("eigen", "svd", "chol"), pre0.9_9994 = FALSE, checkSymmetry = TRUE)
}
\arguments{
 \item{x}{vector or matrix of quantiles. If \code{x} is a matrix, each
   row is taken to be a quantile and columns correspond to the number of dimensions, \code{p}.}
 \item{n}{number of observations.}
 \item{mean}{mean vector, default is \code{rep(0, length = ncol(x))}.}
 \item{sigma}{covariance matrix, default is \code{diag(ncol(x))}.}
 \item{log}{logical; if \code{TRUE}, densities d are given as log(d).}
 \item{method}{string specifying the matrix decomposition used to
   determine the matrix root of \code{sigma}.  Possible methods are
   eigenvalue decomposition (\code{"eigen"}, default),
   singular value decomposition (\code{"svd"}), and
   Cholesky decomposition (\code{"chol"}).  The
   Cholesky is typically fastest, not by much though.}
 \item{pre0.9_9994}{logical; if \code{FALSE}, the output produced in mvtnorm
                    versions up to 0.9-9993 is reproduced. In 0.9-9994, the
                    output is organized such that \code{rmvnorm(10,...)} has the
                    same first ten rows as \code{rmvnorm(100, ...)} when called
                    with the same seed.}
 \item{checkSymmetry}{logical; if \code{FALSE}, skip checking whether the
   covariance matrix is symmetric or not. This will speed up the
   computation but may cause unexpected outputs when ill-behaved
   \code{sigma} is provided. The default value is \code{TRUE}.}
 \item{chol}{an object of class \code{\link{ltMatrices}} specifying
             potentially observation-specific covariance matrices via
             their lower-triagular Cholesky factors.}
 \item{invchol}{an object of class \code{\link{ltMatrices}} specifying
             potentially observation-specific covariance matrices via
             their inverse lower-triagular Cholesky factors (that is, a
             factorisation of the precision matrices).}

}
\details{
\code{dmvnorm} computes the density function of the multivariate normal
specified by mean and the covariance matrix \code{sigma}. Alternatively,
Cholesky factors of \code{sigma} (argument \code{chol}) or its inverse
(argument \code{invchol}) can be used to specify the covariance structure.
In the two latter cases, means and covariances might differ for each row of
\code{x}. \code{slmvnorm} computes the gradients of the log-density with
respect to \code{x} and \code{chol} (or \code{invchol}).

\code{rmvnorm} generates multivariate normal variables. See the package
vignette linked to below for simpler ways to do so based on \code{chol} or
\code{invchol}.

}
\seealso{\code{\link{pmvnorm}}, \code{\link{rnorm}}, \code{\link{qmvnorm}},
         \code{vignette("lmvnorm_src", package = "mvtnorm")}}
\examples{
dmvnorm(x=c(0,0))
dmvnorm(x=c(0,0), mean=c(1,1))

sigma <- matrix(c(4,2,2,3), ncol=2)
x <- rmvnorm(n=500, mean=c(1,2), sigma=sigma)
colMeans(x)
var(x)

x <- rmvnorm(n=500, mean=c(1,2), sigma=sigma, method="chol")
colMeans(x)
var(x)

plot(x)
}
\keyword{distribution}
\keyword{multivariate}
