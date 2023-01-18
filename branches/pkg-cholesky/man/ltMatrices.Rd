\name{ltMatrices}
\alias{ltMatrices}
\alias{Tcrossprod}
\alias{Crossprod}
\alias{diagonals}
\alias{Mult}
\alias{solve.ltMatrices}
\alias{as.array.ltMatrices}
\alias{as.array.syMatrices}
\alias{diagonals.ltMatrices}
\alias{diagonals.syMatrices}
\alias{diagonals.matrix}
\alias{chol.syMatrices}
\title{
  Multiple Lower Triangular Matrices
}
\description{
  A class representing multiple lower triangular matrices and some methods
}
\usage{
ltMatrices(object, diag = FALSE, byrow = FALSE, trans = FALSE, names = TRUE)
\S3method{as.array}{ltMatrices}(x, symmetric = FALSE, \dots)
\S3method{as.array}{syMatrices}(x, \dots)
\S3method{diagonals}{ltMatrices}(x, \dots)
\S3method{diagonals}{syMatrices}(x, \dots)
\S3method{diagonals}{matrix}(x, \dots)
Mult(x, y)
\S3method{solve}{ltMatrices}(a, b, \dots)
Tcrossprod(x, diag_only = FALSE, transpose = FALSE)
Crossprod(x, diag_only = FALSE)
\S3method{chol}{syMatrices}(x, \dots)
}
\arguments{
  \item{object}{a \code{matrix} representing the lower triagular elements of
                \eqn{N} lower triangular matrix, each of dimension \eqn{J \times
                J}. Dimensions of \code{object} depend on \code{trans} and \code{diag}.
}
  \item{diag}{logical, \code{object} contains diagonal elements if
              \code{TRUE}, otherwise unit diagonal elements are assumed.
}
  \item{byrow}{logical, \code{object} represents matrices in row-major order
               if \code{TRUE} or, otherwise, in column-major order.
}
  \item{trans}{logical, \code{ncol(object)} is \eqn{N} if \code{TRUE},
               otherwise, \code{ncol(object)} is \eqn{N}.
}
  \item{names}{logical or character vector of length \eqn{J}.
}
  \item{symmetric}{logical, object is interpreted as a symmetric matrix if
\code{TRUE}.
}
  \item{diag_only}{logical, compute diagonal elements of crossproduct only
if \code{TRUE}.
}
  \item{transpose}{logical, compute \eqn{x^\top x} (one should use
\code{Crossprod}) instead of \eqn{x x^\top} (the default in
\code{Tcrossprod}). 
}
  \item{x}{object of class \code{ltMatrices} or \code{syMatrices} (for \code{chol}).
}
  \item{a}{object of class \code{ltMatrices}.
}
  \item{y}{matrix with \eqn{J} rows.
}
  \item{b}{matrix with \eqn{J} rows.
}
  \item{\dots}{additional arguments, currently ignored.
}
}
\details{
\code{ltMatrices} interprets a matrix as lower triangular elements of
multiple lower triangular matrices. The corresponding class can be used to
store such matrices efficiently. Matrix multiplications, solutions to linear
systems, explicite inverses, and crossproducts can be computed based on such
objects. Details can be found in the \code{lmvnorm_src} package vignette.
}
\value{
The constructor \code{ltMatrices} returns objects of class \code{ltMatrices}
with corresponding methods.
}
\examples{

  J <- 4L
  N <- 2L
  dm <- paste0("d", 1:J)
  xm <- paste0("x", 1:N)
  (C <- ltMatrices(matrix(runif(N * J * (J + 1) / 2), 
                          nrow = N, dimnames = list(xm, NULL)), 
                   diag = TRUE, names = dm))

  ## dimensions and names
  dim(C)
  dimnames(C)
  names(C)

  ## subset
  C[,2:3]

  ## multiplication
  y <- matrix(runif(N * J), nrow = J)
  Mult(C, y)

  ## solve
  solve(C)
  solve(C, y)

  ## tcrossprod
  Tcrossprod(C)

  ## convert to matrix
  as.array(solve(C[1,]))[,,1]

}
\keyword{matrix}
