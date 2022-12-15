\documentclass[a4paper]{report}

%% packages
\usepackage{amsfonts,amstext,amsmath,amssymb,amsthm}

%\VignetteIndexEntry{Multivariate Normal Log-likelihoods}
%\VignetteDepends{mvtnorm,qrng}
%\VignetteKeywords{multivariate normal distribution}
%\VignettePackage{mvtnorm}


\usepackage[utf8]{inputenc}

\newif\ifshowcode
\showcodetrue

\usepackage{latexsym}
%\usepackage{html}

\usepackage{listings}

\usepackage{color}
\definecolor{linkcolor}{rgb}{0, 0, 0.7}

\usepackage[%
backref,%
raiselinks,%
pdfhighlight=/O,%
pagebackref,%
hyperfigures,%
breaklinks,%
colorlinks,%
pdfpagemode=None,%
pdfstartview=FitBH,%
linkcolor={linkcolor},%
anchorcolor={linkcolor},%
citecolor={linkcolor},%
filecolor={linkcolor},%
menucolor={linkcolor},%
pagecolor={linkcolor},%
urlcolor={linkcolor}%
]{hyperref}

\usepackage[round]{natbib}

\setlength{\oddsidemargin}{0in}
\setlength{\evensidemargin}{0in}
\setlength{\topmargin}{0in}
\addtolength{\topmargin}{-\headheight}
\addtolength{\topmargin}{-\headsep}
\setlength{\textheight}{8.9in}
\setlength{\textwidth}{6.5in}
\setlength{\marginparwidth}{0.5in}

\newcommand{\pkg}[1]{\textbf{#1}}
\newcommand{\proglang}[1]{\textsf{#1}}
\newcommand{\code}[1]{\texttt{#1}}
\newcommand{\cmd}[1]{\texttt{#1()}}

\newcommand{\R}{\mathbb{R} }
\newcommand{\Prob}{\mathbb{P} }
\newcommand{\N}{\mathbb{N} }
\newcommand{\J}{J}
\newcommand{\V}{\mathbb{V}} %% cal{\mbox{\textnormal{Var}}} }
\newcommand{\E}{\mathbb{E}} %%mathcal{\mbox{\textnormal{E}}} }
\newcommand{\yvec}{\mathbf{y}}
\newcommand{\avec}{\mathbf{a}}
\newcommand{\bvec}{\mathbf{b}}
\newcommand{\xvec}{\mathbf{x}}
\newcommand{\rY}{\mathbf{Y}}
\newcommand{\rZ}{\mathbf{Z}}
\newcommand{\mC}{\mathbf{C}}
\newcommand{\mL}{\mathbf{L}}
\newcommand{\mI}{\mathbf{I}}
\newcommand{\argmin}{\operatorname{argmin}\displaylimits}
\newcommand{\argmax}{\operatorname{argmax}\displaylimits}


\author{Torsten Hothorn \\ Universit\"at Z\"urich}

\title{Multivariate Normal Log-likelihoods in the \pkg{mvtnorm} Package}

\begin{document}

\pagenumbering{roman}
\maketitle
\tableofcontents

\chapter*{Licence}

{\setlength{\parindent}{0cm}
Copyright (C) 2022-- Torsten Hothorn \\

This file is part of the \pkg{mvtnorm} \proglang{R} add-on package. \\

\pkg{mvtnorm} is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, version 2. \\

\pkg{mvtnorm} is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details. \\

You should have received a copy of the GNU General Public License
along with \pkg{mvtnorm}.  If not, see <http://www.gnu.org/licenses/>.
}

\chapter{Introduction}
\pagenumbering{arabic}

This document describes an implementation of \cite{numerical-:1992} and,
partially, of \cite{Genz_Bretz_2002}, for the  evaluation of
$N$ multivariate $\J$-dimensional normal probabilities
\begin{eqnarray} \label{pmvnorm}
p_i(\mC_i \mid \avec_i, \bvec_i) = \Prob(\avec_i < \rY_i \le \bvec_i \mid \mC_i ) 
  = (2 \pi)^{-\frac{\J}{2}} \text{det}(\mC_i)^{-\frac{1}{2}} 
    \int_{\avec_i}^{\bvec_i} \exp\left(-\frac{1}{2} \yvec^\top \mC_i^{-\top} \mC_i^{-1} \yvec\right) \, d \yvec
\end{eqnarray}
where $\avec_i = (a^{(i)}_1, \dots, a^{(i)}_\J)^\top \in \R^\J$ and 
$\bvec_i = (b^{(i)}_1, \dots, b^{(i)}_\J)^\top \in \R^\J$ are integration
limits, $\mC_i = (c^{(i)}_{j\jmath}) \in \R^{\J \times
\J}$ is a lower triangular matrix with $c^{(i)}_{j \jmath} = 0$ for $1 \le
j < \jmath < \J$, and thus $\rY_i \sim \N_\J(\mathbf{0}_\J, \mC_i \mC_i^\top)$ for $i = 1, \dots, N$.

One application of these integrals is the estimation of the Cholesky factor
$\mC$ of a $\J$-dimensional normal distribution based on $N$ interval-censored
observations $\rY_1, \dots, \rY_\J$ (encoded by $\avec$ and $\bvec$) via maximum-likelihood
\begin{eqnarray*}
\hat{\mC} = \argmax_\mC \sum_{i = 1}^N \log(p_i(\mC \mid \avec_i, \bvec_i)).
\end{eqnarray*}
In other applications, the Cholesky factor might also depend on $i$ in some
structured way.

Function \code{pmvnorm} in package \code{mvtnorm} computes $p_i$ based on
the covariance matrix $\mC_i \mC_i^\top$. However, the Cholesky $\mC_i$ is
computed in \proglang{FORTRAN}. Function \code{pmvnorm} is not vectorised
over $i = 1, \dots, N$ and thus separate calls to this function are
necessary in order to compute likelihood contributions.

The implementation described here is expected to be inferior to Alan Genz'
original \code{FORTRAN} code when accuracy for single $p_i$ matters. We cut
some corners aiming at efficient computation of the log-likelihood $\sum_{i
= 1}^N \log(p_i)$.

The document first describes infrastructure, that is, a class and useful
methods, for dealing with multiple lower triangular matrices $\mC_i, i = 1, \dots,
N$ in Chapter~\ref{ltMatrices}. The multivariate normal log-likelihood is
implemented as outlined in Chapter~\ref{lmvnorm}. An example demonstrating
maximum-likelihood estimation of Cholesky factors in the presence of
interval-censored observations is discussed last in Chapter~\ref{ML}.

\chapter{Lower Triangular Matrices} \label{ltMatrices}

@o ltMatrices.R -cp
@{
@<R Header@>
@<ltMatrices@>
@<dim ltMatrices@>
@<dimnames ltMatrices@>
@<names ltMatrices@>
@<print ltMatrices@>
@<transpose ltMatrices@>
@<reorder ltMatrices@>
@<subset ltMatrices@>
@<diagonals ltMatrices@>
@<mult ltMatrices@>
@<solve ltMatrices@>
@<tcrossprod ltMatrices@>
@}

@o ltMatrices.c -cc
@{
@<C Header@>
#include <R.h>
#include <Rmath.h>
#include <Rinternals.h>
#include <Rdefines.h>
#include <Rconfig.h>
#include <R_ext/Lapack.h> /* for dtptri */
@<solve@>
@<tcrossprod@>
@<mult@>
@}


We first need infrastructure for dealing with multiple lower triangular matrices
$\mC_i \in \R^{\J \times \J}$ for $i = 1, \dots, N$. We note that each such matrix
$\mC$ can be stored in a vector of length $\J (\J + 1) / 2$. If all
diagonal elements are one (that is, $c^{(i)}_{jj} \equiv 1, j = 1, \dots,
\J$), the length of this vector is $\J (\J - 1) / 2$.

\section{Multiple lower triangular matrices}

We can store $N$ such matrices in an $N \times \J (\J + 1) / 2$  (\code{diag = TRUE})
or $N \times \J (\J - 1) / 2$ matrix (\code{diag = FALSE}). Sometimes it is
more convenient to store the transposed $\J (\J + 1) / 2 \times N$ matrix
(\code{trans = TRUE}, \code{diag = TRUE}) or, for \code{diag = FALSE}, the $\J (\J
- 1) / 2 \times N$ matrix.

Each vector might define the corresponding lower triangular matrix
either in row or column-major order:

\begin{eqnarray*}
 \mC & = & \begin{pmatrix}
 c_{11} & & & & 0\\
 c_{21} & c_{22} \\
 c_{31} & c_{32} & c_{33} \\
 \vdots & \vdots & & \ddots & \\
 c_{J1} & c_{J2} & \ldots & &  c_{JJ}
 \end{pmatrix}  \text{matrix indexing}\\
& = &  
\begin{pmatrix}
 c_{1} & & & & 0\\
 c_{2} & c_{J + 1} \\
 c_{3} & c_{J + 2} & c_{2J} \\
 \vdots & \vdots & & \ddots & \\
 c_{J} & c_{2J - 1} & \ldots & &  c_{J(J + 1) / 2}
 \end{pmatrix} \text{column-major, \code{byrow = FALSE}} \\
& = & \begin{pmatrix}
 c_{1} & & & & 0\\
 c_{2} & c_{3} \\
 c_{4} & c_{5} & c_{6} \\
 \vdots & \vdots & & \ddots & \\
 c_{J((J + 1) / 2 -1) + 1} & c_{J((J + 1) / 2 -1) + 2} & \ldots & &  c_{J(J + 1) / 2}
 \end{pmatrix} \text{row-major, \code{byrow = TRUE}}
\end{eqnarray*}

Based on some matrix \code{object}, the dimension $\J$ is computed and checked as
@d ltMatrices dim
@{
J <- floor((1 + sqrt(1 + 4 * 2 * ifelse(trans, nrow(object), ncol(object)))) / 2 - diag)
stopifnot(ifelse(trans, nrow(object), ncol(object)) == J * (J - 1) / 2 + diag * J)
@}

Typically the $\J$ dimensions are associated with names, and we therefore
compute identifiers for the vector elements in either column- or row-major
order on request (for later printing)

@d ltMatrices names
@{
nonames <- FALSE
if (!isTRUE(names)) {
    if (is.character(names))
        stopifnot(is.character(names) &&
                  length(unique(names)) == J)
    else
        nonames <- TRUE
} else {
    names <- as.character(1:J)
}

if (!nonames) {
    L1 <- matrix(names, nrow = J, ncol = J)
    L2 <- matrix(names, nrow = J, ncol = J, byrow = TRUE)
    L <- matrix(paste(L1, L2, sep = "."), nrow = J, ncol = J)
    if (trans) {
        if (byrow)
            rownames(object) <- t(L)[upper.tri(L, diag = diag)]
        else
            rownames(object) <- L[lower.tri(L, diag = diag)]
    } else {
        if (byrow)
            colnames(object) <- t(L)[upper.tri(L, diag = diag)]
        else
            colnames(object) <- L[lower.tri(L, diag = diag)]
    }
}
@}

If \code{object} is already a classed object representing lower triangular
matrices (we will use the class name \code{ltMatrices}), we might want to
change the storage form or transpose the underlying matrix.

@d ltMatrices input
@{
if (inherits(object, "ltMatrices")) {
    ret <- .reorder(object, byrow = byrow)
    ret <- .transpose(ret, trans = trans)
    return(ret)
}
@}

The constructor essentially attaches attributes to a matrix \code{object},
possibly after some reordering / transposing

@d ltMatrices
@{
ltMatrices <- function(object, diag = FALSE, byrow = FALSE, trans = FALSE, names = TRUE) {

    if (!is.matrix(object) && trans) 
        object <- matrix(object, ncol = 1L)
    if (!is.matrix(object) && !trans) 
        object <- matrix(object, nrow = 1L)

    @<ltMatrices input@>

    @<ltMatrices dim@>
    
    @<ltMatrices names@>

    attr(object, "J")       <- J
    attr(object, "diag")    <- diag
    attr(object, "byrow")   <- byrow
    attr(object, "trans")   <- trans
    attr(object, "rcnames") <- names

    class(object) <- c("ltMatrices", class(object))
    object
}
@}

Symmetric matrices are represented by lower triangular matrix objects, but
we change the class from \code{ltMatrices} to \code{syMatrices} (which
disables all functionality except printing and coersion to arrays).

The dimensions of such an object are always $N \times \J \times \J$
(regardless or \code{byrow} and \code{trans}), and are given by

@d dim ltMatrices
@{
dim.ltMatrices <- function(x) {
    J <- attr(x, "J")
    class(x) <- class(x)[-1L]
    return(c(ifelse(attr(x, "trans"), ncol(x), nrow(x)), J, J))
}
dim.syMatrices <- dim.ltMatrices
@}

The corresponding dimnames can be extracted as

@d dimnames ltMatrices
@{
dimnames.ltMatrices <- function(x) {
    if (attr(x, "trans"))
        return(list(colnames(unclass(x)), attr(x, "rcnames"), attr(x, "rcnames")))
    return(list(rownames(unclass(x)), attr(x, "rcnames"), attr(x, "rcnames")))
}
dimnames.syMatrices <- dimnames.ltMatrices
@}

The names identifying rows and columns in each $\mC_i$ are

@d names ltMatrices
@{
names.ltMatrices <- function(x) {
    if (attr(x, "trans"))
        return(rownames(unclass(x)))
    return(colnames(unclass(x)))
}
names.syMatrices <- names.ltMatrices
@}

Let's set-up an example for illustration:

<<example>>=
library("mvtnorm")

chk <- function(...) stopifnot(isTRUE(all.equal(...)))

set.seed(290875)
N <- 4
J <- 5
rn <- paste0("x", 1:N)
nm <- LETTERS[1:J]
Jn <- J * (J - 1) / 2
## data
xn <- matrix(runif(N * Jn), nrow = N, byrow = TRUE)
rownames(xn) <- rn
xd <- matrix(runif(N * (Jn + J)), nrow = N, byrow = TRUE)
rownames(xd) <- rn

(lxn <- ltMatrices(xn, byrow = TRUE, names = nm))
dim(lxn)
dimnames(lxn)
lxd <- ltMatrices(xd, byrow = TRUE, diag = TRUE, names = nm)
dim(lxd)
dimnames(lxd)

class(lxn) <- "syMatrices"
lxn
@@

\section{Printing}

For pretty printing, we coerse object of class \code{ltMatrices} to
\code{array}. The method has an \code{symmetric} argument forcing the lower
triangular matrix to by interpreted as a symmetric matrix.

@d extract slots
@{
diag <- attr(x, "diag")
byrow <- attr(x, "byrow")
trans <- attr(x, "trans")
d <- dim(x)
J <- d[2L]
dn <- dimnames(x)
@}

@d print ltMatrices
@{
as.array.ltMatrices <- function(x, symmetric = FALSE, ...) {

    @<extract slots@>

    class(x) <- class(x)[-1L]
    if (trans) x <- t(x)

    L <- matrix(1L, nrow = J, ncol = J)
    diag(L) <- 2L
    if (byrow) {
        L[upper.tri(L, diag = diag)] <- floor(2L + 1:(J * (J - 1) / 2L + diag * J))
        L <- t(L)
    } else {
        L[lower.tri(L, diag = diag)] <- floor(2L + 1:(J * (J - 1) / 2L + diag * J))
    }
    if (symmetric) {
        L[upper.tri(L)] <- 0L
        dg <- diag(L)
        L <- L + t(L)
        diag(L) <- dg
    }
    ret <- t(cbind(0, 1, x)[, c(L), drop = FALSE])
    class(ret) <- "array"
    dim(ret) <- d[3:1]
    dimnames(ret) <- dn[3:1]
    return(ret)
}

as.array.syMatrices <- function(x, ...)
    return(as.array.ltMatrices(x, symmetric = TRUE))

print.ltMatrices <- function(x, ...)
    print(as.array(x))

print.syMatrices <- function(x, ...)
    print(as.array(x))
@}


\section{Reordering}

It is sometimes convenient to have access to lower triangular matrices in
either column- or row major order and this little helper function switches
between the two forms

@d reorder ltMatrices
@{
.reorder <- function(x, byrow = FALSE) {

    stopifnot(inherits(x, "ltMatrices"))
    if (attr(x, "byrow") == byrow) return(x)

    @<extract slots@>

    class(x) <- class(x)[-1L]

    if (trans) {
        rL <- cL <- diag(0, nrow = J)
        rL[lower.tri(rL, diag = diag)] <- cL[upper.tri(cL, diag = diag)] <- 1:nrow(x)
        cL <- t(cL)
        if (byrow) ### row -> col order
            return(ltMatrices(x[cL[lower.tri(cL, diag = diag)], , drop = FALSE], 
                              diag = diag, byrow = FALSE, trans = TRUE, names = dn[[2L]]))
        ### col -> row order
        return(ltMatrices(x[t(rL)[upper.tri(rL, diag = diag)], , drop = FALSE], 
                          diag = diag, byrow = TRUE, trans = TRUE, names = dn[[2L]]))
    }

    rL <- cL <- diag(0, nrow = J)
    rL[lower.tri(rL, diag = diag)] <- cL[upper.tri(cL, diag = diag)] <- 1:ncol(x)
    cL <- t(cL)
    if (byrow) ### row -> col order
        return(ltMatrices(x[, cL[lower.tri(cL, diag = diag)], drop = FALSE], 
                          diag = diag, byrow = FALSE, trans = FALSE, names = dn[[2L]]))
    ### col -> row order
    return(ltMatrices(x[, t(rL)[upper.tri(rL, diag = diag)], drop = FALSE], 
                      diag = diag, trans = FALSE, byrow = TRUE, names = dn[[2L]]))
}
@}

We can check if this works by switching back and forth between column-major
and row-major order

<<ex-reorder>>=
## constructor + .reorder + as.array
a <- as.array(ltMatrices(xn, byrow = TRUE))
b <- as.array(ltMatrices(ltMatrices(xn, byrow = TRUE), byrow = FALSE))
chk(a, b)

a <- as.array(ltMatrices(xn, byrow = FALSE))
b <- as.array(ltMatrices(ltMatrices(xn, byrow = FALSE), byrow = TRUE))
chk(a, b)

a <- as.array(ltMatrices(xd, byrow = TRUE, diag = TRUE))
b <- as.array(ltMatrices(ltMatrices(xd, byrow = TRUE, diag = TRUE), byrow = FALSE))
chk(a, b)

a <- as.array(ltMatrices(xd, byrow = FALSE, diag = TRUE))
b <- as.array(ltMatrices(ltMatrices(xd, byrow = FALSE, diag = TRUE), byrow = TRUE))
chk(a, b)
@@

The internal representation as $N \times \J (\J + 1) / 2$ matrix to a matrix
of dimensions $\J (\J + 1) / 2 \times N$ can be changed as well (NOTE that
this does not mean the matrix $\mC_i$ is transposed to $\mC_i^\top$!)

@d transpose ltMatrices
@{
.transpose <- function(x, trans = FALSE) {

    stopifnot(inherits(x, "ltMatrices"))
    if (attr(x, "trans") == trans) return(x)

    @<extract slots@>

    class(x) <- class(x)[-1L]

    return(ltMatrices(t(x), diag = diag, byrow = byrow, 
                      trans = !trans, names = dn[[2L]]))
}
@}

which works as advertised

<<ex-trans>>=
## constructor + .reorder + as.array
a <- as.array(ltMatrices(t(xn), trans = TRUE))
b <- as.array(ltMatrices(ltMatrices(t(xn), trans = TRUE), trans = FALSE))
chk(a, b)

a <- as.array(ltMatrices(xn, trans = FALSE))
b <- as.array(ltMatrices(ltMatrices(xn, trans = FALSE), trans = TRUE))
chk(a, b)

a <- as.array(ltMatrices(t(xd), trans = TRUE, diag = TRUE))
b <- as.array(ltMatrices(ltMatrices(t(xd), trans = TRUE, diag = TRUE), trans = FALSE))
chk(a, b)

a <- as.array(ltMatrices(xd, trans = FALSE, diag = TRUE))
b <- as.array(ltMatrices(ltMatrices(xd, trans = FALSE, diag = TRUE), trans = TRUE))
chk(a, b)
@@

\section{Subsetting}

We might want to select subsets of observations $i \in \{1, \dots, N\}$ or
rows/columns $j \in \{1, \dots, \J\}$ of the corresponding matrices $\mC_i$

@d subset ltMatrices
@{
"[.ltMatrices" <- function(x, i, j, ..., drop = FALSE) {

    if (drop) warning("argument drop is ignored")
    if (missing(i) && missing(j)) return(x)

    @<extract slots@>

    class(x) <- class(x)[-1L]

    if (!missing(j)) {
        if (length(j) == 1L && !diag) {
            if (trans)
                return(ltMatrices(matrix(1, ncol = ncol(x), nrow = 1), diag = TRUE, 
                                  trans = TRUE, names = dn[[2L]][j]))
            return(ltMatrices(matrix(1, nrow = nrow(x), ncol = 1), diag = TRUE, 
                              names = dn[[2L]][j]))
        }
        L <- diag(0L, nrow = J)
        Jp <- sum(upper.tri(L, diag = diag))
        if (byrow) {
            L[upper.tri(L, diag = diag)] <- 1:Jp
            L <- L[j, j, drop = FALSE]
            L <- L[upper.tri(L, diag = diag)]
        } else {
            L[lower.tri(L, diag = diag)] <- 1:Jp
            L <- L[j, j, drop = FALSE]
            L <- L[lower.tri(L, diag = diag)]
        }
        if (missing(i)) {
            if (trans)
                return(ltMatrices(x[c(L), , drop = FALSE], diag = diag, 
                                  trans = TRUE, byrow = byrow, names = dn[[2L]][j]))
            return(ltMatrices(x[, c(L), drop = FALSE], diag = diag, 
                              byrow = byrow, names = dn[[2L]][j]))
        }
        if (trans) 
            return(ltMatrices(x[c(L), i, drop = FALSE], diag = diag, 
                              trans = TRUE, byrow = byrow, names = dn[[2L]][j]))
        return(ltMatrices(x[i, c(L), drop = FALSE], diag = diag, 
                          byrow = byrow, names = dn[[2L]][j]))
    }
    if (trans)
        return(ltMatrices(x[, i, drop = FALSE], diag = diag, 
                          trans = trans, byrow = byrow, names = dn[[2L]]))
    return(ltMatrices(x[i, , drop = FALSE], diag = diag, 
                      byrow = byrow, names = dn[[2L]]))
}

"[.syMatrices" <- function(x, i, j, ..., drop = FALSE) {
    class(x)[1L] <- "ltMatrices"
    ret <- x[i, j, ..., drop = drop]
    class(ret)[1L] <- "syMatrices"
    return(ret)
}
@}

We check if this works by first subsetting the \code{ltMatrices} object.
Second, we coerse the object to an array and do the subset for the latter
object. Both results must agree.

<<ex-subset>>=
## subset
a <- as.array(ltMatrices(xn, byrow = FALSE)[1:2, 2:4])
b <- as.array(ltMatrices(xn, byrow = FALSE))[2:4, 2:4, 1:2]
chk(a, b)

a <- as.array(ltMatrices(xn, byrow = TRUE)[1:2, 2:4])
b <- as.array(ltMatrices(xn, byrow = TRUE))[2:4, 2:4, 1:2]
chk(a, b)

a <- as.array(ltMatrices(xd, byrow = FALSE, diag = TRUE)[1:2, 2:4])
b <- as.array(ltMatrices(xd, byrow = FALSE, diag = TRUE))[2:4, 2:4, 1:2]
chk(a, b)

a <- as.array(ltMatrices(xd, byrow = TRUE, diag = TRUE)[1:2, 2:4])
b <- as.array(ltMatrices(xd, byrow = TRUE, diag = TRUE))[2:4, 2:4, 1:2]
chk(a, b)

### with trans
a <- as.array(ltMatrices(t(xn), byrow = FALSE, trans = TRUE)[1:2, 2:4])
b <- as.array(ltMatrices(t(xn), byrow = FALSE, trans = TRUE))[2:4, 2:4, 1:2]
chk(a, b)

a <- as.array(ltMatrices(t(xn), byrow = TRUE, trans = TRUE)[1:2, 2:4])
b <- as.array(ltMatrices(t(xn), byrow = TRUE, trans = TRUE))[2:4, 2:4, 1:2]
chk(a, b)

a <- as.array(ltMatrices(t(xd), byrow = FALSE, diag = TRUE, trans = TRUE)[1:2, 2:4])
b <- as.array(ltMatrices(t(xd), byrow = FALSE, diag = TRUE, trans = TRUE))[2:4, 2:4, 1:2]
chk(a, b)

a <- as.array(ltMatrices(t(xd), byrow = TRUE, diag = TRUE, trans = TRUE)[1:2, 2:4])
b <- as.array(ltMatrices(t(xd), byrow = TRUE, diag = TRUE, trans = TRUE))[2:4, 2:4, 1:2]
chk(a, b)
@@

\section{Diagonal Elements}

The diagonal elements of each matrix $\mC_i$ can be extracted and are
always returned as an $\J \times N$ matrix (regardless of \code{trans}).
The reason is that \code{ltMatrices} with \code{trans = TRUE} can be
standardized elementwise without transposing objects

@d diagonals ltMatrices
@{
diagonals <- function(x, ...)
    UseMethod("diagonals")

diagonals.ltMatrices <- function(x, ...) {

    @<extract slots@>

    class(x) <- class(x)[-1L]

    if (!diag) {
        ret <- matrix(1,  nrow = J, ncol = nrow(x))
        colnames(ret) <- dn[[1L]]
        rownames(ret) <- dn[[2L]]
        return(ret)
    } else {
        if (J == 1L) {
            if (trans) return(x)
            return(t(x))
        }
        if (byrow)
            idx <- cumsum(c(1, 2:J))
        else
            idx <- cumsum(c(1, J:2))
        if (trans)
            ret <- x[idx, , drop = FALSE]
        else
            ret <- t(x[, idx, drop = FALSE])
        rownames(ret) <- dn[[2L]]
        return(ret)
    }
}

diagonals.syMatrices <- diagonals.ltMatrices

diagonals.matrix <- function(x, ...) diag(x)

@}

<<ex-diag>>=
all(diagonals(ltMatrices(xn, byrow = TRUE)) == 1L)
@@

\section{Multiplication}

Multiplications $\mC_i \yvec_i$ with $\yvec_i \in \R^\J$ for $i = 1, \dots,
N$ can be computed with $\code{y}$ being an $J \times N$ matrix of
columns-wise stacked vectors $(\yvec_1 \mid \yvec_2 \mid \dots \mid
\yvec_N)$. If \code{y} is a single vector, it is recycled $N$ times.

@d mult ltMatrices
@{
### C %*% y
Mult <- function(x, y) {

    if (!inherits(x, "ltMatrices")) return(x %*% y)

    @<extract slots@>

    if (!is.matrix(y)) y <- matrix(y, nrow = d[2L], ncol = d[1L])
    N <- ifelse(d[1L] == 1, ncol(y), d[1L])
    stopifnot(nrow(y) == d[2L] && ncol(y) == N)

    x <- ltMatrices(x, byrow = TRUE, trans = TRUE)

    class(x) <- class(x)[-1L]
    storage.mode(x) <- "double"
    storage.mode(y) <- "double"

    ret <- .Call(mvtnorm_R_ltMatrices_Mult, x, y, as.integer(N), 
                 as.integer(d[2L]), as.logical(diag))
    
    rownames(ret) <- dn[[2L]]
    if (length(dn[[1L]]) == N)
        colnames(ret) <- dn[[1L]]
    return(ret)
}
@}

The underlying \proglang{C} code assumes $\mC_i$ (here called \code{C}) to
be in row-major order and in transposed form.

@d RC input
@{
/* pointer to C matrices */
double *dC = REAL(C);
/* number of matrices */
int iN = INTEGER(N)[0];
/* dimension of matrices */
int iJ = INTEGER(J)[0];
/* C contains diagonal elements */
Rboolean Rdiag = asLogical(diag);
/* p = J * (J - 1) / 2 + diag * J */
int len = iJ * (iJ - 1) / 2 + Rdiag * iJ;
@}

We also allow $\mC_i$ to be constant ($N$ is then determined from
\code{ncol(y)}). The following fragment ensures that we only loop over
$\mC_i$ if \code{dim(x)[1L] > 1}

@d C length
@{
int p;
if (LENGTH(C) == len)
    /* C is constant for i = 1, ..., N */
    p = 0;
else 
    /* C contains C_1, ...., C_N */
    p = len;
@}

The \proglang{C} workhorse is now

@d mult
@{
SEXP R_ltMatrices_Mult (SEXP C, SEXP y, SEXP N, SEXP J, SEXP diag) {

    SEXP ans;
    double *dans, *dy = REAL(y);
    int i, j, k, start;

    @<RC input@>

    @<C length@>

    PROTECT(ans = allocMatrix(REALSXP, iJ, iN));
    dans = REAL(ans);
    
    for (i = 0; i < iN; i++) {
        start = 0;
        for (j = 0; j < iJ; j++) {
            dans[j] = 0.0;
            for (k = 0; k < j; k++)
                dans[j] += dC[start + k] * dy[k];
            if (Rdiag) {
                dans[j] += dC[start + j] * dy[j];
                start += j + 1;
            } else {
                dans[j] += dy[j]; 
                start += j;
            }
        }
        dC += p;
        dy += iJ;
        dans += iJ;
    }
    UNPROTECT(1);
    return(ans);
}
@}

Some checks

<<ex-mult>>=
lxn <- ltMatrices(xn, byrow = TRUE)
lxd <- ltMatrices(xd, byrow = TRUE, diag = TRUE)
y <- matrix(runif(N * J), nrow = J)
a <- Mult(lxn, y)
A <- as.array(lxn)
b <- do.call("rbind", lapply(1:ncol(y), function(i) t(A[,,i] %*% y[,i,drop = FALSE])))
chk(a, t(b), check.attributes = FALSE)

a <- Mult(lxd, y)
A <- as.array(lxd)
b <- do.call("rbind", lapply(1:ncol(y), function(i) t(A[,,i] %*% y[,i,drop = FALSE])))
chk(a, t(b), check.attributes = FALSE)

### recycle C
chk(Mult(lxn[rep(1, N),], y), Mult(lxn[1,], y), check.attributes = FALSE)

### recycle y
chk(Mult(lxn, y[,1]), Mult(lxn, y[,rep(1, N)]))

### tcrossprod as multiplication
i <- sample(1:N)[1]
M <- t(as.array(lxn)[,,i])
a <- sapply(1:J, function(j) Mult(lxn[i,], M[,j,drop = FALSE]))
rownames(a) <- colnames(a) <- dimnames(lxn)[[2L]]
b <- as.array(Tcrossprod(lxn[i,]))[,,1]
chk(a, b, check.attributes = FALSE)
@@

\section{Solving linear systems}

Compute $\mC_i^{-1}$ or solve $\mC_i \xvec_i = \yvec_i$ for $\xvec_i$ for
all $i = 1, \dots, N$.


\code{C} is $\mC_i, i = 1, \dots, N$ in transposed column-major order
(matrix of dimension $\J (\J - 1) / 2 + \J \text{diag} \times N$), and
\code{y} is the $\J \times N$ matrix $(\yvec_1 \mid \yvec_2 \mid \dots \mid
\yvec_N)$. This function returns the $\J \times N$ matrix $(\xvec_1 \mid \xvec_2 \mid \dots \mid
\xvec_N)$ of solutions.

If \code{y} is not given, $\mC_i^{-1}$ is returned in transposed
column-major order (matrix of dimension $\J (\J \pm 1) / 2  \times N$). If
all $\mC_i$ have unit diagonals, so will $\mC_i^{-1}$.

@d setup memory
@{
/* return object: include unit diagnonal elements if Rdiag == 0 */

/* add diagonal elements (expected by Lapack) */
nrow = (Rdiag ? len : len + iJ);
ncol = (p > 0 ? iN : 1);
PROTECT(ans = allocMatrix(REALSXP, nrow, ncol));
dans = REAL(ans);

ansx = ans;
dansx = dans;
dy = dans;
if (y != R_NilValue) {
    dy = REAL(y);
    PROTECT(ansx = allocMatrix(REALSXP, iJ, iN));
    dansx = REAL(ansx);
}
@}

The \proglang{LAPACK} functions \code{dtptri} and \code{dtpsv} assume that
diagonal elements are present, even for unit diagonal matrices.

@d copy elements
@{
/* copy data and insert unit diagonal elements when necessary */
if (p > 0 || i == 0) {
    jj = 0;
    k = 0;
    idx = 0;
    j = 0;
    while(j < len) {
        if (!Rdiag && (jj == idx)) {
            dans[jj] = 1.0;
            idx = idx + (iJ - k);
            k++;
        } else {
            dans[jj] = dC[j];
            j++;
        }
        jj++;
    }
    if (!Rdiag) dans[idx] = 1.0;
}

if (y != R_NilValue) {
    for (j = 0; j < iJ; j++)
        dansx[j] = dy[j];
}
@}

The \proglang{LAPACK} workhorses are called here

@d call Lapack
@{
if (y == R_NilValue) {
    /* compute inverse */
    F77_CALL(dtptri)(&lo, &di, &iJ, dans, &info FCONE FCONE);
    if (info != 0)
        error("Cannot solve ltmatices");
} else {
    /* solve linear system */
    F77_CALL(dtpsv)(&lo, &tr, &di, &iJ, dans, dansx, &ONE FCONE FCONE FCONE);
    dansx += iJ;
    dy += iJ;
}
@}

@d return objects
@{
if (y == R_NilValue) {
    UNPROTECT(1);
    /* note: ans always includes diagonal elements */
    return(ans);
} else {
    UNPROTECT(2);
    return(ansx);
}
@}

We finally put everything together in a dedicated \proglang{C} function

@d solve
@{
SEXP R_ltMatrices_solve (SEXP C, SEXP y, SEXP N, SEXP J, SEXP diag)
{

    SEXP ans, ansx;
    double *dans, *dansx, *dy;
    int i, j, k, info, nrow, ncol, jj, idx, ONE = 1;

    @<RC input@>

    @<C length@>

    char di, lo = 'L', tr = 'N';
    if (Rdiag) {
        /* non-unit diagonal elements */
        di = 'N';
    } else {
        /* unit diagonal elements */
        di = 'U';
    }

    @<setup memory@>
    
    /* loop over matrices, ie columns of C  / y */    
    for (i = 0; i < iN; i++) {

        @<copy elements@>

        @<call Lapack@>

        /* next matrix */
        if (p > 0) {
            dans += nrow;
            dC += p;
        }
    }

    @<return objects@>
}
@}

with \proglang{R} interface

@d solve ltMatrices
@{
solve.ltMatrices <- function(a, b, ...) {

    byrow_orig <- attr(a, "byrow")
    trans_orig <- attr(a, "trans")

    x <- ltMatrices(a, byrow = FALSE, trans = TRUE)
    diag <- attr(x, "diag")
    d <- dim(x)
    J <- d[2L]
    dn <- dimnames(x)
    class(x) <- class(x)[-1L]
    storage.mode(x) <- "double"

    if (!missing(b)) {
        if (!is.matrix(b)) b <- matrix(b, nrow = J, ncol = ncol(x))
        stopifnot(nrow(b) == J)
        N <- ifelse(d[1L] == 1, ncol(b), d[1L])
        stopifnot(ncol(b) == N)
        storage.mode(b) <- "double"
        ret <- .Call(mvtnorm_R_ltMatrices_solve, x, b, 
                     as.integer(N), as.integer(J), as.logical(diag))
        if (d[1L] == N) {
            colnames(ret) <- dn[[1L]]
        } else {
            colnames(ret) <- colnames(b)
        }
        rownames(ret) <- dn[[2L]]
        return(ret)
    }

    ret <- try(.Call(mvtnorm_R_ltMatrices_solve, x, NULL,
                     as.integer(ncol(x)), as.integer(J), as.logical(diag)))
    colnames(ret) <- dn[[1L]]

    if (!diag)
        ### ret always includes diagonal elements, remove here
        ret <- ret[- cumsum(c(1, J:2)), , drop = FALSE]

    ret <- ltMatrices(ret, diag = diag, byrow = FALSE, trans = TRUE, 
                      names = dn[[2L]])
    ret <- ltMatrices(ret, byrow = byrow_orig, trans = trans_orig)
    return(ret)
}
@}

and some checks

<<ex-solve>>=
## solve
A <- as.array(lxn)
a <- solve(lxn)
a <- as.array(a)
b <- array(apply(A, 3L, function(x) solve(x), simplify = TRUE), dim = rev(dim(lxn)))
chk(a, b, check.attributes = FALSE)

A <- as.array(lxd)
a <- as.array(solve(lxd))
b <- array(apply(A, 3L, function(x) solve(x), simplify = TRUE), dim = rev(dim(lxd)))
chk(a, b, check.attributes = FALSE)

chk(solve(lxn, y), Mult(solve(lxn), y))
chk(solve(lxd, y), Mult(solve(lxd), y))

### recycle C
chk(solve(lxn[1,], y), as.array(solve(lxn[1,]))[,,1] %*% y)
chk(solve(lxn[rep(1, N),], y), solve(lxn[1,], y), check.attributes = FALSE)

### recycle y
chk(solve(lxn, y[,1]), solve(lxn, y[,rep(1, N)]))
@@

\section{Crossproducts}

Compute $\mC_i \mC_i^\top$ or $\text{diag}(\mC_i \mC_i^\top)$
(\code{diag\_only = TRUE}) for $i = 1, \dots, N$. These are symmetric
matrices, so we store them as a lower triangular matrix using a different
class name \code{syMatrices}.

We differentiate between computation of the diagonal elements of the
crossproduct

@d tcrossprod diagonal only
@{
PROTECT(ans = allocMatrix(REALSXP, iJ, iN));
dans = REAL(ans);
for (n = 0; n < iN; n++) {
    dans[0] = 1.0;
    if (Rdiag) dans[0] = pow(dC[0], 2);
    for (i = 1; i < iJ; i++) {
        dans[i] = 0.0;
        for (k = 0; k < i; k++)
            dans[i] += pow(dC[IDX(i + 1, k + 1, iJ, Rdiag)], 2);
        if (Rdiag) {
            dans[i] += pow(dC[IDX(i + 1, i + 1, iJ, Rdiag)], 2);
        } else {
            dans[i] += 1.0;
        }
    }
    dans += iJ;
    dC += len;
}
@}

and computation of the full $\J \times \J$ crossproduct matrix

@d tcrossprod full
@{
nrow = iJ * (iJ + 1) / 2;
PROTECT(ans = allocMatrix(REALSXP, nrow, iN)); 
dans = REAL(ans);
for (n = 0; n < INTEGER(N)[0]; n++) {
    dans[0] = 1.0;
    if (Rdiag) dans[0] = pow(dC[0], 2);
    for (i = 1; i < iJ; i++) {
        for (j = 0; j <= i; j++) {
            ix = IDX(i + 1, j + 1, iJ, 1);
            dans[ix] = 0.0;
            for (k = 0; k < j; k++)
                dans[ix] += 
                    dC[IDX(i + 1, k + 1, iJ, Rdiag)] *
                    dC[IDX(j + 1, k + 1, iJ, Rdiag)];
            if (Rdiag) {
                dans[ix] += 
                    dC[IDX(i + 1, j + 1, iJ, Rdiag)] *
                    dC[IDX(j + 1, j + 1, iJ, Rdiag)];
            } else {
                if (j < i)
                    dans[ix] += dC[IDX(i + 1, j + 1, iJ, Rdiag)];
                else
                    dans[ix] += 1.0;
            }
        }
    }
    dans += nrow;
    dC += len;
}
@}

and put both cases together

@d tcrossprod
@{
#define IDX(i, j, n, d) ((i) >= (j) ? (n) * ((j) - 1) - ((j) - 2) * ((j) - 1)/2 + (i) - (j) - (!d) * (j) : 0)

SEXP R_ltMatrices_tcrossprod (SEXP C, SEXP N, SEXP J, SEXP diag, SEXP diag_only) {

    SEXP ans;
    double *dans;
    int i, j, n, k, ix, nrow;

    @<RC input@>

    Rboolean Rdiag_only = asLogical(diag_only);

    if (Rdiag_only) {

        @<tcrossprod diagonal only@>

    } else {

        @<tcrossprod full@>

    }
    UNPROTECT(1);
    return(ans);
}
@}

with \proglang{R} interface

@d tcrossprod ltMatrices
@{
### C %*% t(C) => returns object of class syMatrices
### diag(C %*% t(C)) => returns matrix of diagonal elements
Tcrossprod <- function(x, diag_only = FALSE) {

    if (!inherits(x, "ltMatrices")) {
        ret <- tcrossprod(x)
        if (diag_only) ret <- diag(ret)
        return(ret)
    }

    byrow_orig <- attr(x, "byrow")
    trans_orig <- attr(x, "trans")
    diag <- attr(x, "diag")
    d <- dim(x)
    J <- d[2L]
    dn <- dimnames(x)

    x <- ltMatrices(x, byrow = FALSE, trans = TRUE)
    class(x) <- class(x)[-1L]
    N <- d[1L]
    storage.mode(x) <- "double"

    ret <- .Call(mvtnorm_R_ltMatrices_tcrossprod, x, as.integer(N), as.integer(J), 
                 as.logical(diag), as.logical(diag_only))
    colnames(ret) <- dn[[1L]]
    if (diag_only) {
        rownames(ret) <- dn[[2L]]
    } else {
        ret <- ltMatrices(ret, diag = TRUE, byrow = FALSE, trans = TRUE, names = dn[[2L]])
        ret <- ltMatrices(ret, byrow = byrow_orig, trans = trans_orig)
        class(ret)[1L] <- "syMatrices"
    }
    return(ret)
}
@}

We could have created yet another generic \code{tcrossprod}, but
\code{base::tcrossprod} is more general and, because speed is an issue, we
don't want to waste time on methods dispatch.

<<ex-tcrossprod>>=
## Tcrossprod
a <- as.array(Tcrossprod(lxn))
b <- array(apply(as.array(lxn), 3L, function(x) tcrossprod(x), simplify = TRUE), 
           dim = rev(dim(lxn)))
chk(a, b, check.attributes = FALSE)

# diagonal elements only
d <- Tcrossprod(lxn, diag_only = TRUE)
chk(d, apply(a, 3, diag))
chk(d, diagonals(Tcrossprod(lxn)))

a <- as.array(Tcrossprod(lxd))
b <- array(apply(as.array(lxd), 3L, function(x) tcrossprod(x), simplify = TRUE), 
           dim = rev(dim(lxd)))
chk(a, b, check.attributes = FALSE)

# diagonal elements only
d <- Tcrossprod(lxd, diag_only = TRUE)
chk(d, apply(a, 3, diag))
chk(d, diagonals(Tcrossprod(lxd)))
@@

\section{Application Example}

Let's say we have $\rY_i \sim \N_\J(\mathbf{0}_J, \mC_i \mC_i^{\top})$
for $i = 1, \dots, N$ and we know the Cholesky factors $\mL_i = \mC_i^{-1}$ of the $N$
precision matrices $\Sigma^{-1} = \mL_i \mL_i^{\top}$. We generate $\rY_i = \mL_i^{-1}
\rZ_i$ from $\rZ_i \sim \N_\J(\mathbf{0}_\J, \mI_\J)$.
Evaluating the corresponding log-likelihood is now straightforward and fast,
compared to repeated calls to \code{dmvnorm}

<<ex-MV>>=
N <- 1000
J <- 50
lt <- ltMatrices(matrix(runif(N * J * (J + 1) / 2) + 1, ncol = N), 
                 diag = TRUE, byrow = FALSE, trans = TRUE)
Z <- matrix(rnorm(N * J), ncol = N)
Y <- solve(lt, Z)
system.time(ll1 <- sum(dnorm(Mult(lt, Y), log = TRUE)) + sum(log(diagonals(lt))))

system.time(S <- as.array(Tcrossprod(solve(lt))))
system.time(ll2 <- sum(sapply(1:N, function(i) dmvnorm(x = Y[,i], sigma = S[,,i], log = TRUE))))
chk(ll1, ll2)
@@


\chapter{Multivariate Normal Log-likelihoods} \label{lmvnorm}

We now discuss code for evaluating the log-likelihood
\begin{eqnarray*}
\sum_{i = 1}^N \log(p_i(\mC_i \mid \avec_i, \bvec_i))
\end{eqnarray*}

This is relatively simple to achieve using the existing \code{pmvnorm}, so a
prototype might look like

<<lmvnorm_R>>=
library("mvtnorm")
lmvnormR <- function(lower, upper, mean = 0, chol, logLik = TRUE, ...) {

    @<input checks@>

    sigma <- Tcrossprod(chol)
    S <- as.array(sigma)
    idx <- 1

    ret <- error <- numeric(N)
    for (i in 1:N) {
        if (dim(sigma)[[1L]] > 1) idx <- i
        tmp <- pmvnorm(lower = lower[,i], upper = upper[,i], sigma = S[,,idx], ...)
        ret[i] <- tmp
        error[i] <- attr(tmp, "error")
    }
    attr(ret, "error") <- error

    if (logLik)
        return(sum(log(pmax(ret, .Machine$double.eps))))

    ret
}
@@

However, the underlying \code{FORTRAN} code first computes the Cholesky
factor based on the covariance matrix, which is clearly a waste of time.
Repeated calls to \code{FORTRAN} also cost some time. The code \citep[based
on and evaluated in][]{Genz_Bretz_2002} implements a
specific form of quasi-Monte-Carlo integration without allowing the user to
change the scheme (or to fall-back to simple Monte-Carlo). We therefore
implement our own, and simplistic version, with the aim to speed-things up
such that maximum-likelihood estimation becomes a bit faster.

Let's look at an example first. This code estimates $p_1, \dots, p_5$
<<ex-lmvnorm_R>>=
J <- 5
N <- 10

x <- matrix(runif(N * J * (J + 1) / 2), ncol = N)
lx <- ltMatrices(x, byrow = TRUE, trans = TRUE, diag = TRUE)

a <- matrix(runif(N * J), nrow = J) - 2
a[sample(J * N)[1:2]] <- -Inf
b <- a + 2 + matrix(runif(N * J), nrow = J)
b[sample(J * N)[1:2]] <- Inf

(phat <- c(lmvnormR(a, b, chol = lx, logLik = FALSE)))
@@

We want to achieve the same result a bit more general and a bit faster.

\section{Algorithm}

@o lmvnorm.R -cp
@{
@<R Header@>
@<lmvnorm@>
@<smvnorm@>
@}

@o lmvnorm.c -cc
@{
@<C Header@>
#include <R.h>
#include <Rmath.h>
#include <Rinternals.h>
#include <Rdefines.h>
#include <Rconfig.h>
@<pnorm fast@>
@<R lmvnorm@>
@<R smvnorm@>
@}

We implement the algorithm described by \cite{numerical-:1992}. The key
point here is that the original $\J$-dimensional problem~(\ref{pmvnorm}) is transformed into
an integral over $[0, 1]^{\J - 1}$.

For each $i = 1, \dots, N$, do

\begin{enumerate}
  \item Input $\mC_i$ (\code{chol}), $\avec_i$ (\code{lower}), $\bvec_i$ \code{upper}, and control parameters $\alpha$, $\epsilon$, and $M_\text{max}$ (\code{M}).

@d input checks
@{
stopifnot(isTRUE(all.equal(dim(lower), dim(upper))))

stopifnot(inherits(chol, "ltMatrices"))
chol <- ltMatrices(chol, trans = TRUE, byrow = TRUE)
d <- dim(chol)
### allow single matrix C
N <- ifelse(d[1L] == 1, ncol(lower), d[1L])
J <- d[2L]

stopifnot(nrow(lower) == J && ncol(lower) == N)
stopifnot(nrow(upper) == J && ncol(upper) == N)
if (is.matrix(mean))
    stopifnot(nrow(mean) == J && ncol(mean) == N)

lower <- lower - mean
upper <- upper - mean
@}


  \item Standardize integration limits $a^{(i)}_j / c^{(i)}_{jj}$, $b^{(i)}_j / c^{(i)}_{jj}$, and rows $c^{(i)}_{j\jmath} / c^{(i)}_{jj}$ for $1 \le \jmath < j < \J$.


@d standardise
@{
if (attr(chol, "diag")) {
    ### diagonals returns J x N and lower/upper are J x N, so
    ### elementwise standardisation is simple
    dchol <- diagonals(chol)
    ### zero diagonals not allowed
    stopifnot(all(abs(dchol) > sqrt(.Machine$double.eps)))
    ac <- lower / c(dchol)
    bc <- upper / c(dchol)
    ### CHECK if dimensions are correct
    C <- unclass(chol) / c(dchol[rep(1:J, 1:J),])
    if (J > 1) ### else: univariate problem; C is no longer used
        C <- ltMatrices(C[-cumsum(c(1, 2:J)), ], 
                        byrow = TRUE, trans = TRUE, diag = FALSE)
} else {
    ac <- lower
    bc <- upper
    C <- ltMatrices(chol, byrow = TRUE, trans = TRUE)
}
uC <- unclass(C)
@}


  \item Initialize $\text{intsum} = \text{varsum} = 0$, $M = 0$, $d_1 =
\Phi\left(a^{(i)}_1\right)$, $e_1 = \Phi\left(b^{(i)}_1\right)$ and $f_1 = e_1 - d_1$.


@d initialisation
@{
d0 = C_pnorm_fast(da[0], 0.0);
e0 = C_pnorm_fast(db[0], 0.0);
emd0 = e0 - d0;
f0 = emd0;
intsum = (iJ > 1 ? 0.0 : f0);
@}

  \item Repeat

    \begin{enumerate}

      \item Generate uniform $w_1, \dots, w_{\J - 1} \in [0, 1]$.

      \item For $j = 2, \dots, J$ set 
        \begin{eqnarray*}
            y_{j - 1} & = & \Phi^{-1}\left(d_{j - 1} + w_{j - 1} (e_{j - 1} - d_{j - 1})\right)
        \end{eqnarray*}

We either generate $w_{j - 1}$ on the fly or use pre-computed weights
(\code{w}).

@d compute y
@{
Wtmp = (W == R_NilValue ? unif_rand() : dW[j - 1]);
tmp = d + Wtmp * emd;
if (tmp < dtol) {
    y[j - 1] = q0;
} else {
    if (tmp > mdtol)
        y[j - 1] = -q0;
    else
        y[j - 1] = qnorm(tmp, 0.0, 1.0, 1L, 0L);
}
@}

        \begin{eqnarray*}
            x_{j - 1} & = & \sum_{\jmath = 1}^{j - 1} c^{(i)}_{j\jmath} y_j
\end{eqnarray*}

@d compute x
@{
x = 0.0;
for (k = 0; k < j; k++)
    x += dC[start + k] * y[k];
@}

        \begin{eqnarray*}
            d_j & = & \Phi\left(a^{(i)}_j - x_{j - 1}\right) \\
            e_j & = & \Phi\left(b^{(i)}_j - x_{j - 1}\right)
        \end{eqnarray*}

@d update d, e
@{
d = C_pnorm_fast(da[j], x);
e = C_pnorm_fast(db[j], x);
emd = e - d;
@}

        \begin{eqnarray*}
            f_j & = & (e_j - d_j) f_{j - 1}.
       \end{eqnarray*}

@d update f
@{
start += j;
f *= emd;
@}

We put everything together in a loop starting with the second dimension

@d inner loop
@{
for (j = 1; j < iJ; j++) {

    @<compute y@>

    @<compute x@>

    @<update d, e@>

    @<update f@>
}
@}


      \item Set $\text{intsum} = \text{intsum} + f_\J$, $\text{varsum} = \text{varsum} + f^2_\J$, $M = M + 1$, 
            and $\text{error} = \sqrt{(\text{varsum}/M - (\text{intsum}/M)^2) / M}$.

@d increment
@{
intsum += f;
@}

We refrain from early stopping and error estimation. 
    
      \item[Until] $\text{error} < \epsilon$ or $M = M_\text{max}$

    \end{enumerate}
  \item Output $\hat{p}_i = \text{intsum} / M$.

We return $\log{\hat{p}_i}$ for each $i$, or we immediately sum-up over $i$.

@d output
@{
dans[0] += (intsum < dtol ? l0 : log(intsum)) - lM;
if (!RlogLik)
    dans += 1L;
@}

\end{enumerate}

It turned out that calls to \code{pnorm} are expensive, so a slightly faster
alternative \citep[suggested by][]{Matic_Radoicic_Stefanica_2018} is used

@d pnorm fast
@{
/* see https://doi.org/10.2139/ssrn.2842681  */
const double g2 =  -0.0150234471495426236132;
const double g4 = 0.000666098511701018747289;
const double g6 = 5.07937324518981103694e-06;
const double g8 = -2.92345273673194627762e-06;
const double g10 = 1.34797733516989204361e-07;
const double m2dpi = -2.0 / M_PI; //3.141592653589793115998;

double C_pnorm_fast (double x, double m) {

    double tmp, ret;
    double x2, x4, x6, x8, x10;

    if (R_FINITE(x)) {
        x = x - m;
        x2 = x * x;
        x4 = x2 * x2;
        x6 = x4 * x2;
        x8 = x6 * x2;
        x10 = x8 * x2;
        tmp = 1 + g2 * x2 + g4 * x4 + g6 * x6  + g8 * x8 + g10 * x10;
        tmp = m2dpi * x2 * tmp;
        ret = .5 + ((x > 0) - (x < 0)) * sqrt(1 - exp(tmp)) / 2.0;
    } else {
        ret = (x > 0 ? 1.0 : 0.0);
    }
    return(ret);
}
@}

We allow a new set of weights for each observation or one set for all
observations. In the former case, the number of columns is $M \times N$ and
in the latter just $M$.

@d W length
@{
int pW = 0;
if (W != R_NilValue) {
    if (LENGTH(W) == (iJ - 1) * iM) {
        pW = 0;
    } else {
        if (LENGTH(W) != (iJ - 1) * iN * iM)
            error("Length of W incorrect");
        pW = 1;
    }
    dW = REAL(W);
}
@}

@d dimensions
@{
int iM = INTEGER(M)[0]; 
int iN = INTEGER(N)[0]; 
int iJ = INTEGER(J)[0]; 

da = REAL(a);
db = REAL(b);
dC = REAL(C);
dW = REAL(C); // make -Wmaybe-uninitialized happy

if (LENGTH(C) == iJ * (iJ - 1) / 2)
    p = 0;
else 
    p = LENGTH(C) / iN;
@}

@d setup return object
@{
len = (RlogLik ? 1 : iN);
PROTECT(ans = allocVector(REALSXP, len));
dans = REAL(ans);
for (int i = 0; i < len; i++)
    dans[i] = 0.0;
@}

The case $\J = 1$ does not loop over $M$

@d univariate problem
@{
if (iJ == 1) {
    iM = 0; 
    lM = 0.0;
} else {
    lM = log((double) iM);
}
@}

We put the code together in a dedicated \proglang{C} function

@d R lmvnorm
@{
SEXP R_lmvnorm(SEXP a, SEXP b, SEXP C, SEXP N, SEXP J, SEXP W, SEXP M, SEXP tol, SEXP logLik) {

    SEXP ans;
    double *da, *db, *dC, *dW, *dans, dtol = REAL(tol)[0];
    double mdtol = 1.0 - dtol;
    double d0, e0, emd0, f0, q0, l0, lM, intsum;
    int p, len;

    Rboolean RlogLik = asLogical(logLik);

    @<dimensions@>

    @<W length@>

    int start, j, k;
    double tmp, Wtmp, e, d, f, emd, x, y[iJ - 1];

    @<setup return object@>

    q0 = qnorm(dtol, 0.0, 1.0, 1L, 0L);
    l0 = log(dtol);

    @<univariate problem@>

    if (W == R_NilValue)
        GetRNGstate();

    for (int i = 0; i < iN; i++) {

        @<initialisation@>

        if (W != R_NilValue && pW == 0)
            dW = REAL(W);

        for (int m = 0; m < iM; m++) {

            d = d0;
            f = f0;
            emd = emd0;
            start = 0;

            @<inner loop@>

            @<increment@>

            if (W != R_NilValue)
                dW += iJ - 1;
        }

        da += iJ;
        db += iJ;

        @<output@>

        /* constant C? p == 0*/
        if (p > 0)
            dC += p;
    }

    if (W == R_NilValue)
        PutRNGstate();

    UNPROTECT(1);
    return(ans);
}
@}

The \proglang{R} user interface consists of some checks and a call to
\proglang{C}. Note that we need to specify both \code{w} and \code{M} in
case we want a new set of weights for each observation.

@d init random seed, reset on exit
@{
### from stats:::simulate.lm
if (!exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) 
    runif(1)
if (is.null(seed)) 
    RNGstate <- get(".Random.seed", envir = .GlobalEnv)
else {
    R.seed <- get(".Random.seed", envir = .GlobalEnv)
    set.seed(seed)
    RNGstate <- structure(seed, kind = as.list(RNGkind()))
    on.exit(assign(".Random.seed", R.seed, envir = .GlobalEnv))
}
@}

@d check and / or set integration weights
@{
if (!is.null(w)) {
    stopifnot(is.matrix(w))
    stopifnot(nrow(w) == J - 1)
    if (is.null(M))
        M <- ncol(w)
    stopifnot(ncol(w) %in% c(M, M * N))
    storage.mode(w) <- "double"
} else {
    if (J > 1) {
        if (is.null(M)) stop("either w or M must be specified")
    } else {
        M <- 1L
    }
}
@}

@d lmvnorm
@{
lmvnorm <- function(lower, upper, mean = 0, chol, logLik = TRUE, M = NULL, 
                    w = NULL, seed = NULL, tol = .Machine$double.eps) {

    @<init random seed, reset on exit@>

    @<input checks@>

    @<standardise@>

    @<check and / or set integration weights@>

    ret <- .Call(mvtnorm_R_lmvnorm, ac, bc, unclass(C), as.integer(N), 
                 as.integer(J), w, as.integer(M), as.double(tol), 
                 as.logical(logLik));
    return(ret)
}
@}

@d post differentiate
@{
if (attr(chol, "diag")) {
    idx <- cumsum(c(1, 2:J))
    ret <- ret / c(dchol[rep(1:J, 1:J),]) ### because 1 / dchol already there
    ret[idx,] <- -ret[idx,]
}
@}

@d smvnorm
@{
smvnorm <- function(lower, upper, mean = 0, chol, logLik = TRUE, M = NULL, 
                    w = NULL, seed = NULL, tol = sqrt(.Machine$double.eps)) {


    @<init random seed, reset on exit@>

    @<input checks@>

    @<standardise@>

    @<check and / or set integration weights@>

    ret <- .Call(mvtnorm_R_smvnorm, ac, bc, unclass(C), as.integer(N), 
                 as.integer(J), w, as.integer(M), as.double(tol));

    ll <- log(pmax(ret[1L,], tol)) - log(M)
    m <- matrix(ret[1L,], nrow = nrow(ret) - 1, ncol = ncol(ret), byrow = TRUE)
    ret <- ret[-1L,,drop = FALSE] / m

    @<post differentiate@>

    if (logLik) {
        ret <- list(logLik = ll, score = ret)
        return(ret)
    }
    
    return(ret)
}
@}

@d update yprime
@{
ytmp = 1.0 / dnorm(y[j - 1], 0.0, 1.0, 0L);

for (k = 0; k < Jp; k++) yprime[k * (iJ - 1) + (j - 1)] = 0.0;

for (idx = 0; idx < (j + 1) * j / 2; idx++) {
    yprime[idx * (iJ - 1) + (j - 1)] = ytmp;
    yprime[idx * (iJ - 1) + (j - 1)] *= (dprime[idx] + Wtmp * (eprime[idx] - dprime[idx]));
}
@}

@d score wrt new off-diagonals
@{
dtmp = dnorm(da[j], x, 1.0, 0L);
etmp = dnorm(db[j], x, 1.0, 0L);

for (k = 0; k < j; k++) {
    idx = start + j + k;
    dprime[idx] = dtmp * (-1) * y[k];
    eprime[idx] = etmp * (-1) * y[k];
    fprime[idx] = (eprime[idx] - dprime[idx]) * f;
}
@}

@d score wrt new diagonal
@{
idx = (j + 1) * (j + 2) / 2 - 1;
dprime[idx] = dtmp * (da[j] - x);
eprime[idx] = etmp * (db[j] - x);
fprime[idx] = (eprime[idx] - dprime[idx]) * f;
@}

@d update score
@{
for (idx = 0; idx < j * (j + 1) / 2; idx++) {
    xx = 0.0;
    for (k = 0; k < j; k++)
        xx += dC[start + k] * yprime[idx * (iJ - 1) + k];

    dprime[idx] = dtmp * (-1) * xx;
    eprime[idx] = etmp * (-1) * xx;
    fprime[idx] = (eprime[idx] - dprime[idx]) * f + emd * fprime[idx];
}
@}

@d score inner loop
@{
for (j = 1; j < iJ; j++) {

    @<compute y@>

    @<compute x@>

    @<update d, e@>

    @<update yprime@>

    @<score wrt new off-diagonals@>

    @<score wrt new diagonal@>

    @<update score@>

    @<update f@>

}
@}

@d score output
@{
dans[0] += f;
for (j = 0; j < Jp; j++)
    dans[j + 1] += fprime[j];
@}

@d score output object
@{
int Jp = iJ * (iJ + 1) / 2;
double dprime[Jp], eprime[Jp], fprime[Jp], yprime[(iJ - 1) * Jp];
double dtmp, etmp, Wtmp, ytmp, ktmp, xx;

PROTECT(ans = allocMatrix(REALSXP, Jp + 1, iN));
dans = REAL(ans);
for (j = 0; j < LENGTH(ans); j++) dans[j] = 0.0;
@}

@d R smvnorm
@{
SEXP R_smvnorm(SEXP a, SEXP b, SEXP C, SEXP N, SEXP J, SEXP W, SEXP M, SEXP tol) {

    SEXP ans;
    double *da, *db, *dC, *dW, *dans, dtol = REAL(tol)[0];
    double mdtol = 1.0 - dtol;
    double d0, e0, emd0, f0, q0, l0, intsum, lM;
    int p, len, idx;

    @<dimensions@>

    @<W length@>

    int start, j, k;
    double tmp, e, d, f, emd, x, y[iJ - 1];

    @<score output object@>

    q0 = qnorm(dtol, 0.0, 1.0, 1L, 0L);

    @<univariate problem@>

    if (W == R_NilValue)
        GetRNGstate();

    for (int i = 0; i < iN; i++) {

        @<initialisation@>

        dans[0] = intsum;

        dprime[0] = dnorm(da[0], 0.0, 1.0, 0L) * da[0];
        eprime[0] = dnorm(db[0], 0.0, 1.0, 0L) * db[0];
        fprime[0] = eprime[0] - dprime[0];

        if (W != R_NilValue && pW == 0)
            dW = REAL(W);

        for (int m = 0; m < iM; m++) {

            d = d0;
            f = f0;
            emd = emd0;
            start = 0;

            dprime[0] = dnorm(da[0], 0.0, 1.0, 0L) * da[0];
            eprime[0] = dnorm(db[0], 0.0, 1.0, 0L) * db[0];
            fprime[0] = eprime[0] - dprime[0];

            @<score inner loop@>

            @<score output@>

            if (W != R_NilValue)
                dW += iJ - 1;
        }

        da += iJ;
        db += iJ;

        dans += Jp + 1;

        /* constant C? p == 0*/
        if (p > 0)
            dC += p;
    }

    if (W == R_NilValue)
        PutRNGstate();

    UNPROTECT(1);
    return(ans);
}
@}

<<ex-score>>=
J <- 5
N <- 4

S <- crossprod(matrix(runif(J^2), nrow = J))
prm <- t(chol(S))[lower.tri(S, diag = TRUE)]

x <- matrix(prm, ncol = 1)
lx <- ltMatrices(x, byrow = TRUE, trans = TRUE, diag = TRUE)

a <- matrix(runif(N * J), nrow = J) - 2
b <- a + 4

M <- 100
W <- matrix(runif(M * (J - 1)), ncol = M)

phat <- c(lmvnorm(a, b, chol = lx, w = W, M = M, logLik = FALSE))

p <- unclass(lx)
fc <- function(prm, i) {
    L <- ltMatrices(matrix(prm, ncol = 1), byrow = TRUE, trans = TRUE, diag = TRUE)
    lmvnorm(a, b, chol = L, w = W, M = M)
}

S <- smvnorm(a, b, chol = lx, w = W, M = M)

chk(phat, S$logLik)

if (require("numDeriv"))
    print(max(abs(grad(fc, p) - rowSums(S$score))))
@@

Coming back to our simple example, we get (with $25000$ simple Monte-Carlo
iterations)
<<ex-again>>=
phat
exp(lmvnorm(a, b, chol = lx, M = 25000, logLik = FALSE))
@@

Next generate some data and compare our implementation to \code{pmvnorm}
using quasi-Monte-Carlo integration. The \code{pmvnorm}
function uses randomized Korobov rules.
The experiment here applies generalised Halton sequences. Plain Monte-Carlo
(\code{w = NULL}) will also work but produces more variable results. Results
will depend a lot on appropriate choices and it is the users'
responsibility to make sure things work as intended. If you are unsure, you
should use \code{pmvnorm} which provides a well-tested configuration.

<<ex-lmvnorm>>= )
M <- 10000
if (require("qrng")) {
    ### quasi-Monte-Carlo
    W <- t(ghalton(M * N, d = J - 1))
} else {
    ### Monte-Carlo
    W <- matrix(runif(M * N * (J - 1)), ncol = M)
}

### Genz & Bretz, 2001, without early stopping (really?)
pGB <- lmvnormR(a, b, chol = lx, logLik = FALSE, 
                algorithm = GenzBretz(maxpts = M, abseps = 0, releps = 0))
### Genz 1992 with quasi-Monte-Carlo
pGq <- exp(lmvnorm(a, b, chol = lx, w = W, M = M, logLik = FALSE))
### Genz 1992, original Monte-Carlo
pG <- exp(lmvnorm(a, b, chol = lx, w = NULL, M = M, logLik = FALSE))

cbind(pGB, pGq, pG)
@@

The three versions agree nicely. We now check if the code also works for
univariate problems

<<ex-uni>>=
### test univariate problem
### call pmvnorm
pGB <- lmvnormR(a[1,,drop = FALSE], b[1,,drop = FALSE], chol = lx[,1], logLik = FALSE, 
                algorithm = GenzBretz(maxpts = M, abseps = 0, releps = 0))
### call lmvnorm
pGq <- exp(lmvnorm(a[1,,drop = FALSE], b[1,,drop = FALSE], chol = lx[,1], logLik = FALSE))
### ground truth
ptr <- pnorm(b[1,] / c(unclass(lx[,1]))) - pnorm(a[1,] / c(unclass(lx[,1])))

cbind(c(ptr), pGB, pGq)
@@

The reason for small numerical differences is that \code{pmvnorm}
also uses \code{pnorm} but \code{lmvnorm} relies on our faster (but a bit
less accurate) version \code{C\_pnorm\_fast}.




\chapter{Maximum-likelihood Example} \label{ML}

We now discuss how this infrastructure can be used to estimate the Cholesky
factor of a multivariate normal in the presence of interval-censored
observations.

We first generate a covariance matrix $\Sigma = \mC \mC^\top$ and extract the Cholesky factor
$\mC$
<<ex-ML-dgp>>=
J <- 4
R <- diag(J)
R[1,2] <- R[2,1] <- .25
R[1,3] <- R[3,1] <- .5
R[2,4] <- R[4,2] <- .75
(Sigma <- diag(sqrt(1:J / 2)) %*% R %*% diag(sqrt(1:J / 2)))
(C <- t(chol(Sigma)))
@@

We know represent this matrix as \code{ltMatrices} object
<<ex-ML-C>>=
prm <- C[lower.tri(C, diag = TRUE)]
lt <- ltMatrices(matrix(prm, ncol = 1L), 
                 diag = TRUE,    ### has diagonal elements
                 byrow = FALSE,  ### prm is column-major
                 trans = TRUE)   ### store as J * (J + 1) / 2 x 1
lt <- ltMatrices(lt, 
                 byrow = TRUE,   ### convert to row-major
                 trans = TRUE)   ### keep dimensions
chk(C, as.array(lt)[,,1], check.attributes = FALSE)
chk(Sigma, as.array(Tcrossprod(lt))[,,1], check.attributes = FALSE)
@@

We now generate some data from $\N_\J(\mathbf{0}_\J, \Sigma)$. We first sample
from $\rZ \sim \N_\J(\mathbf{0}_\J, \mI_\J)$ and then $\rY = \mC \rZ \sim
\N_\J(\mathbf{0}_\J, \mC \mC^\top)$

<<ex-ML-data>>=
N <- 100
Z <- matrix(rnorm(N * J), nrow = J)
Y <- Mult(lt, Z)
Y <- Y - rowMeans(Y)
@@

Next we add some interval-censoring represented by \code{a} and \code{b}. 

<<ex-ML-cens>>=
sds <- sqrt(c(Tcrossprod(lt, diag_only = TRUE)))
rint <- runif(J * N, min = .5) * sds
a <- Y - rint
b <- Y + rint
@@

The true covariance matrix $\Sigma$ can be estimate from the uncensored data as

<<ex-ML-vcov>>=
(Shat <- var(t(Y)))
@@

Let's do some sanity and performance checks first. For different values of
$M$, we evaluate the log-likelihood using \code{pmvnorm} (called in
\code{lmvnormR}) and the simplified implementation. The comparion is a bit
unfair, because we do not add the time needed to setup Halton sequences, but
we would do this only once and use the stored values for repeated
evaluations of a log-likelihood (because the optimiser expects a
deterministic function to be optimised)

<<ex-ML-chk, eval = FALSE>>=
M <- floor(exp(0:25/10) * 1000)
lGB <- sapply(M, function(m) {
    st <- system.time(ret <- lmvnormR(a, b, chol = lt, algorithm = 
                                      GenzBretz(maxpts = m, abseps = 0, releps = 0)))
    return(c(st["user.self"], ll = ret))
})
lH <- sapply(M, function(m) {
    W <- NULL
    if (require("qrng"))
        W <- t(ghalton(m * N, d = J - 1))
    st <- system.time(ret <- lmvnorm(a, b, chol = lt, w = W, M = m))
    return(c(st["user.self"], ll = ret))
})
@@
The evaluated log-likelihoods and corresponding timings are given in
Figure~\ref{lleval}. It seems that for $M \ge 3000$, results are reasonably
stable.

\begin{figure}
<<ex-ML-fig, eval = FALSE, echo = FALSE, fig = TRUE, pdf = TRUE, width = 6, height = 4>>=
layout(matrix(1:2, nrow = 1))
plot(M, lGB["ll",], ylim = range(c(lGB["ll",], lH["ll",])), ylab = "Log-likelihood")
points(M, lH["ll",], pch = 4)
plot(M, lGB["user.self",], ylim = c(0, max(lGB["user.self",])), ylab = "Time (in sec)")
points(M, lH["user.self",], pch = 4)
legend("bottomright", legend = c("pmvnorm", "lmvnorm"), pch = c(1, 4), bty = "n")
@@
\caption{Evaluated log-likelihoods (left) and timings (right).
\label{lleval}}
\end{figure}

We now define the log-likelihood function. It is important to use weights
via the \code{w} argument (or to set the \code{seed}) such that only the
candidate parameters \code{parm} change with repeated calls to \code{ll}.

<<ex-ML-ll, eval = TRUE>>=
M <- 1000 
if (require("qrng")) {
    ### quasi-Monte-Carlo
    W <- t(ghalton(M * N, d = J - 1))
} else {
    ### Monte-Carlo
    W <- matrix(runif(M * N * (J - 1)), ncol = M)
}
ll <- function(parm) {
     C <- matrix(c(parm), ncol = 1L)
     C <- ltMatrices(C, diag = TRUE, byrow = TRUE, trans = TRUE)
     -lmvnorm(lower = a, upper = b, chol = C, w = W, M = M, logLik = TRUE)
}
@@

We can check the correctness of our log-likelihood function
<<ex-ML-check>>=
ll(unclass(lt))
lmvnormR(a, b, chol = lt, algorithm = GenzBretz(maxpts = M, abseps = 0, releps = 0))
(llprm <- lmvnorm(a, b, chol = lt, w = W, M = M))
chk(llprm, sum(lmvnorm(a, b, chol = lt, w = W, M = M, logLik = FALSE)))
@@

Finally, we can hand-over to \code{optim}. Because we need $\text{diag}(\mC) >
0$, we use box constraints and \code{method = "L-BFGS-B"}. We start with the
true $\mC$

<<ex-ML-sc>>=
sc <- function(parm) {
    C <- matrix(c(parm), ncol = 1L)
    C <- ltMatrices(C, diag = TRUE, byrow = TRUE, trans = TRUE)
    ret <- smvnorm(lower = a, upper = b, chol = C, w = W, M = M, logLik = TRUE)
    return(-rowSums(ret$score))
}
@@

<<ex-ML>>=
lwr <- rep(-Inf, J * (J + 1) / 2)
lwr[cumsum(c(1, 2:J))] <- 0.1

op <- optim(lt, fn = ll, gr = sc, method = "L-BFGS-B", lower = lwr, control = list(trace = TRUE))

op$value ## compare with 
ll(lt)
op$par   ## compare with
lt
@@

We can also compare the results on the scale of the covariance matrix

<<ex-ML-Shat>>=
Tcrossprod(lt)		### true Sigma
Tcrossprod(op$par)      ### interval-censored obs
Shat                    ### "exact" obs
@@

\chapter{Package Infrastructure}

@d R Header
@{
###    Copyright (C) 2022- Torsten Hothorn
###
###    This file is part of the 'mvtnorm' R add-on package.
###
###    'mvtnorm' is free software: you can redistribute it and/or modify
###    it under the terms of the GNU General Public License as published by
###    the Free Software Foundation, version 2.
###
###    'mvtnorm' is distributed in the hope that it will be useful,
###    but WITHOUT ANY WARRANTY; without even the implied warranty of
###    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
###    GNU General Public License for more details.
###
###    You should have received a copy of the GNU General Public License
###    along with 'mvtnorm'.  If not, see <http://www.gnu.org/licenses/>.
###
###
###    DO NOT EDIT THIS FILE
###
###    Edit 'lmvnorm_src.w' and run 'nuweb -r lmvnorm_src.w'
@}

@d C Header
@{
/*
    Copyright (C) 2022- Torsten Hothorn

    This file is part of the 'mvtnorm' R add-on package.

    'mvtnorm' is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    'mvtnorm' is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with 'mvtnorm'.  If not, see <http://www.gnu.org/licenses/>.


    DO NOT EDIT THIS FILE

    Edit 'lmvnorm_src.w' and run 'nuweb -r lmvnorm_src.w'
*/
@}


\chapter*{Index}

\section*{Files}

@f

\section*{Fragments}

@m

%\section*{Identifiers}
%
%@u

\bibliographystyle{plainnat}
\bibliography{\Sexpr{gsub("\\.bib", "", system.file("litdb.bib", package = "mvtnorm"))}}

\end{document}
