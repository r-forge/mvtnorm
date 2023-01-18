\documentclass[a4paper]{report}

%% packages
\usepackage{amsfonts,amstext,amsmath,amssymb,amsthm}

%\VignetteIndexEntry{Multivariate Normal Log-likelihoods}
%\VignetteDepends{mvtnorm,qrng,numDeriv}
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
\newcommand{\svec}{\mathbf{s}}
\newcommand{\jvec}{\mathbf{j}}
\newcommand{\muvec}{\mathbf{\mu}}
\newcommand{\rY}{\mathbf{Y}}
\newcommand{\rZ}{\mathbf{Z}}
\newcommand{\mC}{\mathbf{C}}
\newcommand{\mL}{\mathbf{L}}
\newcommand{\mP}{\mathbf{P}}
\newcommand{\mI}{\mathbf{I}}
\newcommand{\mS}{\mathbf{S}}
\newcommand{\mA}{\mathbf{A}}
\newcommand{\mSigma}{\mathbf{\Sigma}}
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

The implementation described here is a re-implementation (in \proglang{R}
and \proglang{C}) of Alan Genz' original \proglang{FORTRAN} code, focusing 
on efficient computation of the log-likelihood $\sum_{i = 1}^N \log(p_i)$
and the corresponding score function.

The document first describes a class and some useful methods for dealing with multiple lower triangular matrices $\mC_i, i = 1,
\dots, N$ in Chapter~\ref{ltMatrices}.  The multivariate normal
log-likelihood, and the corresponding score function, is implemented as
outlined in Chapter~\ref{lmvnorm}.  An example demonstrating
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
@<crossprod ltMatrices@>
@<chol syMatrices@>
@<marginal@>
@<conditional@>
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
@<chol@>
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

It is important to specify both \code{byrow} and \code{trans}, otherwise
the default arguments might chance the result in an unintended way.

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
rn <- paste0("C_", 1:N)
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
either column- or row-major order and this little helper function switches
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
rows/columns $j \in \{1, \dots, \J\}$ of the corresponding matrices $\mC_i$. 

@d subset ltMatrices
@{
"[.ltMatrices" <- function(x, i, j, ..., drop = FALSE) {

    if (drop) warning("argument drop is ignored")
    if (missing(i) && missing(j)) return(x)

    @<extract slots@>

    class(x) <- class(x)[-1L]

    if (!missing(j)) {

        j <- (1:J)[j] ### get rid of negative indices

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
standardised elementwise without transposing objects

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
/* return object: include unit diagonal elements if Rdiag == 0 */

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
class name \code{syMatrices}. We write one \proglang{C} function for
computing $\mC_i \mC_i^\top$ or $\mC_i^\top \mC_i$ (\code{Rtranspose} being
\code{TRUE}).

We differentiate between computation of the diagonal elements of the
crossproduct

@d first element
@{
dans[0] = 1.0;
if (Rdiag)
    dans[0] = pow(dC[0], 2);
if (Rtranspose) { // crossprod
    for (k = 1; k < iJ; k++) 
        dans[0] += pow(dC[IDX(k + 1, 1, iJ, Rdiag)], 2);
}
@}

@d tcrossprod diagonal only
@{
PROTECT(ans = allocMatrix(REALSXP, iJ, iN));
dans = REAL(ans);
for (n = 0; n < iN; n++) {
    @<first element@>
    for (i = 1; i < iJ; i++) {
        dans[i] = 0.0;
        if (Rtranspose) { // crossprod
            for (k = i + 1; k < iJ; k++)
                dans[i] += pow(dC[IDX(k + 1, i + 1, iJ, Rdiag)], 2);
        } else {         // tcrossprod
            for (k = 0; k < i; k++)
                dans[i] += pow(dC[IDX(i + 1, k + 1, iJ, Rdiag)], 2);
        }
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
    @<first element@>
    for (i = 1; i < iJ; i++) {
        for (j = 0; j <= i; j++) {
            ix = IDX(i + 1, j + 1, iJ, 1);
            dans[ix] = 0.0;
            if (Rtranspose) { // crossprod
                for (k = i + 1; k < iJ; k++)
                    dans[ix] += 
                        dC[IDX(k + 1, i + 1, iJ, Rdiag)] *
                        dC[IDX(k + 1, j + 1, iJ, Rdiag)];
            } else {         // tcrossprod
                for (k = 0; k < j; k++)
                    dans[ix] += 
                        dC[IDX(i + 1, k + 1, iJ, Rdiag)] *
                        dC[IDX(j + 1, k + 1, iJ, Rdiag)];
            }
            if (Rdiag) {
                if (Rtranspose) {
                    dans[ix] += 
                        dC[IDX(i + 1, i + 1, iJ, Rdiag)] *
                        dC[IDX(i + 1, j + 1, iJ, Rdiag)];
                } else {
                    dans[ix] += 
                        dC[IDX(i + 1, j + 1, iJ, Rdiag)] *
                        dC[IDX(j + 1, j + 1, iJ, Rdiag)];
                }
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

@d IDX
@{
#define IDX(i, j, n, d) ((i) >= (j) ? (n) * ((j) - 1) - ((j) - 2) * ((j) - 1)/2 + (i) - (j) - (!d) * (j) : 0)
@}

@d tcrossprod
@{

@<IDX@>

SEXP R_ltMatrices_tcrossprod (SEXP C, SEXP N, SEXP J, SEXP diag, SEXP diag_only, SEXP transpose) {

    SEXP ans;
    double *dans;
    int i, j, n, k, ix, nrow;

    @<RC input@>

    Rboolean Rdiag_only = asLogical(diag_only);
    Rboolean Rtranspose = asLogical(transpose);

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
Tcrossprod <- function(x, diag_only = FALSE, transpose = FALSE) {

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
                 as.logical(diag), as.logical(diag_only), as.logical(transpose))
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

We also add \code{Crossprod}, which is a call to \code{Tcrossprod} with the
\code{transpose} switch turned on

@d crossprod ltMatrices
@{
Crossprod <- function(x, diag_only = FALSE)
    Tcrossprod(x, diag_only = diag_only, transpose = TRUE)
@}

and run some checks

<<ex-crossprod>>=
## Crossprod
a <- as.array(Crossprod(lxn))
b <- array(apply(as.array(lxn), 3L, function(x) crossprod(x), simplify = TRUE), 
           dim = rev(dim(lxn)))
chk(a, b, check.attributes = FALSE)

# diagonal elements only
d <- Crossprod(lxn, diag_only = TRUE)
chk(d, apply(a, 3, diag))
chk(d, diagonals(Crossprod(lxn)))

a <- as.array(Crossprod(lxd))
b <- array(apply(as.array(lxd), 3L, function(x) crossprod(x), simplify = TRUE), 
           dim = rev(dim(lxd)))
chk(a, b, check.attributes = FALSE)

# diagonal elements only
d <- Crossprod(lxd, diag_only = TRUE)
chk(d, apply(a, 3, diag))
chk(d, diagonals(Crossprod(lxd)))
@@


\section{Cholesky Factorisation}

There might arise needs to compute the Cholesky factorisation $\mSigma_i = \mC_i
\mC_i^\top$ for multiple symmetric matrices $\mSigma_i$, stored as a matrix
in class \code{syMatrices}.

@d chol syMatrices
@{
chol.syMatrices <- function(x, ...) {

    byrow_orig <- attr(x, "byrow")
    trans_orig <- attr(x, "trans")
    dnm <- dimnames(x)
    stopifnot(attr(x, "diag"))
    d <- dim(x)

    ### x is of class syMatrices, coerse to ltMatrices first and re-arrange
    ### second
    x <- ltMatrices(unclass(x), diag = TRUE, trans = trans_orig, 
                    byrow = byrow_orig, names = dnm[[2L]])
    x <- ltMatrices(x, trans = TRUE, byrow = FALSE)
    class(x) <- class(x)[-1]
    storage.mode(x) <- "double"

    ret <- .Call(mvtnorm_R_syMatrices_chol, x, 
                 as.integer(d[1L]), as.integer(d[2L]))
    colnames(ret) <- dnm[[1L]]

    ret <- ltMatrices(ret, diag = TRUE, trans = TRUE, 
                      byrow = FALSE, names = dnm[[2L]])
    ret <- ltMatrices(ret, trans = trans_orig, byrow = byrow_orig)

    return(ret)
}
@}

Luckily, we already have the data in the correct packed colum-major storage,
so we swiftly loop over $i = 1, \dots, N$ in \proglang{C} and hand over to
\code{LAPACK}

@d chol
@{
SEXP R_syMatrices_chol (SEXP Sigma, SEXP N, SEXP J) {

    SEXP ans;
    double *dans, *dSigma;
    int iJ = INTEGER(J)[0];
    int pJ = iJ * (iJ + 1) / 2;
    int iN = INTEGER(N)[0];
    int i, j, info = 0;
    char lo = 'L';

    PROTECT(ans = allocMatrix(REALSXP, pJ, iN));
    dans = REAL(ans);
    dSigma = REAL(Sigma);

    for (i = 0; i < iN; i++) {

        /* copy data */
        for (j = 0; j < pJ; j++)
            dans[j] = dSigma[j];

        F77_CALL(dpptrf)(&lo, &iJ, dans, &info FCONE);

        if (info != 0) {
            if (info > 0)
                error("the leading minor of order %d is not positive definite",
                      info);
            error("argument %d of Lapack routine %s had invalid value",
                  -info, "dpptrf");
        }

        dSigma += pJ;
        dans += pJ;
    }
    UNPROTECT(1);
    return(ans);
}
@}

This new \code{chol} method can be used to revert \code{Tcrossprod} for
\code{ltMatrices} with and without unit diagonals:

<<chol>>=
Sigma <- Tcrossprod(lxd)
chk(chol(Sigma), lxd)
Sigma <- Tcrossprod(lxn)
## Sigma and chol(Sigma) always have diagonal, lxn doesn't
chk(as.array(chol(Sigma)), as.array(lxn))
@@

\section{Marginal and Conditional Normal Distributions}

Marginal and conditional distributions from distributions $\rY_i \sim \N_\J(\mathbf{0}_\J, \mC_i \mC_i^\top)$
(\code{chol} argument for $\mC_i$ for $i = 1, \dots, N$) or $\rY_i \sim \N_\J(\mathbf{0}_\J, \mL_i^{-1} \mL_i^{-\top})$
(\code{invchol} argument for $\mL_i$ for $i = 1, \dots, N$) shall be
computed.

@d mc input checks
@{
stopifnot(xor(missing(chol), missing(invchol)))
x <- if (missing(chol)) invchol else chol

stopifnot(inherits(x, "ltMatrices"))

N <- dim(x)[1L]
J <- dim(x)[2L]
if (is.character(which)) which <- match(which, dimnames(x)[[2L]])
stopifnot(all(which %in% 1:J))
@}

The first $j$ marginal distributions can be obtained from subsetting $\mC$
or $\mL$ directly. Arbitrary marginal distributions are based on the
corresponding subset of the covariance matrix for which we compute a
corresponding Cholesky factor (such that we can use \code{lmvnorm} later
on).

@d marginal
@{
marg_mvnorm <- function(chol, invchol, which = 1L) {

    @<mc input checks@>

    if (which[1] == 1L && (length(which) == 1L || 
                           all(diff(which) == 1L))) {
        ### which is 1:j
        tmp <- x[,which]
    } else {
        if (missing(chol)) x <- solve(x)
        tmp <- mvtnorm:::chol.syMatrices(Tcrossprod(x)[,which])
        if (missing(chol)) tmp <- solve(tmp)
    }

    if (missing(chol))
        ret <- list(invchol = tmp)
    else
        ret <- list(chol = tmp)

    ret
}
@}

We compute conditional distributions from the precision matrices
$\mSigma^{-1}_i = \mP_i = \mL_i^\top \mL_i$ (we omit the $i$ index from now
on). For an arbitrary subset $\jvec
\subset \{1, \dots, J\}$, the conditional distribution of $\rY_{-\jvec}$
given $\rY_{\jvec} = \yvec_{\jvec}$ is
\begin{eqnarray*}
\rY_{-\jvec} \mid \rY_{\jvec} = \yvec_{\jvec} \sim 
  \N_{|\jvec|}\left(-\mP^{-1}_{-\jvec,-\jvec} \mP_{-\jvec, \jvec} \yvec_{\jvec}, 
                    \mP^{-1}_{-\jvec,-\jvec}\right)
\end{eqnarray*}
and we return a Cholesky factor $\tilde{\mC}$ such that
$\mP^{-1}_{-\jvec,-\jvec} = \tilde{\mC} \tilde{\mC}^\top$ (if \code{chol} was
given) or $\tilde{\mL} = \tilde{\mC}^{-1}$ (if \code{invchol} was given).

@d conditional
@{
cond_mvnorm <- function(chol, invchol, which = 1L, given) {

    @<mc input checks@>

    if (N == 1) N <- NCOL(given)
    stopifnot(is.matrix(given) && nrow(given) == length(which) && ncol(given) == N)

    if (!missing(chol)) ### chol is C = Cholesky of covariance
        P <- Crossprod(solve(chol)) ### P = t(L) %*% L with L = C^-1
    else                ### invcol is L = Cholesky of precision
        P <- Crossprod(invchol)

    Pw <- P[, -which]
    chol <- solve(mvtnorm:::chol.syMatrices(Pw))
    Pa <- as.array(P)
    Sa <- as.array(S <- Crossprod(chol))
    if (dim(chol)[1L] == 1L) {
        Pa <- Pa[,,1]
        Sa <- Sa[,,1]
        mean <- -Sa %*% Pa[-which, which, drop = FALSE] %*% given
    } else {
        mean <- sapply(1:N, function(i) -Sa[,,i] %*% Pa[-which,which,i] %*% given[,i,drop = FALSE])
    }

    chol <- mvtnorm:::chol.syMatrices(S)
    if (missing(invchol)) 
        return(list(mean = mean, chol = chol))

    return(list(mean = mean, invchol = solve(chol)))
}
@}

Let's check this against the commonly used formula based on the covariance
matrix, first for the marginal distribution

<<marg>>=
Sigma <- Tcrossprod(lxd)
j <- 1:3
chk(Sigma[,j], Tcrossprod(marg_mvnorm(chol = lxd, which = j)$chol))
j <- 2:4
chk(Sigma[,j], Tcrossprod(marg_mvnorm(chol = lxd, which = j)$chol))

Sigma <- Tcrossprod(solve(lxd))
j <- 1:3
chk(Sigma[,j], Tcrossprod(solve(marg_mvnorm(invchol = lxd, which = j)$invchol)))
j <- 2:4
chk(Sigma[,j], Tcrossprod(solve(marg_mvnorm(invchol = lxd, which = j)$invchol)))
@@

and then for conditional distributions

<<cond>>=
Sigma <- as.array(Tcrossprod(lxd))[,,1]
j <- 2:4
y <- matrix(c(-1, 2, 1), nrow = 3)

cm <- Sigma[-j, j,drop = FALSE] %*% solve(Sigma[j,j]) %*%  y
cS <- Sigma[-j, -j] - Sigma[-j,j,drop = FALSE] %*% solve(Sigma[j,j]) %*% Sigma[j,-j,drop = FALSE]

cmv <- cond_mvnorm(chol = lxd[1,], which = j, given = y)

chk(cm, cmv$mean)
chk(cS, as.array(Tcrossprod(cmv$chol))[,,1])

Sigma <- as.array(Tcrossprod(solve(lxd)))[,,1]
j <- 2:4
y <- matrix(c(-1, 2, 1), nrow = 3)

cm <- Sigma[-j, j,drop = FALSE] %*% solve(Sigma[j,j]) %*%  y
cS <- Sigma[-j, -j] - Sigma[-j,j,drop = FALSE] %*% solve(Sigma[j,j]) %*% Sigma[j,-j,drop = FALSE]

cmv <- cond_mvnorm(invchol = lxd[1,], which = j, given = y)

chk(cm, cmv$mean)
chk(cS, as.array(Tcrossprod(solve(cmv$invchol)))[,,1])

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
ll1 <- sum(dnorm(Mult(lt, Y), log = TRUE)) + sum(log(diagonals(lt)))

S <- as.array(Tcrossprod(solve(lt)))
ll2 <- sum(sapply(1:N, function(i) dmvnorm(x = Y[,i], sigma = S[,,i], log = TRUE)))
chk(ll1, ll2)
@@

Sometimes it is preferable to split the joint distribution into a marginal
distribution of some elements and the conditional distribution given these
elements. The joint density is, of course, the product of the marginal and
conditional densities and we can check if this works for our example by

<<ex-MV-mc>>=
## marginal of and conditional on these
(j <- 1:5 * 10)
md <- marg_mvnorm(invchol = lt, which = j)
cd <- cond_mvnorm(invchol = lt, which = j, given = Y[j,])

ll3 <- sum(dnorm(Mult(md$invchol, Y[j,]), log = TRUE)) + 
       sum(log(diagonals(md$invchol))) +
       sum(dnorm(Mult(cd$invchol, Y[-j,] - cd$mean), log = TRUE)) + 
       sum(log(diagonals(cd$invchol)))
chk(ll1, ll3)
@@


\chapter{Multivariate Normal Log-likelihoods} \label{lmvnorm}

We now discuss code for evaluating the log-likelihood
\begin{eqnarray*}
\sum_{i = 1}^N \log(p_i(\mC_i \mid \avec_i, \bvec_i))
\end{eqnarray*}

This is relatively simple to achieve using the existing \code{pmvnorm}, so a
prototype might look like

@d lmvnormR
@{
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
@}

<<fct-lmvnormR, echo = FALSE>>=
@<lmvnormR@>
@@

However, the underlying \proglang{FORTRAN} code first computes the Cholesky
factor based on the covariance matrix, which is clearly a waste of time.
Repeated calls to \proglang{FORTRAN} also cost some time. The code \citep[based
on and evaluated in][]{Genz_Bretz_2002} implements a
specific form of quasi-Monte-Carlo integration without allowing the user to
change the scheme (or to fall-back to simple Monte-Carlo). We therefore
implement our own, and simplistic version, with the aim to speed-things up
such that maximum-likelihood estimation becomes a bit faster.

Let's look at an example first. This code estimates $p_1, \dots, p_{10}$ for
a $5$-dimensional normal
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
@<.gradSolveL@>
@}

@o lmvnorm.c -cc
@{
@<C Header@>
#include <R.h>
#include <Rmath.h>
#include <Rinternals.h>
#include <Rdefines.h>
#include <Rconfig.h>
#include <R_ext/BLAS.h> /* for dtrmm */
@<pnorm fast@>
@<pnorm slow@>
@<R lmvnorm@>
@<R smvnorm@>
@<grad solve(L)@>
@}

We implement the algorithm described by \cite{numerical-:1992}. The key
point here is that the original $\J$-dimensional problem~(\ref{pmvnorm}) is transformed into
an integral over $[0, 1]^{\J - 1}$.

For each $i = 1, \dots, N$, do

\begin{enumerate}
  \item Input $\mC_i$ (\code{chol}), $\avec_i$ (\code{lower}), $\bvec_i$
(\code{upper}), and control parameters $\alpha$, $\epsilon$, and $M_\text{max}$ (\code{M}).

@d input checks
@{
if (!is.matrix(lower)) lower <- matrix(lower, ncol = 1)
if (!is.matrix(upper)) upper <- matrix(upper, ncol = 1)
stopifnot(isTRUE(all.equal(dim(lower), dim(upper))))

stopifnot(inherits(chol, "ltMatrices"))
byrow_orig <- attr(chol, "byrow")
trans_orig <- attr(chol, "trans")
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


  \item Standardise integration limits $a^{(i)}_j / c^{(i)}_{jj}$, $b^{(i)}_j / c^{(i)}_{jj}$, and rows $c^{(i)}_{j\jmath} / c^{(i)}_{jj}$ for $1 \le \jmath < j < \J$.


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


  \item Initialise $\text{intsum} = \text{varsum} = 0$, $M = 0$, $d_1 =
\Phi\left(a^{(i)}_1\right)$, $e_1 = \Phi\left(b^{(i)}_1\right)$ and $f_1 = e_1 - d_1$.


@d initialisation
@{
d0 = pnorm_ptr(da[0], 0.0);
e0 = pnorm_ptr(db[0], 0.0);
emd0 = e0 - d0;
f0 = emd0;
intsum = (iJ > 1 ? 0.0 : f0);
@}

  \item Repeat

@d init logLik loop
@{
d = d0;
f = f0;
emd = emd0;
start = 0;
@}

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
d = pnorm_ptr(da[j], x);
e = pnorm_ptr(db[j], x);
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

@d inner logLik loop
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

and move on to the next observation (note that \code{p} might be 0, in case
$\mC_i \equiv \mC$).

@d move on
@{
da += iJ;
db += iJ;
dC += p;
@}

\end{enumerate}

It turned out that calls to \code{pnorm} are expensive, so a slightly faster
alternative \citep[suggested by][]{Matic_Radoicic_Stefanica_2018} can be used
(\code{fast = TRUE} in the calls to \code{lmvnorm} and \code{smvnorm}):

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

@d pnorm slow
@{
double C_pnorm_slow (double x, double m) {
    return(pnorm(x, m, 1.0, 1L, 0L));
}
@}

The \code{fast} argument can be used to switch on the faster but less
accurate version of \code{pnorm}

@d pnorm
@{
Rboolean Rfast = asLogical(fast);
double (*pnorm_ptr)(double, double) = C_pnorm_slow;
if (Rfast)
    pnorm_ptr = C_pnorm_fast;
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
SEXP R_lmvnorm(SEXP a, SEXP b, SEXP C, SEXP N, SEXP J, SEXP W, SEXP M, SEXP tol, SEXP logLik, SEXP fast) {

    SEXP ans;
    double *da, *db, *dC, *dW, *dans, dtol = REAL(tol)[0];
    double mdtol = 1.0 - dtol;
    double d0, e0, emd0, f0, q0, l0, lM, intsum;
    int p, len;

    Rboolean RlogLik = asLogical(logLik);

    @<pnorm@>

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

            @<init logLik loop@>

            @<inner logLik loop@>

            @<increment@>

            if (W != R_NilValue)
                dW += iJ - 1;
        }

        @<output@>

        @<move on@>
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

Sometimes we want to evaluate the log-likelihood based on $\mL = \mC^{-1}$,
the Cholesky factor of the precision (not the covariance) matrix. In this
case, we explicitly invert $\mL$ to give $\mC$ (both matrices are lower
triangular, so this is fast).

@d Cholesky of precision
@{
stopifnot(xor(missing(chol), missing(invchol)))
if (missing(chol)) chol <- solve(invchol)
@}

@d lmvnorm
@{
lmvnorm <- function(lower, upper, mean = 0, chol, invchol, logLik = TRUE, M = NULL, 
                    w = NULL, seed = NULL, tol = .Machine$double.eps, fast = FALSE) {

    @<init random seed, reset on exit@>

    @<Cholesky of precision@>

    @<input checks@>

    @<standardise@>

    @<check and / or set integration weights@>

    ret <- .Call(mvtnorm_R_lmvnorm, ac, bc, unclass(C), as.integer(N), 
                 as.integer(J), w, as.integer(M), as.double(tol), 
                 as.logical(logLik), as.logical(fast));
    return(ret)
}
@}


Coming back to our simple example, we get (with $25000$ simple Monte-Carlo
iterations)
<<ex-again>>=
phat
exp(lmvnorm(a, b, chol = lx, M = 25000, logLik = FALSE, fast = TRUE))
exp(lmvnorm(a, b, chol = lx, M = 25000, logLik = FALSE, fast = FALSE))
@@

Next we generate some data and compare our implementation to \code{pmvnorm}
using quasi-Monte-Carlo integration. The \code{pmvnorm}
function uses randomised Korobov rules.
The experiment here applies generalised Halton sequences. Plain Monte-Carlo
(\code{w = NULL}) will also work but produces more variable results. Results
will depend a lot on appropriate choices and it is the users
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
### Genz 1992 with quasi-Monte-Carlo, fast pnorm
pGqf <- exp(lmvnorm(a, b, chol = lx, w = W, M = M, logLik = FALSE, fast = TRUE))
### Genz 1992, original Monte-Carlo, fast pnorm
pGf <- exp(lmvnorm(a, b, chol = lx, w = NULL, M = M, logLik = FALSE, fast = TRUE))
### Genz 1992 with quasi-Monte-Carlo, R::pnorm
pGqs <- exp(lmvnorm(a, b, chol = lx, w = W, M = M, logLik = FALSE, fast = FALSE))
### Genz 1992, original Monte-Carlo, R::pnorm
pGs <- exp(lmvnorm(a, b, chol = lx, w = NULL, M = M, logLik = FALSE, fast = FALSE))

cbind(pGB, pGqf, pGf, pGqs, pGs)
@@

The three versions agree nicely. We now check if the code also works for
univariate problems

<<ex-uni>>=
### test univariate problem
### call pmvnorm
pGB <- lmvnormR(a[1,,drop = FALSE], b[1,,drop = FALSE], chol = lx[,1], 
                logLik = FALSE, 
                algorithm = GenzBretz(maxpts = M, abseps = 0, releps = 0))
### call lmvnorm
pGq <- exp(lmvnorm(a[1,,drop = FALSE], b[1,,drop = FALSE], chol = lx[,1], 
                   logLik = FALSE))
### ground truth
ptr <- pnorm(b[1,] / c(unclass(lx[,1]))) - pnorm(a[1,] / c(unclass(lx[,1])))

cbind(c(ptr), pGB, pGq)
@@

Because the default \code{fast = FALSE} was used here, all results are
identical.

\section{Score Function}

In addition to the log-likelihood, we would also like to have access to the
scores with respect to $\mC_i$. Because every element of $\mC_i$ only enters
once, the chain rule rules, so to speak.


@d score output object
@{
int Jp = iJ * (iJ + 1) / 2;
double dprime[Jp], eprime[Jp], fprime[Jp], yprime[(iJ - 1) * Jp];
double aprime[iJ], bprime[iJ], fmprime[iJ], ymprime[(iJ - 1) * iJ];
double dtmp, etmp, Wtmp, ytmp, xx;

PROTECT(ans = allocMatrix(REALSXP, Jp + 1 + iJ, iN));
dans = REAL(ans);
for (j = 0; j < LENGTH(ans); j++) dans[j] = 0.0;
@}

For each $i = 1, \dots, N$, do

\begin{enumerate}
  \item Input $\mC_i$ (\code{chol}), $\avec_i$ (\code{lower}), $\bvec_i$
(\code{upper}), and control parameters $\alpha$, $\epsilon$, and $M_\text{max}$ (\code{M}).

  \item Standardise integration limits $a^{(i)}_j / c^{(i)}_{jj}$, $b^{(i)}_j / c^{(i)}_{jj}$, and rows $c^{(i)}_{j\jmath} / c^{(i)}_{jj}$ for $1 \le \jmath < j < \J$.

Note: We later need derivatives wrt $c^{(i)}_{jj}$, so we compute derivates
wrt $a^{(i)}_j$ and $b^{(i)}_j$ and post-differentiate later.

  \item Initialise $\text{intsum} = \text{varsum} = 0$, $M = 0$, $d_1 =
\Phi\left(a^{(i)}_1\right)$, $e_1 = \Phi\left(b^{(i)}_1\right)$ and $f_1 = e_1 - d_1$.

We start initialised the score wrt to $c^{(i)}_{11}$ (the parameter is non-existent
here due to standardisation)

@d score c11
@{
dprime[0] = (R_FINITE(da[0]) ? dnorm(da[0], 0.0, 1.0, 0L) * da[0] : 0);
eprime[0] = (R_FINITE(db[0]) ? dnorm(db[0], 0.0, 1.0, 0L) * db[0] : 0);
fprime[0] = eprime[0] - dprime[0];
@}

@d score a, b
@{
aprime[0] = (R_FINITE(da[0]) ? dnorm(da[0], 0.0, 1.0, 0L) : 0);
bprime[0] = (R_FINITE(db[0]) ? dnorm(db[0], 0.0, 1.0, 0L) : 0);
fmprime[0] = bprime[0] - aprime[0];
@}

  \item Repeat

@d init score loop
@{
@<init logLik loop@>
@<score c11@>
@<score a, b@>
@}

    \begin{enumerate}

      \item Generate uniform $w_1, \dots, w_{\J - 1} \in [0, 1]$.

      \item For $j = 2, \dots, J$ set 
        \begin{eqnarray*}
            y_{j - 1} & = & \Phi^{-1}\left(d_{j - 1} + w_{j - 1} (e_{j - 1} - d_{j - 1})\right)
        \end{eqnarray*}

We again either generate $w_{j - 1}$ on the fly or use pre-computed weights
(\code{w}). We first compute the scores with respect to the already existing
parameters.

@d update yprime
@{
ytmp = exp(- dnorm(y[j - 1], 0.0, 1.0, 1L)); // = 1 / dnorm(y[j - 1], 0.0, 1.0, 0L)

for (k = 0; k < Jp; k++) yprime[k * (iJ - 1) + (j - 1)] = 0.0;

for (idx = 0; idx < (j + 1) * j / 2; idx++) {
    yprime[idx * (iJ - 1) + (j - 1)] = ytmp;
    yprime[idx * (iJ - 1) + (j - 1)] *= (dprime[idx] + Wtmp * (eprime[idx] - dprime[idx]));
}
@}

@d update yprime for a, b
@{
for (k = 0; k < iJ; k++)
    ymprime[k * (iJ - 1) + (j - 1)] = 0.0;

for (idx = 0; idx < j; idx++) {
    ymprime[idx * (iJ - 1) + (j - 1)] = ytmp;
    ymprime[idx * (iJ - 1) + (j - 1)] *= (aprime[idx] + Wtmp * (bprime[idx] - aprime[idx]));
}
@}

        \begin{eqnarray*}
            x_{j - 1} & = & \sum_{\jmath = 1}^{j - 1} c^{(i)}_{j\jmath} y_j
\end{eqnarray*}

        \begin{eqnarray*}
            d_j & = & \Phi\left(a^{(i)}_j - x_{j - 1}\right) \\
            e_j & = & \Phi\left(b^{(i)}_j - x_{j - 1}\right)
        \end{eqnarray*}

        \begin{eqnarray*}
            f_j & = & (e_j - d_j) f_{j - 1}.
       \end{eqnarray*}

The scores with respect to $c^{(i)}_{j\jmath}, \jmath = 1, \dots, j - 1$ are

@d score wrt new off-diagonals
@{
dtmp = dnorm(da[j], x, 1.0, 0L);
etmp = dnorm(db[j], x, 1.0, 0L);

for (k = 0; k < j; k++) {
    idx = start + j + k;
    dprime[idx] = dtmp * (-1.0) * y[k];
    eprime[idx] = etmp * (-1.0) * y[k];
    fprime[idx] = (eprime[idx] - dprime[idx]) * f;
}
@}

and the score with respect to (the here non-existing) $c^{(i)}_{jj}$ is

@d score wrt new diagonal
@{
idx = (j + 1) * (j + 2) / 2 - 1;
dprime[idx] = (R_FINITE(da[j]) ? dtmp * (da[j] - x) : 0);
eprime[idx] = (R_FINITE(db[j]) ? etmp * (db[j] - x) : 0);
fprime[idx] = (eprime[idx] - dprime[idx]) * f;
@}

@d new score a, b
@{
aprime[j] = (R_FINITE(da[j]) ? dtmp : 0);
bprime[j] = (R_FINITE(db[j]) ? etmp : 0);
fmprime[j] = (bprime[j] - aprime[j]) * f;
@}


We next update scores for parameters introduced for smaller $j$

@d update score
@{
for (idx = 0; idx < j * (j + 1) / 2; idx++) {
    xx = 0.0;
    for (k = 0; k < j; k++)
        xx += dC[start + k] * yprime[idx * (iJ - 1) + k];

    dprime[idx] = dtmp * (-1.0) * xx;
    eprime[idx] = etmp * (-1.0) * xx;
    fprime[idx] = (eprime[idx] - dprime[idx]) * f + emd * fprime[idx];
}
@}

@d update score a, b
@{
for (idx = 0; idx < j; idx++) {
    xx = 0.0;
    for (k = 0; k < j; k++)
        xx += dC[start + k] * ymprime[idx * (iJ - 1) + k];

    aprime[idx] = dtmp * (-1.0) * xx;
    bprime[idx] = etmp * (-1.0) * xx;
    fmprime[idx] = (bprime[idx] - aprime[idx]) * f + emd * fmprime[idx];
}
@}


We put everything together in a loop starting with the second dimension

@d inner score loop
@{
for (j = 1; j < iJ; j++) {

    @<compute y@>

    @<compute x@>

    @<update d, e@>

    @<update yprime@>

    @<update yprime for a, b@>

    @<score wrt new off-diagonals@>

    @<score wrt new diagonal@>

    @<new score a, b@>

    @<update score@>

    @<update score a, b@>

    @<update f@>

}
@}

      \item Set $\text{intsum} = \text{intsum} + f_\J$, $\text{varsum} = \text{varsum} + f^2_\J$, $M = M + 1$, 
            and $\text{error} = \sqrt{(\text{varsum}/M - (\text{intsum}/M)^2) / M}$.

We refrain from early stopping and error estimation. 
    
      \item[Until] $\text{error} < \epsilon$ or $M = M_\text{max}$

    \end{enumerate}
  \item Output $\hat{p}_i = \text{intsum} / M$.

We return $\log{\hat{p}_i}$ for each $i$, or we immediately sum-up over $i$.


@d score output
@{
dans[0] += f;
for (j = 0; j < Jp; j++)
    dans[j + 1] += fprime[j];
for (j = 0; j < iJ; j++)
    dans[Jp + j + 1] += fmprime[j];
@}

\end{enumerate}

We put everything together in \proglang{C}

@d R smvnorm
@{
SEXP R_smvnorm(SEXP a, SEXP b, SEXP C, SEXP N, SEXP J, SEXP W, 
               SEXP M, SEXP tol, SEXP fast) {

    SEXP ans;
    double *da, *db, *dC, *dW, *dans, dtol = REAL(tol)[0];
    double mdtol = 1.0 - dtol;
    double d0, e0, emd0, f0, q0, intsum;
    int p, idx;

    @<dimensions@>
    @<pnorm@>
    @<W length@>

    int start, j, k;
    double tmp, e, d, f, emd, x, y[iJ - 1];

    @<score output object@>

    q0 = qnorm(dtol, 0.0, 1.0, 1L, 0L);

    /* univariate problem */
    if (iJ == 1) iM = 0; 

    if (W == R_NilValue)
        GetRNGstate();

    for (int i = 0; i < iN; i++) {

        @<initialisation@>
        @<score c11@>

        if (iM == 0) {
            dans[0] = intsum;
            dans[1] = fprime[0];
        }

        if (W != R_NilValue && pW == 0)
            dW = REAL(W);

        for (int m = 0; m < iM; m++) {

            @<init score loop@>
            @<inner score loop@>
            @<score output@>

            if (W != R_NilValue)
                dW += iJ - 1;
        }

        @<move on@>

        dans += Jp + 1 + iJ;
    }

    if (W == R_NilValue)
        PutRNGstate();

    UNPROTECT(1);
    return(ans);
}
@}

The \proglang{R} code is now essentially identical to \code{lmvnorm},
however, we need to undo the effect of standardisation once the scores have
been computed

@d post differentiate mean score
@{
Jp <- J * (J + 1) / 2;
smean <- - ret[Jp + 1:J, , drop = FALSE] / c(dchol)
@}

@d post differentiate chol score
@{
if (J == 1) {
    idx <- 1L
} else {
    idx <- cumsum(c(1, 2:J))
}
if (attr(chol, "diag")) {
    ret <- ret / c(dchol[rep(1:J, 1:J),]) ### because 1 / dchol already there
    ret[idx,] <- -ret[idx,]
} else {
    ### remove scores for constant diagonal elements
    ret <- ret[-idx, drop = FALSE]    
}
ret <- ltMatrices(ret, diag = attr(chol, "diag"), trans = TRUE, byrow = TRUE)
@}

We sometimes parameterise models in terms of $\mL = \mC^{-1}$, the Cholesky
factor of the precision matrix. The log-likelihood operates on $\mC$, so we
need to post-differentiate the score function. We have
\begin{eqnarray*}
\mA = \frac{\partial \mL^{-1}}{\partial \mL} = - \mL^{-\top} \otimes \mL^{-1}
\end{eqnarray*}
and computing $\svec \mA$ for a score vector $\svec$ with respect to $\mL$ can be
implemented by the ``vec trick''
\begin{eqnarray*}
\svec \mA = \mL^{-\top} \mS \mL^{-\top}
\end{eqnarray*}
where $\svec = \text{vec}(\mS)$.

@d t(C) S t(C)
@{
char si = 'R', lo = 'L', tr = 'N', trT = 'T', di = 'N';
double ONE = 1.0;
int idx;
int iJ2 = iJ * iJ;

double tmp[iJ2];
for (j = 0; j < iJ2; j++) tmp[j] = 0.0;

ans = PROTECT(allocMatrix(REALSXP, iJ2, iN));
dans = REAL(ans);

for (i = 0; i < LENGTH(ans); i++) dans[i] = 0.0;

for (i = 0; i < iN; i++) {

    /* B := t(C) */
    for (j = 0; j < iJ; j++) {
        for (k = 0; k <= j; k++) {
            idx = IDX(j + 1, k + 1, iJ, 1L);
            dans[j * iJ + k] = dC[idx];
            /* argument A in dtrmm is not in packed form, so exand in J x J
               matrix */
            tmp[k * iJ + j] = dS[idx];
        }
    }

    /* B := B %*% S */
    F77_CALL(dtrmm)(&si, &lo, &tr , &di, &iJ, &iJ, &ONE, tmp, &iJ, 
                    dans, &iJ FCONE FCONE FCONE FCONE);

    for (j = 0; j < iJ; j++) {
        for (k = 0; k <= j; k++)
            tmp[k * iJ + j] = dC[IDX(j + 1, k + 1, iJ, 1L)];
    }

    /* B := B %*% t(C) */
    F77_CALL(dtrmm)(&si, &lo, &trT, &di, &iJ, &iJ, &ONE, tmp, &iJ, 
                    dans, &iJ FCONE FCONE FCONE FCONE);

    dans += iJ2;
    dC += p;
    dS += len;
}    
@}

@d grad solve(L)
@{

@<IDX@>

SEXP R_gradSolveL(SEXP C, SEXP N, SEXP J, SEXP S, SEXP diag) {

    int i, j, k;
    SEXP ans;
    double *dS, *dans;

    @<RC input@>
    @<C length@>
    dS = REAL(S);
    @<t(C) S t(C)@>

    UNPROTECT(1);
    return(ans);
}
@}

@d .gradSolveL
@{
.gradSolveL <- function(C, S, diag = FALSE) {

    stopifnot(inherits(C, "ltMatrices"))
    stopifnot(attr(C, "diag"))
    stopifnot(inherits(S, "ltMatrices"))
    stopifnot(attr(S, "diag"))

    C_byrow_orig <- attr(C, "byrow")
    C_trans_orig <- attr(C, "trans")
    S_byrow_orig <- attr(S, "byrow")
    S_trans_orig <- attr(S, "trans")

    stopifnot(S_byrow_orig == C_byrow_orig)
    stopifnot(S_trans_orig == C_trans_orig)

    C <- ltMatrices(C, byrow = FALSE, trans = TRUE)
    dC <- dim(C)
    nm <- attr(C, "rcnames")
    S <- ltMatrices(S, byrow = FALSE, trans = TRUE)
    dS <- dim(S)
    stopifnot(dC[2L] == dS[2L])
    if (dC[1] != 1L)
        stopifnot(dC[1L] == dS[1L])
    N <- dS[1L]
    J <- dS[2L]

    class(C) <- class(C)[-1L]
    storage.mode(C) <- "double"
    class(S) <- class(S)[-1L]
    storage.mode(S) <- "double"
            
    ret <- .Call(mvtnorm_R_gradSolveL, C, as.integer(N), as.integer(J), S, as.logical(TRUE))

    L <- matrix(1:(J^2), nrow = J)
    ret <- ltMatrices(ret[L[lower.tri(L, diag = diag)],,drop = FALSE], 
                      diag = diag, byrow = FALSE, trans = TRUE, names = nm)
    ret <- ltMatrices(ret, byrow = C_byrow_orig, trans = C_trans_orig)
    return(ret)
}
@}

Here is a small example

<<kronecker>>=
J <- 10

d <- TRUE
L <- diag(J)
L[lower.tri(L, diag = d)] <- prm <- runif(J * (J + c(-1, 1)[d + 1]) / 2)

C <- solve(L)

D <- -kronecker(t(C), C)

S <- diag(J)
S[lower.tri(S, diag = TRUE)] <- x <- runif(J * (J + 1) / 2)

SD0 <- matrix(c(S) %*% D, ncol = J)

SD1 <- -crossprod(C, tcrossprod(S, C))

a <- ltMatrices(C[lower.tri(C, diag = TRUE)], diag = TRUE, byrow = FALSE, trans = TRUE)
b <- ltMatrices(x, diag = TRUE, byrow = FALSE, trans = TRUE)

SD2 <- -mvtnorm:::.gradSolveL(a, b, diag = d)

chk(SD0[lower.tri(SD0, diag = d)], 
    SD1[lower.tri(SD1, diag = d)])
chk(SD0[lower.tri(SD0, diag = d)],
    c(unclass(SD2)))
@@

@d post differentiate invchol score
@{
if (!missing(invchol))
    ret <- - .gradSolveL(chol, ret, diag = TRUE)
@}

We can now finally put everything together in a single score function.

@d smvnorm
@{
smvnorm <- function(lower, upper, mean = 0, chol, invchol, logLik = TRUE, M = NULL, 
                    w = NULL, seed = NULL, tol = .Machine$double.eps, fast = FALSE) {

    @<init random seed, reset on exit@>

    @<Cholesky of precision@>

    @<input checks@>

    @<standardise@>

    @<check and / or set integration weights@>

    ret <- .Call(mvtnorm_R_smvnorm, ac, bc, unclass(C), as.integer(N), 
                 as.integer(J), w, as.integer(M), as.double(tol), as.logical(fast));

    ll <- log(pmax(ret[1L,], tol)) - log(M)
    intsum <- ret[1L,]
    intsum <- pmax(intsum, tol^(1/3) * M) ### see mlt:::..mlt_score_interval
    m <- matrix(intsum, nrow = nrow(ret) - 1, ncol = ncol(ret), byrow = TRUE)
    ret <- ret[-1L,,drop = FALSE] / m

    @<post differentiate mean score@>

    ret <- ret[1:Jp, , drop = FALSE]

    @<post differentiate chol score@>

    @<post differentiate invchol score@>

    ret <- ltMatrices(ret, byrow = byrow_orig, trans = trans_orig)

    if (logLik) {
        ret <- list(logLik = ll, 
                    mean = smean, 
                    chol = ret)
        return(ret)
    }
    
    return(ret)
}
@}

Let's look at an example, where we use \code{numDeriv::grad} to check the
results

<<ex-score>>=
J <- 5
N <- 4

S <- crossprod(matrix(runif(J^2), nrow = J))
prm <- t(chol(S))[lower.tri(S, diag = TRUE)]

### define C
mC <- ltMatrices(matrix(prm, ncol = 1), trans = TRUE, diag = TRUE)

a <- matrix(runif(N * J), nrow = J) - 2
b <- a + 4
a[2,] <- -Inf
b[3,] <- Inf

M <- 10000
W <- matrix(runif(M * (J - 1)), ncol = M)

lli <- c(lmvnorm(a, b, chol = mC, w = W, M = M, logLik = FALSE))

fC <- function(prm) {
    C <- ltMatrices(matrix(prm, ncol = 1), trans = TRUE, diag = TRUE)
    lmvnorm(a, b, chol = C, w = W, M = M)
}

sC <- smvnorm(a, b, chol = mC, w = W, M = M)

chk(lli, sC$logLik)

if (require("numDeriv"))
    print(max(abs(grad(fC, unclass(mC)) - rowSums(unclass(sC$chol)))))
@@

We can do the same when $\mL$ (and not $\mC$) is given
<<ex-Lscore>>=
mL <- solve(mC)

lliL <- c(lmvnorm(a, b, invchol = mL, w = W, M = M, logLik = FALSE))

chk(lli, lliL)

fL <- function(prm) {
    L <- ltMatrices(matrix(prm, ncol = 1), trans = TRUE, diag = TRUE)
    lmvnorm(a, b, invchol = L, w = W, M = M)
}

sL <- smvnorm(a, b, invchol = mL, w = W, M = M)

chk(lliL, sL$logLik)

if (require("numDeriv"))
    print(max(abs(grad(fL, unclass(mL)) - rowSums(unclass(sL$chol)))))
@@

The score function also works for univariate problems
<<ex-uni-score>>=
ptr <- pnorm(b[1,] / c(unclass(mC[,1]))) - pnorm(a[1,] / c(unclass(mC[,1])))
log(ptr)
lmvnorm(a[1,,drop = FALSE], b[1,,drop = FALSE], chol = mC[,1], logLik = FALSE)
lapply(smvnorm(a[1,,drop = FALSE], b[1,,drop = FALSE], chol = mC[,1], logLik =
TRUE), unclass)
sd1 <- c(unclass(mC[,1]))
(dnorm(b[1,] / sd1) * b[1,] - dnorm(a[1,] / sd1) * a[1,]) * (-1) / sd1^2 / ptr
@@

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

We now represent this matrix as \code{ltMatrices} object
<<ex-ML-C>>=
prm <- C[lower.tri(C, diag = TRUE)]
lt <- ltMatrices(matrix(prm, ncol = 1L), 
                 diag = TRUE,    ### has diagonal elements
                 byrow = FALSE,  ### prm is column-major
                 trans = TRUE)   ### store as J * (J + 1) / 2 x 1
BYROW <- FALSE   ### later checks
lt <- ltMatrices(lt, 
                 byrow = BYROW,   ### convert to row-major
                 trans = TRUE)   ### keep dimensions
chk(C, as.array(lt)[,,1], check.attributes = FALSE)
chk(Sigma, as.array(Tcrossprod(lt))[,,1], check.attributes = FALSE)
@@

We generate some data from $\N_\J(\mathbf{0}_\J, \Sigma)$ by first sampling
from $\rZ \sim \N_\J(\mathbf{0}_\J, \mI_\J)$ and then computing $\rY = \mC \rZ +
\muvec \sim \N_\J(\muvec, \mC \mC^\top)$

<<ex-ML-data>>=
N <- 100
Z <- matrix(rnorm(N * J), nrow = J)
Y <- Mult(lt, Z) + (mn <- 1:J)
@@

Next we add some interval-censoring represented by \code{lwr} and \code{upr}. 

<<ex-ML-cens>>=
prb <- 1:9 / 10
sds <- sqrt(diag(Sigma))
ct <- sapply(1:J, function(j) qnorm(prb, mean = mn[j], sd = sds[j])) 
lwr <- upr <- Y
for (j in 1:J) {
    f <- cut(Y[j,], breaks = c(-Inf, ct[,j], Inf))
    lwr[j,] <- c(-Inf, ct[,j])[f]
    upr[j,] <- c(ct[,j], Inf)[f]
}
@@

The true mean $\muvec$ and the true covariance matrix $\Sigma$ can be estimated from the uncensored data as

<<ex-ML-mu-vcov>>=
rowMeans(Y)
(Shat <- var(t(Y)))
@@

Let's do some sanity and performance checks first. For different values of
$M$, we evaluate the log-likelihood using \code{pmvnorm} (called in
\code{lmvnormR}) and the simplified implementation (fast and slow). The comparion is a bit
unfair, because we do not add the time needed to setup Halton sequences, but
we would do this only once and use the stored values for repeated
evaluations of a log-likelihood (because the optimiser expects a
deterministic function to be optimised)

<<ex-ML-chk, eval = TRUE>>=
M <- floor(exp(0:25/10) * 1000)
lGB <- sapply(M, function(m) {
    st <- system.time(ret <- lmvnormR(lwr, upr, mean = mn, chol = lt, algorithm = 
                                      GenzBretz(maxpts = m, abseps = 0, releps = 0)))
    return(c(st["user.self"], ll = ret))
})
lH <- sapply(M, function(m) {
    W <- NULL
    if (require("qrng"))
        W <- t(ghalton(m * N, d = J - 1))
    st <- system.time(ret <- lmvnorm(lwr, upr, mean = mn, chol = lt, w = W, M = m))
    return(c(st["user.self"], ll = ret))
})
lHf <- sapply(M, function(m) {
    W <- NULL
    if (require("qrng"))
        W <- t(ghalton(m * N, d = J - 1))
    st <- system.time(ret <- lmvnorm(lwr, upr, mean = mn, chol = lt, w = W, M = m, 
                                     fast = TRUE))
    return(c(st["user.self"], ll = ret))
})
@@
The evaluated log-likelihoods and corresponding timings are given in
Figure~\ref{lleval}. It seems that for $M \ge 3000$, results are reasonably
stable.

\begin{figure}
\begin{center}
<<ex-ML-fig, eval = TRUE, echo = FALSE, fig = TRUE, pdf = TRUE, width = 8, height = 5>>=
layout(matrix(1:2, nrow = 1))
plot(M, lGB["ll",], ylim = range(c(lGB["ll",], lH["ll",], lHf["ll",])), ylab = "Log-likelihood")
points(M, lH["ll",], pch = 4)
points(M, lHf["ll",], pch = 5)
plot(M, lGB["user.self",], ylim = c(0, max(lGB["user.self",])), ylab = "Time (in sec)")
points(M, lH["user.self",], pch = 4)
points(M, lHf["user.self",], pch = 5)
legend("bottomright", legend = c("pmvnorm", "lmvnorm", "lmvnorm(fast)"), pch = c(1, 4, 5), bty = "n")
@@
\caption{Evaluated log-likelihoods (left) and timings (right).
\label{lleval}}
\end{center}
\end{figure}

We now define the log-likelihood function. It is important to use weights
via the \code{w} argument (or to set the \code{seed}) such that only the
candidate parameters \code{parm} change with repeated calls to \code{ll}. We
use an extremely low number of integration points \code{M}, let's see if
this still works out.

<<ex-ML-ll, eval = TRUE>>=
M <- 500 
if (require("qrng")) {
    ### quasi-Monte-Carlo
    W <- t(ghalton(M * N, d = J - 1))
} else {
    ### Monte-Carlo
    W <- matrix(runif(M * N * (J - 1)), ncol = M)
}
ll <- function(parm, J) {
     m <- parm[1:J]		### mean parameters
     parm <- parm[-(1:J)]	### chol parameters
     C <- matrix(c(parm), ncol = 1L)
     C <- ltMatrices(C, diag = TRUE, byrow = BYROW, trans = TRUE)
     -lmvnorm(lower = lwr, upper = upr, mean = m, chol = C, w = W, M = M, logLik = TRUE)
}
@@

We can check the correctness of our log-likelihood function
<<ex-ML-check>>=
prm <- c(mn, unclass(lt))
ll(prm, J = J)
lmvnormR(lwr, upr, mean = mn, chol = lt, 
         algorithm = GenzBretz(maxpts = M, abseps = 0, releps = 0))
(llprm <- lmvnorm(lwr, upr, mean = mn, chol = lt, w = W, M = M))
chk(llprm, sum(lmvnorm(lwr, upr, mean = mn, chol = lt, w = W, M = M, logLik = FALSE)))
@@

Before we hand over to the optimiser, we define the score function with
respect to $\muvec$ and $\mC$

<<ex-ML-sc>>=
sc <- function(parm, J) {
    m <- parm[1:J]             ### mean parameters
    parm <- parm[-(1:J)]       ### chol parameters
    C <- matrix(c(parm), ncol = 1L)
    C <- ltMatrices(C, diag = TRUE, byrow = BYROW, trans = TRUE)
    ret <- smvnorm(lower = lwr, upper = upr, mean = m, chol = C, 
                   w = W, M = M, logLik = TRUE)
    return(-c(rowSums(ret$mean), rowSums(unclass(ret$chol))))
}

if (require("numDeriv"))
    print(abs(max(grad(ll, prm, J = J) - sc(prm, J = J))))
### TODO: using a different seed gives much better agreement
@@


Finally, we can hand-over to \code{optim}. Because we need $\text{diag}(\mC) >
0$, we use box constraints and \code{method = "L-BFGS-B"}. We start with the
estimates obtained from the original continuous data.

<<ex-ML>>=
llim <- rep(-Inf, J + J * (J + 1) / 2)
llim[J + which(rownames(unclass(lt)) %in% paste(1:J, 1:J, sep = "."))] <- 1e-4

if (BYROW) {
  start <- c(rowMeans(Y), chol(Shat)[upper.tri(Shat, diag = TRUE)])
} else {
  start <- c(rowMeans(Y), t(chol(Shat))[lower.tri(Shat, diag = TRUE)])
}

ll(start, J = J)

op <- optim(start, fn = ll, gr = sc, J = J, method = "L-BFGS-B", 
            lower = llim, control = list(trace = TRUE))

op$value ## compare with 
ll(prm, J = J)
@@

We can now compare the true and estimated Cholesky factor of our covariance
matrix
<<ex-ML-L>>=
(L <- ltMatrices(matrix(op$par[-(1:J)], ncol = 1), 
                 diag = TRUE, byrow = BYROW, trans = TRUE) )
lt
@@
and the estimated means
<<ex-ML-mu>>=
op$par[1:J]
mn
@@

We can also compare the results on the scale of the covariance matrix

<<ex-ML-Shat>>=
Tcrossprod(lt)		### true Sigma
Tcrossprod(L)           ### interval-censored obs
Shat                    ### "exact" obs
@@

This looks reasonably close.

\textbf{Warning:} Do NOT assume the choices made here (especially \code{M}
and \code{W}) to be universally applicable. Make sure to investigate the
accuracy depending on these parameters 
of the log-likelihood and score function in your application.



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
