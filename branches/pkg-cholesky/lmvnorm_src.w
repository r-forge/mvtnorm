\documentclass[a4paper]{report}

%% packages
\usepackage{amsfonts,amstext,amsmath,amssymb,amsthm}

%\VignetteIndexEntry{Multivariate Normal Log-likelihoods}
%\VignetteDepends{mvtnorm,randtoolbox}
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
\newcommand{\mC}{\mathbf{C}}
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
Copyright (C) 2022 Torsten Hothorn \\

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

This document describes an implementation of \cite{numerical-:1992} for the  evaluation of
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


\chapter{Lower Triangular Matrices}

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
 c_{21}\ & c_{22} \\
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
disables all functionality except printing).

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

The names identifying rows and columns in $\mC$ are

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


It is sometimes convenient to have access to lower triangular matrices in
either column- or row major order and this little helper function switches
between the two forms.

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
this does not mean the matrix $\mC_i$ is transposed to $\mC_i^\top$!).

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
        if (byrow) {
            L[upper.tri(L, diag = diag)] <- 1:ncol(x)
            L <- L[j, j, drop = FALSE]
            L <- L[upper.tri(L, diag = diag)]
        } else {
            L[lower.tri(L, diag = diag)] <- 1:ncol(x)
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
@@

The diagonal elements of each matrix $\mC_i$ can be extracted and are
always returned as an $\J \times N$ matrix (regardless of \code{trans}).
The reason is that \code{ltMatrices} with \code{trans = TRUE} can be
standardized elementwise without transposing objects.

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

    ret <- .Call("R_ltMatrices_Mult", x, y, as.integer(N), 
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
int p, len = iJ * (iJ - 1) / 2 + Rdiag * iJ;
if (LENGTH(C) == len)
    /* C is constant for i = 1, ..., N */
    p = 0;
else 
    /* C contains C_1, ...., C_N */
    p = len;
int i, j;
@}

@d mult
@{
SEXP R_ltMatrices_Mult (SEXP C, SEXP y, SEXP N, SEXP J, SEXP diag) {

    SEXP ans;
    double *dans, *dy = REAL(y);
    int k, start;

    @<RC input@>

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

@d solve
@{
SEXP R_ltMatrices_solve (SEXP C, SEXP y, SEXP N, SEXP J, SEXP diag)
{

    SEXP ans, ansx;
    double *dans, *dansx, *dy;
    int k, info, nrow, ncol, jj, idx, ONE = 1;

    @<RC input@>

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
    for (int i = 0; i < iN; i++) {

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

@d setup memory
@{
/* return object: include unit diagnonal elements if Rdiag == 0 */

/* add diagonal elements (expected by Lapack) */
nrow = (Rdiag ? len : len + iJ);
ncol = (p > 0 ? iN : 1);
PROTECT(ans = allocMatrix(REALSXP, nrow, ncol));
dans = REAL(ans);

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
    F77_CALL(dtpsv)(&lo, &tr, &di, &iJ, dans, dansx, &ONE FCONE FCONE);
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
        ret <- .Call("R_ltMatrices_solve", x, b, 
                     as.integer(N), as.integer(J), as.logical(diag))
        if (d[1L] == N) {
            colnames(ret) <- dn[[1L]]
        } else {
            colnames(ret) <- colnames(b)
        }
        rownames(ret) <- dn[[2L]]
        return(ret)
    }

    ret <- try(.Call("R_ltMatrices_solve", x, NULL,
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

@d tcrossprod
@{
#define IDX(i, j, n, d) ((i) >= (j) ? (n) * ((j) - 1) - ((j) - 2) * ((j) - 1)/2 + (i) - (j) - (!d) * (j) : 0)

SEXP R_ltMatrices_tcrossprod (SEXP C, SEXP N, SEXP J, SEXP diag, SEXP diag_only) {

    SEXP ans;
    double *dans;
    int n, k, ix, nrow;

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

    ret <- .Call("R_ltMatrices_tcrossprod", x, as.integer(N), as.integer(J), 
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

<<ex-tcrossprod>>=
## tcrossprod
a <- as.array(Tcrossprod(lxn))
b <- array(apply(as.array(lxn), 3L, function(x) tcrossprod(x), simplify = TRUE), dim = rev(dim(lxn)))
chk(a, b, check.attributes = FALSE)

# diagonal elements only
d <- Tcrossprod(lxn, diag_only = TRUE)
chk(d, apply(a, 3, diag))
chk(d, diagonals(Tcrossprod(lxn)))

a <- as.array(Tcrossprod(lxd))
b <- array(apply(as.array(lxd), 3L, function(x) tcrossprod(x), simplify = TRUE), dim = rev(dim(lxd)))
chk(a, b, check.attributes = FALSE)

# diagonal elements only
d <- Tcrossprod(lxd, diag_only = TRUE)
chk(d, apply(a, 3, diag))
chk(d, diagonals(Tcrossprod(lxd)))
@@


\chapter{Multivariate Normal Log-likelihoods}

We now discuss code for evaluating the log-likelihood
\begin{eqnarray*}
\sum_{i = 1}^N \log(p_i(\mC_i \mid \avec_i, \bvec_i))
\end{eqnarray*}

This is relatively simple to achieve using \code{pmvnorm}:

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
Repeated calls to \code{FORTRAN} also cost some time. The code implements a
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

lmvnormR(a, b, chol = lx, logLik = FALSE)
@@


\section{Algorithm}

@o lmvnorm.R -cp
@{
@<R Header@>
@<lmvnorm@>
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
@}

We implement the algorithm described by \cite{numerical-:1992}. The key
point here is that the original $\J$-dimensional problem~(\ref{pmvnorm}) is transformed into
an integral over $[0, 1]^{\J - 1}$.

For each $i = 1, \dots, N$, do

\begin{enumerate}
  \item Input $\mC_i$, $\avec_i$, $\bvec_i$, and control parameters $\alpha$, $\epsilon$, and $M_\text{max}$.

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
    C <- ltMatrices(C[-cumsum(c(1, 2:J)), ], byrow = TRUE, trans = TRUE, diag = FALSE)
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
intsum = 0.0;
@}

  \item Repeat

    \begin{enumerate}

      \item Generate uniform $w_1, \dots, w_{\J - 1} \in [0, 1]$.

      \item For $j = 2, \dots, J$ set 
        \begin{eqnarray*}
            y_{j - 1} & = & \Phi^{-1}\left(d_{j - 1} + w_{j - 1} (e_{j - 1} - d_{j - 1})\right) \\
            x_{j - 1} & = & \sum_{\jmath = 1}^{j - 1} c^{(i)}_{j\jmath} y_j \\
            d_j & = & \Phi\left(a^{(i)}_j - x_{j - 1}\right) \\
            e_j & = & \Phi\left(b^{(i)}_j - x_{j - 1}\right) \\
            f_j & = & (e_j - d_j) f_{j - 1}.
       \end{eqnarray*}


@d inner loop
@{
for (j = 1; j < iJ; j++) {
    if (W == R_NilValue) {
        tmp = d + unif_rand() * emd;
    } else {
        tmp = d + dW[j - 1] * emd;
    }

    if (tmp < dtol) {
        y[j - 1] = q0;
    } else {
        if (tmp > mdtol)
            y[j - 1] = -q0;
        else
            y[j - 1] = qnorm(tmp, 0.0, 1.0, 1L, 0L);
    }
    x = 0.0;
    for (k = 0; k < j; k++)
        x += dC[start + k] * y[k];
    start += j;
    d = C_pnorm_fast(da[j], x);
    e = C_pnorm_fast(db[j], x);
    emd = e - d;
    f *= emd;
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
dans[0] = (intsum < dtol ? l0 : log(intsum)) - lM;
if (!RlogLik)
    dans += 1L;
@}

\end{enumerate}

It turned out that calls to \code{pnorm} are expensive, so a slightly faster
alternative is used

@d pnorm fast
@{
/* see https://ssrn.com/abstract=2842681 */
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

    int iM = INTEGER(M)[0]; 
    int iN = INTEGER(N)[0]; 
    int iJ = INTEGER(J)[0]; 

    if (LENGTH(C) == iJ * (iJ - 1) / 2)
        p = 0;
    else 
        p = LENGTH(C) / iN;

    int start, j, k;
    double tmp, e, d, f, emd, x, y[iJ - 1];

    len = (RlogLik ? 1 : iN);
    PROTECT(ans = allocVector(REALSXP, len));
    dans = REAL(ans);

    da = REAL(a);
    db = REAL(b);
    dC = REAL(C);

    q0 = qnorm(dtol, 0.0, 1.0, 1L, 0L);
    l0 = log(dtol);
    lM = log((double) iM);

    if (W == R_NilValue)
        GetRNGstate();

    for (int i = 0; i < iN; i++) {

        @<initialisation@>

        if (W != R_NilValue)
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
\proglang{C}

@d lmvnorm
@{
lmvnorm <- function(lower, upper, mean = 0, chol, logLik = TRUE, M = 25000, 
                    w = NULL, seed = NULL) {

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

    @<input checks@>

    @<standardise@>

    if (!is.null(w)) {
        stopifnot(is.matrix(w))
        stopifnot(nrow(w) == J - 1)
        M <- ncol(w)
        storage.mode(w) <- "double"
    }

    ret <- .Call("R_lmvnorm", ac, bc, unclass(C), as.integer(N), 
                 as.integer(J), w, as.integer(M), .Machine$double.eps, as.logical(logLik));
    return(ret)
}
@}


<<ex-lmvnorm>>= )
M <- 10000
set.seed(29)

if (require("randtoolbox")) {
    ### quasi-Monte-Carlo
    W <- t(halton(M, dim = J - 1))
} else {
    ### Monte-Carlo
    W <- matrix(runif(M * (J - 1)), ncol = M)
}

### Genz & Bretz, 2001, without early stopping
pGB <- lmvnormR(a, b, chol = lx, logLik = FALSE, 
                algorithm = GenzBretz(maxpts = M, abseps = 0, releps = 0))
### Genz 1992 with quasi-Monte-Carlo
pGq <- exp(lmvnorm(a, b, chol = lx, w = W, logLik = FALSE))
### Genz 1992, original Monte-Carlo
pG <- exp(lmvnorm(a, b, chol = lx, w = NULL, M = M, logLik = FALSE))

cbind(pGB, pGq, pG)
@@

The three versions agree nicely.

\chapter{Maximum-likelihood Example}

We now discuss how this infrastructure can be used to estimate the Cholesky
factor of a multivariate normal in the presence of interval-censored
observations.

We first generate data, where \code{prm} are the true parameters
<<ex-ML-data>>=
N <- 250
J <- 4
L <- matrix(prm <- runif(J * (J + 1) / 2), ncol = 1L)
lx <- ltMatrices(L, diag = TRUE, byrow = TRUE, trans = TRUE)
Z <- matrix(rnorm(N * J), nrow = J)
Y <- Mult(lx, Z)
Y <- Y - rowMeans(Y)
a <- Y - runif(N * J, max = .1)
b <- Y + runif(N * J, max = .1)
@@

The interval-censoring is represented by \code{a} and \code{b}. The true
covariance matrix can be estimate from the uncensored data as

<<ex-ML-vcov>>=
(S <- var(t(Y)))
Tcrossprod(lx[1,])
lhat <- chol(S)[upper.tri(S, diag = TRUE)]
@@

We now define the log-likelihood function. It is important to use weights
via the \code{w} argument (or to set the \code{seed}) such that only the
candidate parameters \code{parm} change with repeated calls to \code{ll}.

<<ex-ML-ll, eval = TRUE>>=
M <- 500 ### faster for vignette
if (require("randtoolbox")) {
    ### quasi-Monte-Carlo
    W <- t(halton(M, dim = J - 1))
} else {
    ### Monte-Carlo
    W <- matrix(runif(M * (J - 1)), ncol = M)
}

ll <- function(parm) {

     C <- matrix(parm, ncol = 1L)
     C <- ltMatrices(C, diag = TRUE, byrow = TRUE, trans = TRUE)
     -lmvnorm(lower = a, upper = b, chol = C, w = W, logLik = TRUE)
}
@@

We can check the correctness of our log-likelihood function
<<ex-ML-check>>=
ll(prm)
lmvnormR(a, b, chol = lx, algorithm = GenzBretz(maxpts = M, abseps = 0, releps = 0))
lmvnorm(a, b, chol = lx, w = W)
@@

Finally, we can hand-over to \code{optim} for the unconstrained optimisation
and compare the estimates with the true values and the estimates obtained
from the uncensored observations.

<<ex-ML>>=
op <- optim(lhat, fn = ll)
op$value ## compare with 
ll(prm)
cbind(true = prm, est_int = op$par, est_raw = lhat)
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
