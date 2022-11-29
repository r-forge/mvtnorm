\documentclass[a4paper]{report}

%% packages
\usepackage{amsfonts,amstext,amsmath,amssymb,amsthm}

%\VignetteIndexEntry{Parallel pmvnorm()}
%\VignetteDepends{mvtnorm}
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

\title{Vectorised \cmd{pmvnorm} in the \pkg{mvtnorm} Package}

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
\begin{eqnarray*}
p_i(\mC_i \mid \avec_i, \bvec_i) = \Prob(\avec_i < \rY_i \le \bvec_i \mid \mC_i ) 
  = (2 \pi)^{-\frac{\J}{2}} \text{det}(\mC_i)^{-\frac{1}{2}} 
    \int_{\avec_i}^{\bvec_i} \exp\left(-\frac{1}{2} \yvec^\top \mC_i^{-\top} \mC_i^{-1} \yvec\right) \, d \yvec
\end{eqnarray*}
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


\section{Algorithm}

For each $i = 1, \dots, N$, do

\begin{enumerate}
  \item Input $\mC_i$, $\avec_i$, $\bvec_i$, and control parameters $\alpha$, $\epsilon$, and $M_\text{max}$.

  \item Standardize integration limits $a^{(i)}_j / c^{(i)}_{jj}$, $b^{(i)}_j / c^{(i)}_{jj}$, and rows $c^{(i)}_{j\jmath} / c^{(i)}_{jj}$ for $1 \le \jmath < j < \J$.

  \item Initialize $\text{intsum} = \text{varsum} = 0$, $M = 0$, $d_1 =
\Phi\left(a^{(i)}_1\right)$, $e_1 = \Phi\left(b^{(i)}_1\right)$ and $f_1 = e_1 - d_1$.

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

      \item Set $\text{intsum} = \text{intsum} + f_\J$, $\text{varsum} = \text{varsum} + f^2_\J$, $M = M + 1$, 
            and $\text{error} = \sqrt{(\text{varsum}/M - (\text{intsum}/M)^2) / M}$.
    
      \item[Until] $\text{error} < \epsilon$ or $M = M_\text{max}$

    \end{enumerate}
  \item Output $\hat{p}_i = \text{intsum} / M$.

\end{enumerate}

\chapter{Lower Triangular Matrices}

@o ltmatrices.R -cp
@{
@<ltmatrices@>
@<dim.ltmatrices@>
@<print ltmatrices@>
@<transpose ltmatrices@>
@<reorder ltmatrices@>
@<subset ltmatrices@>
@<diagonals ltmatrices@>
@<mult ltmatrices@>
@<solve ltmatrices@>
@<tcrossprod ltmatrices@>
@}

We first need infrastructure for dealing with multiple lower triangular matrices
$\mC_i \in \R^{\J \times \J}$ for $i = 1, \dots, N$. We note that each such matrix
$\mC$ can be stored in a vector of length $\J (\J + 1) / 2$. If all
diagonal elements are one $c^{(i)}_{jj} \equiv 1, j = 1, \dots, \J$, the
length of this vector is $\J (\J - 1) / 2$.

\section{Multiple lower triangular matrices}

Therefore, we can store $N$ such matrices in an $N \times \J (\J + 1) / 2$  (\code{diag = TRUE})
or $N \times \J (\J - 1) / 2$ matrix (\code{diag = FALSE}). Sometimes it is
more convenient to store the transposed $\J (\J + 1) / 2 \times N$ matrix
(\code{trans = TRUE}).

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
@d ltmatrices dim
@{
    J <- floor((1 + sqrt(1 + 4 * 2 * ifelse(trans, nrow(object), ncol(object)))) / 2 - diag)
    stopifnot(ifelse(trans, nrow(object), ncol(object)) == J * (J - 1) / 2 + diag * J)
@}

Typically the $\J$ dimensions are associated with names, and we therefore
compute identifiers for the vector elements in either column- or row-major
order (for later printing)

@d ltmatrices names
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

@d ltmatrices input
@{
if (inherits(object, "ltmatrices")) {
    ret <- .reorder(object, byrow = byrow)
    ret <- .transpose(object, trans = trans)
    return(ret)
}
@}

The constructor essentially attaches attributes to a matrix \code{object}

@d ltmatrices
@{
ltmatrices <- function(object, diag = FALSE, byrow = FALSE, trans = FALSE, names = TRUE) {

    if (!is.matrix(object) && trans) 
        object <- matrix(object, ncol = 1L)
    if (!is.matrix(object) && !trans) 
        object <- matrix(object, nrow = 1L)

    @<ltmatrices input@>

    @<ltmatrices dim@>
    
    @<ltmatrices names@>

    attr(object, "diag")    <- diag
    attr(object, "byrow")   <- byrow
    attr(object, "trans")   <- trans
    attr(object, "rcnames") <- names

    class(object) <- c("ltmatrices", class(object))
    object
}
@}

The dimensions of such an object are always $N \times \J \times \J$
(regardless or \code{byrow} and \code{trans}, and are given by

@d dim.ltmatrices
@{
dim.ltmatrices <- function(x) {
    J <- length(attr(x, "rcnames"))
    class(x) <- class(x)[-1L]
    return(c(ifelse(attr(x, "trans"), ncol(x), nrow(x)), J, J))
}
dim.symatrices <- dim.ltmatrices
@}

Let's set-up an example for illustration:

<<example>>=
source("ltmatrices.R")
dyn.load("ltmatrices.so")
J <- 4
N <- 3
diag <- TRUE
nm <- LETTERS[1:J]

x <- matrix(1:(N * (J * (J - 1) / 2 + diag * J)), byrow = TRUE, nrow = N)
rownames(x) <- paste0("x", 1:nrow(x))
lx <- ltmatrices(x, diag = TRUE, byrow = TRUE, names = nm)
dim(lx)
unclass(lx)
@@

For pretty printing, we coerse object of class \code{ltmatrices} to
\code{array}. The method has an \code{symmetric} argument forcing the lower
triangular matrix to by interpreted as a symmetric matrix.

@d print ltmatrices
@{
as.array.ltmatrices <- function(x, symmetric = FALSE, ...) {

    diag <- attr(x, "diag")
    byrow <- attr(x, "byrow")
    trans <- attr(x, "trans")
    rcnames <- attr(x, "rcnames")
    class(x) <- class(x)[-1L]
    if (trans) x <- t(x)
    J <- length(rcnames)

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
    dim(ret) <- c(J, J, nrow(x))
    dimnames(ret) <- list(rcnames, rcnames, rownames(x))
    return(ret)
}

as.array.symatrices <- function(x, ...)
    return(as.array.ltmatrices(x, symmetric = TRUE))

print.ltmatrices <- function(x, ...)
    print(as.array(x))

print.symatrices <- function(x, ...)
    print(as.array(x))
@}


<<ex-print>>=
print(lx)
## array subsetting
ax <- as.array(lx)
ax[,,1]
@@

It is sometimes convenient to have access to lower triangular matrices in
either column- or row major order and this little helper function switches
between the two forms.

@d reorder ltmatrices
@{
.reorder <- function(x, byrow = FALSE) {

    stopifnot(inherits(x, "ltmatrices"))
    if (attr(x, "byrow") == byrow) return(x)

    diag <- attr(x, "diag")
    trans <- attr(x, "trans")
    rcnames <- attr(x, "rcnames")
    J <- length(rcnames)
    class(x) <- class(x)[-1L]

    if (trans) {
        rL <- cL <- diag(0, nrow = J)
        rL[lower.tri(rL, diag = diag)] <- cL[upper.tri(cL, diag = diag)] <- 1:nrow(x)
        cL <- t(cL)
        if (attr(x, "byrow")) ### row -> col order
            return(ltmatrices(x[cL[lower.tri(cL, diag = diag)], , drop = FALSE], 
                              diag = diag, byrow = FALSE, trans = TRUE, names = rcnames))
        ### col -> row order
        return(ltmatrices(x[t(rL)[upper.tri(rL, diag = diag)], , drop = FALSE], 
                          diag = diag, byrow = TRUE, trans = TRUE, names = rcnames))
    }

    rL <- cL <- diag(0, nrow = J)
    rL[lower.tri(rL, diag = diag)] <- cL[upper.tri(cL, diag = diag)] <- 1:ncol(x)
    cL <- t(cL)
    if (attr(x, "byrow")) ### row -> col order
        return(ltmatrices(x[, cL[lower.tri(cL, diag = diag)], drop = FALSE], 
                          diag = diag, byrow = FALSE, names = rcnames))
    ### col -> row order
    return(ltmatrices(x[, t(rL)[upper.tri(rL, diag = diag)], drop = FALSE], 
                      diag = diag, byrow = TRUE, names = rcnames))
}
@}

<<ex-reorder>>=
(rx <- ltmatrices(lx, byrow = FALSE))
all.equal(as.array(rx), ax)
@@

The internal representation as $N \times \J (\J + 1) / 2$ matrix to a matrix
of dimensions $\J (\J + 1) / 2 \times N$ can be changed as well (NOTE that
this does not mean the matrix $\mC_i$ is transposed to $\mC_i^\top$!).

@d transpose ltmatrices
@{
.transpose <- function(x, trans = FALSE) {

    stopifnot(inherits(x, "ltmatrices"))
    if (attr(x, "trans") == trans) return(x)

    diag <- attr(x, "diag")
    rcnames <- attr(x, "rcnames")
    byrow <- attr(x, "byrow")
    class(x) <- class(x)[-1L]

    return(ltmatrices(t(x), diag = diag, byrow = byrow, 
                      trans = trans, names = rcnames))
}
@}

<<ex-trans>>=
(tx <- ltmatrices(lx, trans = TRUE))
all.equal(as.array(tx), ax)
@@

We might want to select subsets of observations $i \in \{1, \dots, N\}$ or
rows/columns $j \in \{1, \dots, \J\}$ of the corresponding matrices $\mC_i$

@d subset ltmatrices
@{
"[.ltmatrices" <- function(x, i, j, ..., drop = FALSE) {

    if (drop) warning("argument drop is ignored")
    if (missing(i) && missing(j)) return(x)
    diag <- attr(x, "diag")
    byrow <- attr(x, "byrow")
    trans <- attr(x, "trans")
    rcnames <- attr(x, "rcnames")
    class(x) <- class(x)[-1L]
    J <- length(rcnames)
    if (!missing(j)) {
        if (length(j) == 1L && !diag) {
            if (trans)
                return(ltmatrices(matrix(1, ncol = ncol(x), nrow = 1), diag = TRUE, 
                                  trans = TRUE, names = rcnames[j]))
            return(ltmatrices(matrix(1, nrow = nrow(x), ncol = 1), diag = TRUE, 
                              names = rcnames[j]))
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
                return(ltmatrices(x[c(L), , drop = FALSE], diag = diag, 
                                  trans = TRUE, byrow = byrow, names = rcnames[j]))
            return(ltmatrices(x[, c(L), drop = FALSE], diag = diag, 
                              byrow = byrow, names = rcnames[j]))
        }
        if (trans) 
            return(ltmatrices(x[c(L), i, drop = FALSE], diag = diag, 
                              trans = TRUE, byrow = byrow, names = rcnames[j]))
        return(ltmatrices(x[i, c(L), drop = FALSE], diag = diag, 
                          byrow = byrow, names = rcnames[j]))
    }
    if (trans)
        return(ltmatrices(x[, i, drop = FALSE], diag = diag, 
                          trans = trans, byrow = byrow, names = rcnames))
    return(ltmatrices(x[i, , drop = FALSE], diag = diag, 
                      byrow = byrow, names = rcnames))
}

"[.symatrices" <- function(x, i, j, ..., drop = FALSE) {
    class(x)[1L] <- "ltmatrices"
    ret <- x[i, j, ..., drop = drop]
    class(ret)[1L] <- "symatrices"
    return(ret)
}
@}

<<ex-subset>>=
sx <- lx[c(1, 3), c(2, 4)]
all.equal(as.array(sx), ax[c(2, 4), c(2, 4), c(1, 3)])
@@

The diagonal elements of each matrix $\mC_i$ can be extracted and are
always returned as an $N \times \J$ matrix (also when \code{trans = TRUE}).

@d diagonals ltmatrices
@{
diagonals <- function(x, ...)
    UseMethod("diagonals")

diagonals.ltmatrices <- function(x, ...) {

    rcnames <- attr(x, "rcnames")
    diag <- attr(x, "diag")
    byrow <- attr(x, "byrow")
    trans <- attr(x, "trans")
    diag <- attr(x, "diag")
    J <- length(rcnames)
    class(x) <- class(x)[-1L]

    if (!diag) {
        ret <- matrix(1, nrow = nrow(x), ncol = J)
        rownames(ret) <- rownames(x)
        colnames(ret) <- rcnames
        return(ret)
    } else {
        if (byrow)
            idx <- cumsum(c(1, 2:J))
        else
            idx <- cumsum(c(1, J:2))
        if (trans)
            ret <- t(x[idx, drop = FALSE])
        else
            ret <- x[, idx, drop = FALSE]
        colnames(ret) <- rcnames
        return(ret)
    }
}

diagonals.symatrices <- diagonals.ltmatrices

@}

<<ex-diag>>=
diagonals(lx)
@@

\section{Multiplication}

Multiplications $\mC_i \yvec_i$ with $\yvec_i \in \R^\J$ for $i = 1, \dots,
N$ can be computed with $\code{y}$ being an $N \times \J$ matrix (FIXME?)

@d mult ltmatrices
@{
### L %*% y
.mult <- function(x, y) {

    stopifnot(inherits(x, "ltmatrices"))

    rcnames <- attr(x, "rcnames")
    diag <- attr(x, "diag")
    J <- length(rcnames)
    mx <- ifelse(J > 10, Matrix, matrix)
    x <- .reorder(x, byrow = TRUE)
    x <- .transpose(x, trans = FALSE)
    class(x) <- class(x)[-1L]

    if (!diag) {
        idx <- 1
        S <- 1
        if (J > 2) {
            S <- mx(rep(rep(1:0, (J - 1)), c(rbind(1:(J - 1), ncol(x)))), nrow = ncol(x))[, -J,drop = FALSE]
            idx <- unlist(lapply(colSums(S), seq_len))
        }
    } else {
        S <- mx(rep(rep(1:0, J),
                    c(rbind(1:J, ncol(x)))), nrow = ncol(x))[, -(J + 1), drop = FALSE]
        idx <- unlist(lapply(colSums(S), seq_len))
    }

    if (!diag) {
        A <- y[, idx] * x
        B <- A %*% S + y[, -1L, drop = FALSE]
        ret <- cbind(y[, 1L, drop = FALSE], as(B, "matrix"))
    } else {
        A <- y[, idx] * x
        ret <- as(A %*% S, "matrix")
    }
    colnames(ret) <- rcnames
    rownames(ret) <- rownames(x)
    return(ret)
}
@}

<<ex-mult>>=
y <- matrix(runif(J * N), nrow = N)
(mx <- .mult(lx, y))
m <- matrix(0, nrow = N, ncol = J, dimnames = list(rownames(x), nm))
for (i in 1:N) 
    m[i,] <- ax[,,i] %*% y[i,]
all.equal(mx, m)

@@

\section{Solving linear systems}

Compute $\mC_i^{-1}$ or solve $\mC_i \xvec_i = \yvec_i$ for $\xvec_i$ for
all $i = 1, \dots, N$.

@o ltmatrices.c -cc
@{
#include <R.h>
#include <Rmath.h>
#include <Rinternals.h>
#include <Rdefines.h>
#include <Rconfig.h>
#include <R_ext/Lapack.h> /* for dtptri */
@<solve@>
@<tcrossprod@>
@}

\code{A} is $\mC_i, i = 1, \dots, N$ in transposed column-major order
(matrix of dimension $\J (\J - 1) / 2 + \J \text{diag} \times N$), and
\code{b} is the $\J \times N$ matrix $(\yvec_1 \mid \yvec_2 \mid \dots \mid
\yvec_N)$. This function returns the $\J \times N$ matrix $(\xvec_1 \mid \xvec_2 \mid \dots \mid
\xvec_N)$ of solutions.

If \code{b} is not given, $\mC_i^{-1}$ is returned in transposed
column-major order (matrix of dimension $\J (\J 1 1) / 2  \times N$) ALWAYS
including the diagonals.

@d solve
@{

SEXP R_ltmatrices_solve (SEXP A, SEXP b, SEXP N, SEXP J, SEXP diag)
{

    SEXP ans, ansx;
    double *dA, *db, *dans, *dansx;
    int n, p, info, k, nrow, j, jj, idx, ONE = 1;

    Rboolean Rdiag = asLogical(diag);
    char di, lo = 'L', tr = 'N';
    if (Rdiag) {
        /* non-unit diagonal elements */
        di = 'N';
    } else {
        /* unit diagonal elements */
        di = 'U';
    }

    /* number of matrices */
    int iN = INTEGER(N)[0];
    /* dimension of matrices */
    int iJ = INTEGER(J)[0];

    /* p = J * (J - 1) / 2 + diag * J */
    p = LENGTH(A) / iN;

    /* add diagonal elements (expected by Lapack) */
    nrow = (p + (1 - Rdiag) * iJ);

    @<setup memory@>
    
    /* loop over matrices, ie columns of x */    
    for (int n = 0; n < iN; n++) {

        @<copy elements@>

        @<call Lapack@>

        /* next matrix */
        dans = dans + nrow;
        dA = dA + p;
    }

    @<return objects@>
}
@}

@d setup memory
@{
/* return object: include unit diagnonal elements if Rdiag == 0 */

dA = REAL(A);
PROTECT(ans = allocMatrix(REALSXP, nrow, iN));
dans = REAL(ans);

if (b != R_NilValue) {
    db = REAL(b);
    PROTECT(ansx = allocMatrix(REALSXP, iJ, iN));
    dansx = REAL(ansx);
}
@}

@d copy elements
@{
/* copy data and insert unit diagonal elements when necessary */
jj = 0;
k = 0;
idx = 0;
j = 0;
while(j < p) {
    if (!Rdiag && (jj == idx)) {
        dans[jj] = 1.0;
        idx = idx + (iJ - k);
        k++;
    } else {
        dans[jj] = dA[j];
        j++;
    }
    jj++;
}
if (!Rdiag) dans[idx] = 1.0;

if (b != R_NilValue) {
    for (j = 0; j < iJ; j++)
        dansx[j] = db[j];
}
@}

@d call Lapack
@{
if (b == R_NilValue) {
    /* compute inverse */
    F77_CALL(dtptri)(&lo, &di, &iJ, dans, &info FCONE FCONE);
    if (info != 0)
        error("Cannot solve ltmatices");
} else {
    /* solve linear system */
    F77_CALL(dtpsv)(&lo, &tr, &di, &iJ, dans, dansx, &ONE FCONE FCONE);
    dansx = dansx + iJ;
    db = db + iJ;
}
@}

@d return objects
@{
if (b == R_NilValue) {
    UNPROTECT(1);
    /* note: ans always includes diagonal elements */
    return(ans);
} else {
    UNPROTECT(2);
    return(ansx);
}
@}

@d solve ltmatrices
@{

### inverse of ltmatrices
### returns inverse matrices as ltmatrices in same storage order (missing b)
### or mult(solve(a), b)
solve.ltmatrices <- function(a, b, ...) {

    byrow_orig <- attr(a, "byrow")
    trans_orig <- attr(a, "trans")

    x <- .reorder(a, byrow = FALSE)
    x <- .transpose(x, trans = TRUE)
    diag <- attr(x, "diag")
    rcnames <- attr(x, "rcnames")
    J <- length(rcnames)
    class(x) <- class(x)[-1L]
    storage.mode(x) <- "double"

    if (!missing(b)) {
        if (!is.matrix(b)) b <- matrix(b, nrow = J, ncol = ncol(x))
        stopifnot(ncol(b) == ncol(x))
        stopifnot(nrow(b) == J)
        storage.mode(b) <- "double"
        ret <- .Call("R_ltmatrices_solve", x, b, 
                     as.integer(ncol(x)), as.integer(J), as.logical(diag))
        colnames(ret) <- colnames(x)
        rownames(ret) <- rcnames
        return(ret)
    }

    ret <- try(.Call("R_ltmatrices_solve", x, NULL,
                 as.integer(ncol(x)), as.integer(J), as.logical(diag)))
    colnames(ret) <- colnames(x)

    if (!diag)
        ### ret always includes diagonal elements
        ret <- ret[- cumsum(c(1, J:2)), , drop = FALSE]

    ret <- ltmatrices(ret, diag = diag, byrow = FALSE, trans = TRUE, 
                      names = rcnames)
    ret <- .reorder(ret, byrow = byrow_orig)
    ret <- .transpose(ret, trans = trans_orig)
    return(ret)
}
@}

<<ex-solve>>=
(sx <- solve(lx))
apply(ax, 3, solve, simplify = FALSE)
for (i in 1:N)
    print(round(ax[,,i] %*% as.array(sx)[,,i], 3))

y <- runif(J)
solve(lx, y)
apply(ax, 3, solve, b = y, simplify = TRUE)
@@

\section{Crossproducts}

Compute $\mC_i \mC_i^\top$ or $\text{diag}(\mC_i \mC_i^\top)$
(\code{diag\_only = TRUE}) for $i = 1, \dots, N$. These are symmetric
matrices, so we store them as a lower triangular matrix using a different
class name \code{symatrices}.

@d tcrossprod
@{
#define IDX(i, j, n, d) ((i) >= (j) ? (n) * ((j) - 1) - ((j) - 2) * ((j) - 1)/2 + (i) - (j) - (!d) * (j) : 0)

SEXP R_ltmatrices_tcrossprod (SEXP A, SEXP N, SEXP J, SEXP diag, SEXP diag_only) {

    SEXP ans;
    int iJ = INTEGER(J)[0];
    int iN = INTEGER(N)[0];
    int i, j, k, ix, nrow;
    double *dans, *dA;
    Rboolean Rdiag_only = asLogical(diag_only);
    Rboolean Rdiag = asLogical(diag);

    dA = REAL(A);
    int p = LENGTH(A) / iN;

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
for (int n = 0; n < iN; n++) {
    dans[0] = 1.0;
    if (Rdiag) dans[0] = pow(dA[0], 2);
    for (i = 1; i < iJ; i++) {
        dans[i] = 0.0;
        for (k = 0; k < i; k++)
            dans[i] += pow(dA[IDX(i + 1, k + 1, iJ, Rdiag)], 2);
        if (Rdiag) {
            dans[i] += pow(dA[IDX(i + 1, i + 1, iJ, Rdiag)], 2);
        } else {
            dans[i] += 1.0;
        }
    }
    dans = dans + iJ;
    dA = dA + p;
}
@}

@d tcrossprod full
@{
nrow = iJ * (iJ + 1) / 2;
PROTECT(ans = allocMatrix(REALSXP, nrow, iN)); 
dans = REAL(ans);
for (int n = 0; n < INTEGER(N)[0]; n++) {
    dans[0] = 1.0;
    if (Rdiag) dans[0] = pow(dA[0], 2);
    for (i = 1; i < iJ; i++) {
        for (j = 0; j <= i; j++) {
            ix = IDX(i + 1, j + 1, iJ, 1);
            dans[ix] = 0.0;
            for (k = 0; k < j; k++)
                dans[ix] += 
                    dA[IDX(i + 1, k + 1, iJ, Rdiag)] *
                    dA[IDX(j + 1, k + 1, iJ, Rdiag)];
            if (Rdiag) {
                dans[ix] += 
                    dA[IDX(i + 1, j + 1, iJ, Rdiag)] *
                    dA[IDX(j + 1, j + 1, iJ, Rdiag)];
            } else {
                if (j < i)
                    dans[ix] += dA[IDX(i + 1, j + 1, iJ, Rdiag)];
                else
                    dans[ix] += 1.0;
            }
        }
    }
    dans = dans + nrow;
    dA = dA + p;
}
@}

@d tcrossprod ltmatrices
@{
### L %*% t(L) => returns object of class symatrices
### diag(L %*% t(L)) => returns matrix of diagonal elements
.tcrossprod.ltmatrices <- function(x, diag_only = FALSE) {

    byrow_orig <- attr(x, "byrow")
    trans_orig <- attr(x, "trans")
    rcnames <- attr(x, "rcnames")
    diag <- attr(x, "diag")
    J <- length(rcnames)

    x <- .reorder(x, byrow = FALSE)
    x <- .transpose(x, trans = TRUE)
    class(x) <- class(x)[-1L]
    N <- ncol(x)
    storage.mode(x) <- "double"

    ret <- .Call("R_ltmatrices_tcrossprod", x, as.integer(N), as.integer(J), 
                        as.logical(diag), as.logical(diag_only))
    colnames(ret) <- colnames(x)
    if (diag_only) {
        rownames(ret) <- rcnames
    } else {
        ret <- ltmatrices(ret, diag = TRUE, byrow = FALSE, trans = TRUE, names = rcnames)
        ret <- .reorder(ret, byrow = byrow_orig)
        ret <- .transpose(ret, trans = trans_orig)
        class(ret)[1L] <- "symatrices"
    }
    return(ret)
}
@}

<<ex-tcrossprod>>=
d <- .tcrossprod.ltmatrices(lx, diag_only = TRUE)
a1 <- as.array(t1 <- .tcrossprod.ltmatrices(lx))
diagonals(t1)
d
a2 <- array(unlist(apply(ax, 3, tcrossprod, simplify = FALSE)), dim = dim(ax),
dimnames = dimnames(ax))
all.equal(a1, a2)
@@


\chapter{Prototyping}

@o prototype.R -cp
@{
@<pMVN@>
@<pMVN2@>
@<pMVN3@>
@}

@d pMVN
@{
pMVN <- function(lower, upper, mean = 0, chol, M = 10000, 
                 w = matrix(runif(M * (J - 1)), ncol = M), ...) {

    @<input checks@>

    @<standardise@>

    intsum <- varsum <- numeric(N)
    M <- ncol(w)

    for (k in 1:M) {

         @<inner loop@>

    }

    ret <- intsum / M
    error <- 2.5 * sqrt((varsum / M - (intsum / M)^2) / M)
    attr(ret, "error") <- error
    ret
}
@}

Step 1

@d input checks
@{
stopifnot(inherits(chol, "ltmatrices"))
chol <- ltmatrices(chol, trans = TRUE)
chol <- ltmatrices(chol, byrow = TRUE)
d <- dim(chol)
N <- d[1L]
J <- d[2L]

stopifnot(nrow(lower) == J && ncol(lower) == N)
stopifnot(nrow(upper) == J && ncol(upper) == N)
if (is.matrix(mean))
    stopifnot(nrow(mean) == J && ncol(mean) == N)

lower <- lower - mean
upper <- upper - mean
@}

Step 2

@d standardise
@{
if (attr(chol, "diag")) {
    dchol <- diagonals(chol)
    ac <- lower / t(dchol)
    bc <- upper / t(dchol)
    C <- unclass(chol) / dchol[,rep(1:J, 1:J)]
    C <- ltmatrices(C[, -cumsum(c(1, 2:J))], byrow = TRUE, diag = FALSE)
} else {
    ac <- lower
    bc <- upper
    C <- ltmatrices(chol, byrow = TRUE)
}
uC <- unclass(C)
@}

Step 3 / 4

@d inner loop
@{
d <- pnorm(ac[1,])
e <- pnorm(bc[1,])
f <- e - d
y <- matrix(0, nrow = N, ncol = J - 1)
start <- 1

for (i in 2:J) {
    idx <- start:((start - 1) + (i - 1))
    start <- max(idx) + 1
    tmp <- d + w[i - 1, k] * (e - d)
    tmp <- pmax(.Machine$double.eps, tmp)
    tmp <- pmin(1 - .Machine$double.eps, tmp)
    y[,i - 1] <- qnorm(tmp)
    x <- rowSums(uC[,idx,drop = FALSE] * y[,1:(i - 1)])
    d <- pnorm(ac[i,] - x)
    e <- pnorm(bc[i,] - x)
    f <- (e - d) * f
}
intsum <- intsum + f
varsum <- varsum + f^2
@}

@d pMVN2
@{
pMVN2 <- function(lower, upper, mean = 0, chol, M = 10000, ...) {

    @<input checks@>

    sigma <- .tcrossprod.ltmatrices(chol)

    ret <- error <- numeric(N)
    for (i in 1:N) {
        tmp <- pmvnorm(lower = lower[,i], upper = upper[,i], sigma = as.array(sigma[i,])[,,1])
        ret[i] <- tmp
        error[i] <- attr(tmp, "error")
    }
    attr(ret, "error") <- error
    ret
}
@}

<<ex-pMVN>>=
library("mvtnorm")
source("prototype.R")
source("ltmatrices.R")
J <- 5
N <- 10

x <- matrix(runif(N * J * (J + 1) / 2), ncol = N)
lx <- ltmatrices(x, byrow = TRUE, trans = TRUE, diag = TRUE)

a <- matrix(runif(N * J), nrow = J) - 2
b <- a + 2 + matrix(runif(N * J), nrow = J)

pMVN(a, b, chol = lx, M = 25000)
pMVN2(a, b, chol = lx)
@@

@o pMVN.c -cc
@{
#include <R.h>
#include <Rmath.h>
#include <Rinternals.h>
#include <Rdefines.h>
#include <Rconfig.h>
@<pnorm fast@>
@<R pMVN@>
@}

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

@d R pMVN
@{
SEXP R_pMVN(SEXP a, SEXP b, SEXP C, SEXP N, SEXP J, SEXP W, SEXP M, SEXP tol) {

    SEXP ans;
    double *da, *db, *dC, *dW, *intsum, *varsum, dtol = REAL(tol)[0];
    double mdtol = 1.0 - dtol;
    int iM = INTEGER(M)[0]; 
    int iN = INTEGER(N)[0]; 
    int iJ = INTEGER(J)[0]; 
    int p = LENGTH(C) / iN;
    int start, j, k;
    double tmp, e, d, f, emd, x, y[iJ - 1];

    PROTECT(ans = allocMatrix(REALSXP, iN, 2));
    intsum = REAL(ans);
    varsum = intsum + iN;
    intsum[0] = 0.0;
    varsum[0] = 0.0;

    dW = REAL(W);

    for (int m = 0; m < iM; m++) {
        da = REAL(a);
        db = REAL(b);
        dC = REAL(C);

        for (int i = 0; i < iN; i++) {
            @<C inner loop@>
            intsum[i] += f;
            varsum[i] += pow(f, 2);
            da = da + iJ;
            db = db + iJ;
            dC = dC + p;
        }
        dW = dW + (iJ - 1);
    }

    UNPROTECT(1);
    return(ans);
}
@}

@d C inner loop
@{
d = C_pnorm_fast(da[0], 0.0);
e = C_pnorm_fast(db[0], 0.0);
emd = e - d;
f = emd;
start = 0;
for (j = 1; j < iJ; j++) {
    tmp = d + dW[j - 1] * emd;
    if (tmp < dtol) tmp = dtol;
    if (tmp > mdtol) tmp = mdtol;
    y[j - 1] = qnorm(tmp, 0.0, 1.0, 1, 0);
    x = 0.0;
    for (k = 0; k < j; k++)
        x += dC[start + k] * y[k];
    start += j;
    d = C_pnorm_fast(da[j], x);
    e = C_pnorm_fast(db[j], x);
    emd = e - d;
    f = emd * f;
}
@}

@d pMVN3
@{
pMVN3 <- function(lower, upper, mean = 0, chol, M = 10000, 
                  w = matrix(runif(M * (J - 1)), ncol = M), ...) {

    @<input checks@>

    @<standardise@>

    ret <- .Call("R_pMVN", ac, bc, t(unclass(C)), N, J, w, ncol(w), .Machine$double.eps);
    intsum <- ret[,1]
    varsum <- ret[,2]

    M <- ncol(w)
    ret <- intsum / M
    error <- 2.5 * sqrt((varsum / M - ret^2) / M)
    attr(ret, "error") <- error
    ret
}
@}


<<ex-pMVN3>>= 
dyn.load("pMVN.so")

M <- 10000
W <- matrix(runif(M * (J - 1)), ncol = M)
system.time(p1 <- pMVN(a, b, chol = lx, w = W))
system.time(p2 <- pMVN2(a, b, chol = lx,  w = W))
system.time(p3 <- pMVN3(a, b, chol = lx,  w = W))
all.equal(p1, p2)
all.equal(p3, p2)

cbind(p1, p2, p3)

@@

\chapter{Maximum-likelihood Example}

<<ex-ML>>=
N <- 1000
J <- 3
L <- matrix(prm <- runif(J * (J + 1) / 2), nrow = J * (J + 1) / 2, ncol = N)
lx <- ltmatrices(L, diag = TRUE, byrow = FALSE, trans = TRUE)
Z <- matrix(rnorm(N * J), ncol = J)

Y <- .mult(lx, Z)

var(Y)
.tcrossprod.ltmatrices(lx[1,])

a <- t(Y - runif(N * J, max = .1))
b <- t(Y + runif(N * J, max = .1))

M <- 2000
W <- matrix(runif(M * (J - 1)), ncol = M)

ll <- function(parm) {

     C <- matrix(parm, nrow = J * (J + 1) / 2, ncol = N)
     C <- ltmatrices(C, diag = TRUE, byrow = FALSE, trans = TRUE)
     -sum(log(pMVN3(lower = a, upper = b, chol = C, w = W)))
}

ll(prm)

#optim(rep(0, J * (J + 1) / 2), fn = ll)
@@

\chapter{Package Infrastructure}


\chapter*{Index}

\section*{Files}

@f

\section*{Fragments}

@m

\section*{Identifiers}

@u

\bibliographystyle{plainnat}
\bibliography{\Sexpr{gsub("\\.bib", "", system.file("litdb.bib", package = "mvtnorm"))}}

\end{document}
