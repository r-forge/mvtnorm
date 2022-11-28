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
\newcommand{\code}[1]{\texttt{#1()}}

\newcommand{\R}{\mathbb{R} }
\newcommand{\Prob}{\mathbb{P} }
\newcommand{\N}{\mathbb{N} }
\newcommand{\J}{J}
\newcommand{\V}{\mathbb{V}} %% cal{\mbox{\textnormal{Var}}} }
\newcommand{\E}{\mathbb{E}} %%mathcal{\mbox{\textnormal{E}}} }
\newcommand{\yvec}{\mathbf{y}}
\newcommand{\avec}{\mathbf{a}}
\newcommand{\bvec}{\mathbf{b}}
\newcommand{\rY}{\mathbf{Y}}
\newcommand{\mC}{\mathbf{C}}
\newcommand{\argmin}{\operatorname{argmin}\displaylimits}
\newcommand{\argmax}{\operatorname{argmax}\displaylimits}


\author{Torsten Hothorn \\ Universit\"at Z\"urich}

\title{Parallel \code{pmvnorm} in the \pkg{mvtnorm} Package}

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


@d pmvnorm2 prototype
@{pmvnorm2(lower, upper, mean, chol, control, ...)@}

\chapter{\proglang{R} Code}


\section{Variables}


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
