
mvtnorm.Rnw:	mvtnorm.w
	nuweb -r mvtnorm.w
	R CMD SHLIB ltMatrices.c
	R CMD SHLIB pMVN.c
	mv mvtnorm.tex mvtnorm.Rnw
#	mv *.h src
#	mv *.c src
#	mv Makevars src
#	mv *def src
#	mv *.R R
#	mv *.Rd man

mvtnorm.R: mvtnorm.Rnw
	Stangle mvtnorm.Rnw
#	mv mvtnorm.R tests

mvtnorm.tex:	mvtnorm.Rnw
	R CMD Sweave mvtnorm.Rnw
	
mvtnorm.pdf:	mvtnorm.tex
	texi2dvi --pdf mvtnorm.tex
	### NEED to run nuweb TWICE to
	### sort-out page numbers
	nuweb -r mvtnorm.w
	R CMD SHLIB ltMatrices.c
	R CMD SHLIB pMVN.c
	mv mvtnorm.tex mvtnorm.Rnw
	R CMD Sweave mvtnorm.Rnw
	pdflatex mvtnorm.tex
	pdflatex mvtnorm.tex
	# cp mvtnorm.Rnw vignettes

all:	mvtnorm.pdf

clean:	
	./cleanup
	rm -rf *~
	rm -rf mvtnorm*.pdf
	rm -rf mvtnorm.R
	texclean mvtnorm
	rm -rf mvtnorm.tex
	rm -rf mvtnorm.Rnw
	rm -rf *.c
	rm -rf *.h
	rm -rf *.o
	rm -rf *.so	
	rm -rf *.def
	rm -rf *.Rd
	rm -rf *.R
	rm -rf nw*
	rm -rf *bbl
	rm -rf *brf
	rm -rf *wk

#distclean: clean
#	rm -rf src/*
#	rm -rf R/*
#	rm -rf man/*
#	rm -rf vignettes/mvtnorm.Rnw
#	rm -rf tests/mvtnorm.R*
	