
lmvnorm_src.Rnw:	lmvnorm_src.w
	nuweb -r lmvnorm_src.w
	mv lmvnorm_src.tex lmvnorm_src.Rnw
	cp ltMatrices.R R
	cp ltMatrices.c src
	cp lmvnorm.c src
	cp lmvnorm.R R

lmvnorm_src.R: lmvnorm_src.Rnw
	Stangle lmvnorm_src.Rnw

lmvnorm_src.tex:	lmvnorm_src.Rnw
	R CMD Sweave lmvnorm_src.Rnw
	
lmvnorm_src.pdf:	lmvnorm_src.tex
	texi2dvi --pdf lmvnorm_src.tex
	### NEED to run nuweb TWICE to
	### sort-out page numbers
	nuweb -r lmvnorm_src.w
	mv lmvnorm_src.tex lmvnorm_src.Rnw
	R CMD Sweave lmvnorm_src.Rnw
	pdflatex lmvnorm_src.tex
	pdflatex lmvnorm_src.tex
	cp lmvnorm_src.Rnw vignettes

all:	lmvnorm_src.pdf

clean:	
	./cleanup
	rm -rf *~
	rm -rf lmvnorm_src*.pdf
	rm -rf lmvnorm_src.R
	texclean lmvnorm_src
	rm -rf lmvnorm_src.tex
	rm -rf lmvnorm_src.aux
	rm -rf lmvnorm_src.log
	rm -rf lmvnorm_src.bbl
	rm -rf lmvnorm_src.blg
	rm -rf lmvnorm_src.out
	rm -rf lmvnorm_src.toc
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
#	rm -rf vignettes/lmvnorm_src.Rnw
#	rm -rf tests/lmvnorm_src.R*
	