PRE = figures

all: $(PRE).pdf

$(PRE).pdf: $(PRE).tex
	pdflatex -shell-escape $(PRE).tex

clean:
	rm -f *.log *.aux figures.pdf *.dvi
