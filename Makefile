tex:
	# pdflatex main.tex
	xelatex main.tex

clean:
	rm -rf \
		*.aux \
		*.fdb_latexmk \
		*.log \
		*.out \
		*.fls \
		*.synctex.gz \
		*.pdf
