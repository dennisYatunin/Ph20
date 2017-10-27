TEX = pdflatex -shell-escape -interaction=nonstopmode -file-line-error
MAKEPLOT = python -W ignore diffeq.py
PLOTNAMES = \
	explicit_euler explicit_euler_errors explicit_euler_energy \
	explicit_euler_max_errors implicit_euler implicit_euler_errors \
	implicit_euler_energy implicit_euler_max_errors phase_space \
	symplectic_euler symplectic_euler_energy
IMAGES := $(foreach plotname, $(PLOTNAMES), images/$(plotname).png)

all: Report.pdf

view:
	open Report.pdf

clean:
	rm -rf Report.pdf main.aux main.log images

images/%.png: diffeq.py
	mkdir -p images
	$(MAKEPLOT) $*

Report.pdf: settings.sty main.tex $(IMAGES)
	$(TEX) main.tex
	$(TEX) main.tex
	mv main.pdf Report.pdf
