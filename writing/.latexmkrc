# LaTeXmk configuration for project-wide output directory
$out_dir = 'out';
$aux_dir = 'out';
$pdf_mode = 1;
$pdflatex = 'pdflatex -synctex=1 -interaction=nonstopmode -file-line-error %O %S';
