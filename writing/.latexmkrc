# LaTeXmk configuration for project-wide output directory
$out_dir = 'out';
$aux_dir = 'out';
$pdf_mode = 1;
$pdflatex = 'pdflatex -synctex=1 -interaction=nonstopmode -file-line-error %O %S';

use File::Basename qw(fileparse);

# Regenerate glossaries/acronyms automatically when .glo/.acn changes.
add_cus_dep('glo', 'gls', 0, 'run_makeglossaries');
add_cus_dep('acn', 'acr', 0, 'run_makeglossaries');

sub run_makeglossaries {
	my ($name) = fileparse($_[0], qr/\.[^.]*$/);
	my $dir = $aux_dir || '.';
	return system('makeglossaries', '-d', $dir, $name);
}
