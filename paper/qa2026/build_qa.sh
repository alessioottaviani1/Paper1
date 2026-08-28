#!/usr/bin/env bash
set -e
pdflatex -interaction=nonstopmode -halt-on-error main_qa.tex
bibtex main_qa
pdflatex -interaction=nonstopmode -halt-on-error main_qa.tex
pdflatex -interaction=nonstopmode -halt-on-error main_qa.tex
if pdfinfo main_qa.pdf | grep -qiE "ottaviani|edhec|rebonato"; then
  echo "FAIL: identifying string in PDF metadata"; exit 1
fi
if pdftotext main_qa.pdf - | grep -qiE "ottaviani|edhec|luiss"; then
  echo "FAIL: identifying string in PDF text"; exit 1
fi
echo "Anonymity check: PASSED"
cp main_qa.pdf "Quant Awards - Alessio Ottaviani - EDHEC Business School.pdf"
echo "Wrote: Quant Awards - Alessio Ottaviani - EDHEC Business School.pdf"
pdfinfo main_qa.pdf | grep Pages
