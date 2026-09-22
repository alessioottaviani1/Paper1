#!/bin/bash
# =============================================================================
# Build script for PhD Dissertation
# Usage:
#   ./build.sh draft    → compiles main_draft.tex (skeleton, inline tables)
#   ./build.sh final    → compiles main_final.tex (EDHEC format, tables at end)
#   ./build.sh          → defaults to draft
# =============================================================================

MODE=${1:-draft}
TEXFILE="main_${MODE}"

if [ ! -f "${TEXFILE}.tex" ]; then
    echo "❌ ERROR: ${TEXFILE}.tex not found."
    echo "   Usage: ./build.sh draft  OR  ./build.sh final"
    exit 1
fi

echo "=== Building PhD Dissertation (${MODE}) ==="

echo "[1/4] First pdflatex pass..."
pdflatex -interaction=nonstopmode ${TEXFILE}.tex > /dev/null 2>&1

echo "[2/4] Building bibliography..."
bibtex ${TEXFILE} > /dev/null 2>&1

echo "[3/4] Second pdflatex pass..."
pdflatex -interaction=nonstopmode ${TEXFILE}.tex > /dev/null 2>&1

echo "[4/4] Third pdflatex pass..."
pdflatex -interaction=nonstopmode ${TEXFILE}.tex > /dev/null 2>&1

if [ -f ${TEXFILE}.pdf ]; then
    echo ""
    echo "✅ SUCCESS: ${TEXFILE}.pdf generated"
    echo "   Size: $(du -h ${TEXFILE}.pdf | cut -f1)"
else
    echo ""
    echo "❌ ERROR: ${TEXFILE}.pdf not found. Check ${TEXFILE}.log for errors."
fi
