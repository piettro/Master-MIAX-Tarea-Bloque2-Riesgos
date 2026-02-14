
from pathlib import Path

try:
    import markdown
    from weasyprint import HTML, CSS
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    print("Warning: markdown/weasyprint not available. PDF generation will be skipped.")

BASE_DIR = Path("..")
REPORT_DIR = BASE_DIR / "Riesgos" / "report"

markdown_path = REPORT_DIR / "INFORME_EJECUTIVO.md"

def markdown_to_pdf(markdown_path: Path, pdf_path: Path) -> None:
    """Convert markdown file to PDF using markdown + weasyprint.
    
    Parameters
    ----------
    markdown_path : Path
        Path to input markdown file.
    pdf_path : Path
        Path to output PDF file.
    """
    
    try:
        # Read markdown
        with open(markdown_path, "r", encoding="utf-8") as f:
            md_content = f.read()
        
        # Convert markdown to HTML
        html_content = markdown.markdown(
            md_content,
            extensions=['extra', 'tables', 'codehilite'],
            extension_configs={
                'codehilite': {
                    'css_class': 'highlight'
                }
            }
        )
        
        # Add basic CSS styling for PDF
        html_with_style = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                @page {{
                    size: A4;
                    margin: 1cm;
                }}
                body {{
                    font-family: 'Arial', sans-serif;
                    font-size: 10pt;
                    line-height: 1.6;
                    color: #333;
                }}
                h1 {{
                    font-size: 16pt;
                    color: #1a1a1a;
                    border-bottom: 2px solid #333;
                }}
                h2 {{
                    font-size: 14pt;
                    color: #2a2a2a;
                    margin-top: 0.3cm;
                    margin-bottom: 0.4cm;
                }}
                h3 {{
                    font-size: 12pt;
                    color: #3a3a3a;
                    margin-top: 0.3cm;
                    margin-bottom: 0.3cm;
                }}
                table {{
                    border-collapse: collapse;
                    width: 100%;
                    margin: 0.1cm 0;
                    font-size: 9pt;
                }}
                th, td {{
                    border: 1px solid #ddd;
                    padding: 1px;
                    text-align: left;
                }}
                th {{
                    background-color: #f2f2f2;
                    font-weight: bold;
                }}
                code {{
                    background-color: #f4f4f4;
                    padding: 2px 4px;
                    border-radius: 3px;
                    font-family: 'Courier New', monospace;
                    font-size: 9pt;
                }}
                pre {{
                    background-color: #f4f4f4;
                    padding: 0.5cm;
                    border-radius: 5px;
                    overflow-x: auto;
                    font-size: 8pt;
                }}
                img {{
                    max-width: 100%;
                    height: auto;
                    page-break-inside: avoid;
                }}
                p {{
                    margin: 0.3cm 0;
                }}
                ul, ol {{
                    margin: 0.3cm 0;
                    padding-left: 1cm;
                }}
            </style>
        </head>
        <body>
            {html_content}
        </body>
        </html>
        """
        
        # Convert HTML to PDF
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        HTML(string=html_with_style, base_url=str(markdown_path.parent)).write_pdf(pdf_path)
        print(f"✓ PDF generado: {pdf_path}")
        
    except Exception as e:
        print(f"Error generando PDF: {e}")
        print("Asegúrate de tener instalado: pip install markdown weasyprint")


# Generar PDF a partir del markdown
pdf_path = REPORT_DIR / "INFORME_EJECUTIVO.pdf"
markdown_to_pdf(markdown_path, pdf_path)