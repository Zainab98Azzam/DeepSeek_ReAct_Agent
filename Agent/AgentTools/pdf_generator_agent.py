# AgentTools/pdf_generator_agent.py

import io
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from .base_tool import BaseTool

class PDFGeneratorAgent(BaseTool):
    """
    A tool that converts a final, long text report into a binary PDF document.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def use(self, report_text: str) -> bytes:
        """
        Converts the provided report_text (in Markdown or plain text format) into a PDF file. 
        Returns the binary content of the PDF.
        """
        # Create a file-like buffer to receive PDF data.
        buffer = io.BytesIO()
        
        # Create the PDF document object
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        flowables = []

        # Simple way to parse paragraphs (for structured text like the report)
        # We replace newlines with a recognizable token for splitting
        
        # Basic parsing to separate blocks of text (assuming report has headers/paragraphs)
        paragraphs = report_text.strip().split('\n\n')

        for para in paragraphs:
            # Clean up potential markdown headers before adding
            cleaned_text = para.replace('#', '').strip()
            
            # Simple heuristic to make headers bold
            if len(cleaned_text.split()) < 8 and (para.startswith('#') or para.isupper()):
                 style = styles['Heading2']
            else:
                 style = styles['Normal']

            if cleaned_text:
                # Add the paragraph content
                flowables.append(Paragraph(cleaned_text, style))
                # Add a small space after each paragraph
                flowables.append(Spacer(1, 0.1 * letter[1])) 

        # Build the PDF document
        doc.build(flowables)
        
        # Get the value of the BytesIO buffer (the PDF content) and return it
        pdf_content = buffer.getvalue()
        buffer.close()
        return pdf_content