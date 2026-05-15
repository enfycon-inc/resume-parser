import fitz  # PyMuPDF
import docx
import pytesseract
from PIL import Image
import io
import logging
import os

# Set Tesseract path for Windows (Update this if you installed it elsewhere)
if os.name == 'nt':
    pytesseract.pytesseract.tesseract_cmd = r'C:\Users\deb\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResumeExtractor:
    @staticmethod
    def extract_from_pdf(file_content: bytes) -> str:
        """Extracts text from PDF, falling back to OCR if no text is found."""
        text = ""
        try:
            doc = fitz.open(stream=file_content, filetype="pdf")
            for page in doc:
                # "blocks" extraction identifies visual boxes on the page
                # We sort by block number (vertical then horizontal)
                blocks = page.get_text("blocks", sort=True)
                for b in blocks:
                    # b[4] is the actual text in the block
                    text += b[4] + "\n"
            
            # If extraction yields very little text, it might be a scanned PDF
            if len(text.strip()) < 50:
                logger.info("PDF appears to be scanned. Falling back to OCR.")
                text = ""
                for page in doc:
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2)) # Higher DPI for OCR
                    img = Image.open(io.BytesIO(pix.tobytes()))
                    text += pytesseract.image_to_string(img)
            
            doc.close()
        except Exception as e:
            logger.error(f"Error extracting from PDF: {e}")
            raise e
        return text

    @staticmethod
    def extract_from_docx(file_content: bytes) -> str:
        """Extracts text from DOCX files."""
        try:
            doc = docx.Document(io.BytesIO(file_content))
            return "\n".join([para.text for para in doc.paragraphs])
        except Exception as e:
            logger.error(f"Error extracting from DOCX: {e}")
            raise e

    @staticmethod
    def extract_from_image(file_content: bytes) -> str:
        """Extracts text from raw images (JPG, PNG) with pre-processing."""
        try:
            img = Image.open(io.BytesIO(file_content))
            
            # Pre-processing for better OCR (especially for handwriting/scans)
            # 1. Convert to grayscale
            img = img.convert('L')
            
            # 2. Increase contrast/thresholding can help, but let's start with basic sharpening
            from PIL import ImageOps, ImageFilter
            img = ImageOps.autocontrast(img)
            
            # Use PSM 1 (Automatic page segmentation with OSD) or PSM 3
            # PSM 6 is also good for uniform blocks of text
            custom_config = r'--oem 3 --psm 3'
            text = pytesseract.image_to_string(img, config=custom_config)
            
            logger.info(f"Extracted {len(text)} characters from image.")
            return text
        except Exception as e:
            logger.error(f"Error extracting from Image: {e}")
            raise e
