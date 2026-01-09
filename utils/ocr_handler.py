import pytesseract
from PIL import Image
import json

class OCRHandler:
    def __init__(self):
        self.confidence_threshold = 0.7
    
    def extract_text_from_image(self, image_path_or_pil: any):
        """
        Extract text from image using Tesseract
        Returns: (text, confidence, warnings)
        """
        if isinstance(image_path_or_pil, str):
            image = Image.open(image_path_or_pil)
        else:
            image = image_path_or_pil
        
        # Extract text with details
        data = pytesseract.image_to_data(
            image,
            output_type=pytesseract.Output.DICT
        )
        
        # Calculate average confidence
        confidences = [int(c) for c in data['conf'] if int(c) > 0]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        avg_confidence = avg_confidence / 100.0
        
        extracted_text = pytesseract.image_to_string(image)
        
        warnings = []
        if avg_confidence < self.confidence_threshold:
            warnings.append(f"⚠️ Low OCR confidence ({avg_confidence:.0%}). Please review and correct.")
        
        if len(extracted_text.strip()) < 10:
            warnings.append("⚠️ Very little text detected. Image might be unclear.")
        
        return {
            "text": extracted_text,
            "confidence": avg_confidence,
            "warnings": warnings,
            "needs_review": avg_confidence < self.confidence_threshold
        }
