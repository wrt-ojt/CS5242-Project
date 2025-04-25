import os
import glob
from pathlib import Path
import pytesseract
from PIL import Image

# Set the path to the raw_data directory
RAW_DATA_DIR = "raw_data"

# If Tesseract is not in your system PATH, you need to specify the path
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Windows path example
# pytesseract.pytesseract.tesseract_cmd = r'/usr/local/bin/tesseract'  # Mac/Linux path example
pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/Cellar/tesseract/5.5.0_1/bin/tesseract'  # Mac路径示例

def ocr_image(image_path):
    """Process image using pytesseract and return recognized text"""
    try:
        # Open the image
        img = Image.open(image_path)
        
        # Use pytesseract to recognize text
        # Adjust the lang parameter based on your image content
        # 'chi_sim' for Simplified Chinese, 'eng' for English
        text = pytesseract.image_to_string(img, lang='eng') # limitation: only use English OCR
        # text = pytesseract.image_to_string(img, lang='chi_sim+eng')
        
        return text
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return ""

def perform_ocr_on_all_images():
    """Process all JPG images in the raw_data directory"""
    # Ensure raw_data directory exists
    if not os.path.exists(RAW_DATA_DIR):
        print(f"Directory {RAW_DATA_DIR} does not exist")
        return
    
    # Get all JPG files
    jpg_files = glob.glob(os.path.join(RAW_DATA_DIR, "*.jpg"))
    
    print(f"Found {len(jpg_files)} JPG files")
    
    for jpg_file in jpg_files:
        try:
            # Get the base name (without extension)
            base_name = Path(jpg_file).stem
            # Build output filename
            output_file = os.path.join(RAW_DATA_DIR, f"{base_name}_ocr.txt")
            
            print(f"Processing: {jpg_file}")
            
            # Call OCR processing
            text = ocr_image(jpg_file)
            
            # Save results to text file
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(text)
            
            print(f"Saved OCR results to: {output_file}")
            
        except Exception as e:
            print(f"Error processing {jpg_file}: {str(e)}")

if __name__ == "__main__":
    perform_ocr_on_all_images()
