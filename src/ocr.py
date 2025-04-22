import os
import glob
from pathlib import Path
import pytesseract
from PIL import Image

# 设置raw_data目录路径
RAW_DATA_DIR = "raw_data"

# 如果你的Tesseract不在系统PATH中，需要指定路径
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Windows路径示例
# pytesseract.pytesseract.tesseract_cmd = r'/usr/local/bin/tesseract'  # Mac/Linux路径示例
pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/Cellar/tesseract/5.5.0_1/bin/tesseract'  # Mac路径示例

def ocr_image(image_path):
    """使用pytesseract处理图像，返回识别的文本"""
    try:
        # 打开图像
        img = Image.open(image_path)
        
        # 使用pytesseract识别文本
        # 可以根据图片语言调整lang参数，例如中文使用'chi_sim'，英文使用'eng'
        # limitation：只用了英文OCR
        text = pytesseract.image_to_string(img, lang='eng')
        # text = pytesseract.image_to_string(img, lang='chi_sim+eng')
        
        return text
    except Exception as e:
        print(f"处理图像 {image_path} 时出错: {str(e)}")
        return ""

def process_all_images():
    """处理raw_data目录中的所有JPG图像"""
    # 确保raw_data目录存在
    if not os.path.exists(RAW_DATA_DIR):
        print(f"目录 {RAW_DATA_DIR} 不存在")
        return
    
    # 获取所有JPG文件
    jpg_files = glob.glob(os.path.join(RAW_DATA_DIR, "*.jpg"))
    
    print(f"找到 {len(jpg_files)} 个JPG文件")
    
    for jpg_file in jpg_files:
        try:
            # 获取文件基本名（不含扩展名）
            base_name = Path(jpg_file).stem
            # 构建输出文件名
            output_file = os.path.join(RAW_DATA_DIR, f"{base_name}_ocr.txt")
            
            print(f"正在处理: {jpg_file}")
            
            # 调用OCR处理
            text = ocr_image(jpg_file)
            
            # 保存结果到文本文件
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(text)
            
            print(f"保存OCR结果到: {output_file}")
            
        except Exception as e:
            print(f"处理 {jpg_file} 时出错: {str(e)}")

if __name__ == "__main__":
    process_all_images()
