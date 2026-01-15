# 🇪🇬 Egyptian Document Verification System

> **AI-Powered Document Authentication** | Computer Vision & OCR Solution for Egyptian National IDs and Driving Licenses

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.0+-green.svg)](https://opencv.org/)
[![Tesseract](https://img.shields.io/badge/Tesseract-OCR-orange.svg)](https://github.com/tesseract-ocr/tesseract)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🎯 Overview

A sophisticated **document verification system** that uses advanced computer vision and OCR techniques to authenticate Egyptian National IDs and Driving Licenses. This project demonstrates expertise in **image processing**, **multilingual OCR**, **pattern recognition**, and **validation algorithms**.

### ✨ Key Features

- 🔍 **Multi-layered Validation**: 8+ comprehensive checks for driving licenses, 10+ for national IDs
- 🌐 **Multilingual OCR**: Supports Arabic and English text extraction using Tesseract & EasyOCR
- 🖼️ **Advanced Image Preprocessing**: CLAHE, adaptive thresholding, denoising, sharpening
- 📊 **Confidence Scoring**: Weighted validation system with detailed breakdown reports
- 🔎 **Feature Detection**: Barcode detection, photo region validation, emblem/shape recognition
- 📝 **Debug Mode**: Comprehensive logging and visual debugging information
- ⚡ **Batch Processing**: Validate entire folders of images efficiently

---

## 🚀 Technical Highlights

### Computer Vision & Image Processing
- **OpenCV** for image manipulation and analysis
- Multiple preprocessing pipelines for optimal OCR accuracy
- Color space analysis (HSV, BGR) for document authenticity
- Edge detection and contour analysis for shape recognition
- Face detection using Haar cascades

### OCR & Text Extraction
- **Tesseract OCR** with Arabic language pack support
- **EasyOCR** for enhanced multilingual recognition
- Multiple PSM (Page Segmentation Mode) strategies
- Text extraction with confidence scores

### Validation Algorithms
- **Dimension Analysis**: Aspect ratio and size validation
- **Keyword Detection**: Critical and secondary Arabic/English keywords
- **Pattern Matching**: License numbers, dates, vehicle classes
- **Structural Validation**: Photo regions, barcodes, emblems
- **Color Scheme Analysis**: Document authenticity verification

---

## 📁 Project Structure

```
.
├── driving.py                              # Egyptian Driving License Validator
├── back_id.py                              # National ID (Back Side) Validator  
├── production_egyptian_id_verifier_front.py # National ID (Front Side) Validator
└── README.md                               # This file
```

---

## 🛠️ Technologies Used

| Technology | Purpose |
|------------|---------|
| **Python 3.7+** | Core programming language |
| **OpenCV (cv2)** | Computer vision and image processing |
| **Tesseract OCR** | Optical character recognition |
| **EasyOCR** | Alternative OCR engine for Arabic/English |
| **NumPy** | Numerical operations and array manipulation |
| **PIL/Pillow** | Image enhancement and preprocessing |
| **pytesseract** | Python wrapper for Tesseract OCR |

---

## 📋 Requirements

### Prerequisites
- Python 3.7 or higher
- Tesseract OCR installed ([Download](https://github.com/UB-Mannheim/tesseract/wiki))
- Arabic language pack for Tesseract (`tesseract-ocr-ara`)

### Python Dependencies
```bash
pip install opencv-python
pip install pytesseract
pip install numpy
pip install Pillow
pip install easyocr  # Optional but recommended
```

---

## 💻 Usage

### Driving License Validation

```python
from driving import EgyptianDrivingLicenseValidator

# Initialize validator
validator = EgyptianDrivingLicenseValidator(debug=True)

# Validate a single image
result = validator.validate("path/to/license.jpg")

# Validate entire folder
results = validator.validate_folder("path/to/licenses/")

# Check results
print(f"Status: {result.status}")
print(f"Confidence: {result.confidence * 100}%")
print(f"Score Breakdown: {result.score_breakdown}")
```

### National ID Validation

```python
from back_id import EgyptianIDBackValidator
from production_egyptian_id_verifier_front import EgyptianIDFrontValidator

# Initialize validators
back_validator = EgyptianIDBackValidator()
front_validator = EgyptianIDFrontValidator()

# Validate ID images
back_result = back_validator.validate("id_back.jpg")
front_result = front_validator.validate("id_front.jpg")
```

---

## 🔬 Validation Checks

### Driving License Validator
1. ✅ **Dimensions**: Aspect ratio and minimum size validation
2. ✅ **Keywords**: Critical Arabic/English keyword detection
3. ✅ **License Number**: 14-digit format validation
4. ✅ **Dates**: Issue and expiry date pattern recognition
5. ✅ **Photo Region**: Face detection and content analysis
6. ✅ **Barcode**: Vertical line detection at bottom
7. ✅ **Color Scheme**: Yellowish/beige color matching
8. ✅ **Vehicle Class**: Class indicators (B, C, CE, DE, D)

### National ID Validator
- Front side: Photo, emblem, watermark, text regions
- Back side: Shape-based emblem detection, barcode, security features

---

## 📊 Example Output

```
======================================================================
                    EGYPTIAN DRIVING LICENSE VALIDATOR v3.0
                    Debug Mode - Improved OCR
======================================================================

Status: ✓ VALID
Confidence: 87.5%

Score Breakdown:
  ✓ 1_Dimensions              [██████████] 100.0%
  ✓ 2_Keywords                [█████████░]  90.0%
  ✓ 3_License_Number          [██████████] 100.0%
  ✓ 4_Dates                   [█████████░]  85.0%
  ✓ 5_Photo                   [████████░░]  80.0%
  ✓ 6_Barcode                 [███████░░░]  70.0%
  ✓ 7_Color                   [████████░░]  75.0%
  ✓ 8_Vehicle_Class           [██████████] 100.0%
```

---

## 🎓 Skills Demonstrated

### Technical Skills
- ✅ **Computer Vision**: Image processing, feature detection, pattern recognition
- ✅ **OCR**: Multilingual text extraction (Arabic & English)
- ✅ **Python**: Object-oriented design, dataclasses, type hints
- ✅ **Image Preprocessing**: CLAHE, thresholding, denoising, sharpening
- ✅ **Algorithm Design**: Weighted scoring systems, validation pipelines
- ✅ **Debugging**: Comprehensive logging and error handling

### Domain Knowledge
- ✅ Document authentication and fraud detection
- ✅ Multilingual text processing
- ✅ Image quality assessment
- ✅ Pattern matching and validation logic

---

## 🔧 Configuration

### Tesseract Path (Windows)
If Tesseract is not in your PATH, specify the executable:

```python
validator = EgyptianDrivingLicenseValidator(
    tesseract_path=r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    debug=True
)
```

### Validation Thresholds
Adjustable confidence thresholds and weighted scoring system allow fine-tuning of validation sensitivity.

---

## 📈 Future Enhancements

- [ ] Web interface for document upload
- [ ] API endpoints for integration
- [ ] Machine learning model for enhanced fraud detection
- [ ] Support for additional document types
- [ ] Cloud-based processing option
- [ ] Real-time video validation

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

---

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

---

## 👨‍💻 Author

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your Profile](https://linkedin.com/in/yourprofile)
- Email: your.email@example.com

---

## 🙏 Acknowledgments

- Tesseract OCR team for the excellent OCR engine
- OpenCV community for comprehensive computer vision tools
- EasyOCR for multilingual OCR capabilities

---

<div align="center">

**⭐ Star this repo if you find it helpful! ⭐**

Made with ❤️ using Python, OpenCV, and Tesseract

</div>

