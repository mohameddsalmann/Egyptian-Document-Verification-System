"""
Egyptian National ID (Back Side) Validator
Standards-Based Validation with Shape-Based Emblem Detection
"""

import cv2
import numpy as np
import os
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Dict, Any
from enum import Enum
import glob
import re
from datetime import datetime, date, timedelta


# ============ OCR INITIALIZATION ============
print("=" * 60)
print("Initializing OCR Engine (one-time load)...")
print("=" * 60)

TESSERACT_AVAILABLE = False
try:
    import pytesseract
    version = pytesseract.get_tesseract_version()
    print(f"[OK] Tesseract OCR v{version}")
    TESSERACT_AVAILABLE = True
except Exception as e:
    print(f"[WARN] Tesseract not available: {e}")

EASYOCR_AVAILABLE = False
easyocr_reader = None
try:
    import easyocr
    print("[OK] Loading EasyOCR (Arabic + English)...")
    easyocr_reader = easyocr.Reader(['ar', 'en'], gpu=False)
    print("[OK] EasyOCR ready")
    EASYOCR_AVAILABLE = True
except Exception as e:
    print(f"[WARN] EasyOCR not available: {e}")

CURRENT_DATE = date.today()
print("=" * 60)
print(f"[INFO] Current Date: {CURRENT_DATE.strftime('%Y-%m-%d')}")
print("=" * 60)
print()


# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║                    EGYPTIAN NATIONAL ID STANDARDS TABLE                        ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝

STANDARDS = {
    "card": {
        "aspect_ratio_min": 1.2,
        "aspect_ratio_max": 2.2,
        "min_width": 250,
        "min_height": 150,
    },
    "national_id": {
        "length": 14,
        "century_codes": ["2", "3"],
        "governorates": {
            "01": "القاهرة", "02": "الإسكندرية", "03": "بورسعيد", "04": "السويس",
            "11": "دمياط", "12": "الدقهلية", "13": "الشرقية", "14": "القليوبية",
            "15": "كفر الشيخ", "16": "الغربية", "17": "المنوفية", "18": "البحيرة",
            "19": "الإسماعيلية", "21": "الجيزة", "22": "بني سويف", "23": "الفيوم",
            "24": "المنيا", "25": "أسيوط", "26": "سوهاج", "27": "قنا", "28": "أسوان",
            "29": "الأقصر", "31": "البحر الأحمر", "32": "الوادي الجديد", "33": "مطروح",
            "34": "شمال سيناء", "35": "جنوب سيناء", "88": "خارج الجمهورية",
        },
    },
    "gender": {
        "male": ["ذكر", "ذکر"],
        "female": ["أنثى", "انثى", "أنثي", "انثي", "انتى", "أنتي"],
    },
    "religion": {
        "muslim_m": ["مسلم"],
        "muslim_f": ["مسلمة", "مسلمه"],
        "christian_m": ["مسيحى", "مسيحي"],
        "christian_f": ["مسيحية", "مسيحيه"],
        "other": ["أخرى", "اخرى", "أخري", "اخري"],
    },
    "marital": {
        "single_m": ["أعزب", "اعزب", "عزب"],
        "single_f": ["آنسة", "انسة", "أنسة", "انسه", "آنسه"],
        "married_m": ["متزوج"],
        "married_f": ["متزوجة", "متزوجه"],
        "divorced_m": ["مطلق"],
        "divorced_f": ["مطلقة", "مطلقه"],
        "widowed_m": ["أرمل", "ارمل"],
        "widowed_f": ["أرملة", "ارملة", "ارمله"],
    },
    "occupation": [
        "بدون عمل", "بدون", "لا يعمل", "لا تعمل",
        "ربة منزل", "ربه منزل", "ربة بيت",
        "طالب", "طالبة", "موظف", "موظفة",
        "مهندس", "مهندسة", "طبيب", "طبيبة", "دكتور", "دكتورة",
        "محاسب", "محاسبة", "محامى", "محامي", "محامية",
        "صيدلى", "صيدلي", "صيدلية", "صيدلانى", "صيدلاني", "صيدلانية",
        "مدرس", "مدرسة", "معلم", "معلمة",
        "سائق", "سواق", "عامل", "عاملة", "فلاح",
        "تاجر", "مدير", "مديرة", "مدير تجارى", "مدير تجاري",
        "رجل اعمال", "سيدة اعمال", "حر", "أعمال حرة", "اعمال حرة",
        "ضابط", "شركة", "للملاحة",
    ],
    "validity_phrases": [
        "البطاقة سارية حتى", "البطاقه ساريه حتى", "البطاقة سارية حتي",
        "سارية حتى", "ساريه حتى", "سارية حتي", "سارية", "ساريه", "حتى", "حتي",
    ],
    "arabic_numerals": {
        '٠': '0', '١': '1', '٢': '2', '٣': '3', '٤': '4',
        '٥': '5', '٦': '6', '٧': '7', '٨': '8', '٩': '9',
    },
}


# ============ ENUMS & DATA CLASSES ============
class ValidationStatus(Enum):
    VALID = "valid"
    INVALID = "invalid"
    NEEDS_REVIEW = "needs_review"
    EXPIRED = "expired"


class ExpiryStatus(Enum):
    VALID = "valid"
    EXPIRED = "expired"
    EXPIRING_SOON = "expiring_soon"
    UNKNOWN = "unknown"


@dataclass
class ExpiryInfo:
    status: ExpiryStatus = ExpiryStatus.UNKNOWN
    expiry_date: Optional[date] = None
    expiry_date_str: Optional[str] = None
    days_until_expiry: Optional[int] = None
    is_expired: bool = False
    message: str = "Could not determine expiry date"


@dataclass
class ExtractedFields:
    national_id: Optional[str] = None
    governorate: Optional[str] = None
    birth_date: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    gender_from_id: Optional[str] = None
    religion: Optional[str] = None
    marital_status: Optional[str] = None
    occupation: Optional[str] = None
    issue_date: Optional[str] = None
    expiry_date: Optional[str] = None
    expiry_date_parsed: Optional[date] = None
    validity_phrase: Optional[str] = None
    raw_ocr_text: Optional[str] = None


@dataclass
class ValidationResult:
    status: ValidationStatus
    confidence: float
    checks_passed: List[str]
    checks_failed: List[str]
    fields: ExtractedFields
    expiry_info: ExpiryInfo
    details: Dict


# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║                    SHAPE-BASED EMBLEM DETECTOR                                 ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝

class ShapeBasedEmblemDetector:
    """
    Detects Pharaoh and Eagle emblems using shape analysis.
    Independent of color/lighting conditions.
    
    Pharaoh Emblem Characteristics:
    - Vertically oriented (taller than wide)
    - Symmetric along vertical axis
    - Has a distinctive head shape at top
    - Contains internal details (stripes pattern)
    
    Eagle Emblem Characteristics:
    - Wider shape (wings spread)
    - Has horizontal stripe pattern (flag)
    - Symmetric along vertical axis
    - Contains shield shape in center
    """
    
    @staticmethod
    def preprocess_for_shape(image: np.ndarray) -> List[np.ndarray]:
        """Generate multiple binary images for shape detection."""
        results = []
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 1. Otsu threshold
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        results.append(otsu)
        
        # 2. Inverted Otsu
        results.append(cv2.bitwise_not(otsu))
        
        # 3. Adaptive threshold
        adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 11, 2)
        results.append(adaptive)
        
        # 4. Canny edges
        edges = cv2.Canny(gray, 30, 100)
        # Dilate edges to connect nearby lines
        kernel = np.ones((2, 2), np.uint8)
        edges_dilated = cv2.dilate(edges, kernel, iterations=1)
        results.append(edges_dilated)
        
        # 5. Multiple fixed thresholds
        for thresh in [80, 100, 127, 150, 180]:
            _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
            results.append(binary)
        
        # 6. CLAHE enhanced + Otsu
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        _, clahe_otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        results.append(clahe_otsu)
        
        return results
    
    @staticmethod
    def compute_shape_features(binary: np.ndarray) -> Dict:
        """Compute shape-based features from binary image."""
        features = {
            "contour_count": 0,
            "total_contour_area": 0,
            "largest_contour_area": 0,
            "largest_contour_aspect": 0,
            "symmetry_score": 0,
            "edge_density": 0,
            "complexity": 0,
            "solidity": 0,
            "extent": 0,
            "hu_moments": None,
        }
        
        h, w = binary.shape
        total_pixels = h * w
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return features
        
        features["contour_count"] = len(contours)
        
        # Total contour area
        total_area = sum(cv2.contourArea(c) for c in contours)
        features["total_contour_area"] = total_area / total_pixels
        
        # Largest contour analysis
        largest = max(contours, key=cv2.contourArea)
        largest_area = cv2.contourArea(largest)
        features["largest_contour_area"] = largest_area / total_pixels
        
        # Bounding rect of largest contour
        x, y, cw, ch = cv2.boundingRect(largest)
        features["largest_contour_aspect"] = ch / cw if cw > 0 else 0
        
        # Solidity (contour area / convex hull area)
        hull = cv2.convexHull(largest)
        hull_area = cv2.contourArea(hull)
        features["solidity"] = largest_area / hull_area if hull_area > 0 else 0
        
        # Extent (contour area / bounding rect area)
        rect_area = cw * ch
        features["extent"] = largest_area / rect_area if rect_area > 0 else 0
        
        # Hu Moments (shape descriptor - rotation/scale invariant)
        moments = cv2.moments(largest)
        hu_moments = cv2.HuMoments(moments).flatten()
        features["hu_moments"] = hu_moments
        
        # Complexity (perimeter^2 / area) - higher for complex shapes
        perimeter = cv2.arcLength(largest, True)
        if largest_area > 0:
            features["complexity"] = (perimeter ** 2) / largest_area
        
        return features
    
    @staticmethod
    def compute_symmetry(binary: np.ndarray) -> float:
        """Compute vertical symmetry score."""
        h, w = binary.shape
        
        # Split into left and right halves
        mid = w // 2
        left = binary[:, :mid]
        right = binary[:, mid:mid + left.shape[1]]
        
        # Flip right half
        right_flipped = cv2.flip(right, 1)
        
        # Ensure same size
        min_w = min(left.shape[1], right_flipped.shape[1])
        left = left[:, :min_w]
        right_flipped = right_flipped[:, :min_w]
        
        # Compare
        if left.size == 0:
            return 0
        
        # XOR to find differences
        diff = cv2.bitwise_xor(left, right_flipped)
        diff_ratio = np.sum(diff > 0) / diff.size
        
        # Symmetry score (1 = perfect symmetry)
        symmetry = 1 - diff_ratio
        
        return symmetry
    
    @staticmethod
    def compute_edge_density(gray: np.ndarray) -> float:
        """Compute edge density."""
        edges = cv2.Canny(gray, 30, 100)
        return np.sum(edges > 0) / edges.size
    
    @staticmethod
    def compute_texture_features(gray: np.ndarray) -> Dict:
        """Compute texture features using gradients."""
        # Sobel gradients
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Gradient magnitude
        magnitude = np.sqrt(gx**2 + gy**2)
        
        # Gradient direction
        direction = np.arctan2(gy, gx)
        
        return {
            "gradient_mean": np.mean(magnitude),
            "gradient_std": np.std(magnitude),
            "gradient_max": np.max(magnitude),
            "horizontal_energy": np.sum(np.abs(gx)),
            "vertical_energy": np.sum(np.abs(gy)),
        }
    
    @staticmethod
    def detect_pharaoh_emblem(roi: np.ndarray) -> Tuple[bool, float, Dict]:
        """
        Detect Pharaoh emblem using shape features.
        
        Expected characteristics:
        - Vertical orientation (aspect ratio > 1)
        - High vertical symmetry
        - Moderate complexity (detailed but structured)
        - Contains multiple internal contours
        - Edge density in middle range
        """
        detector = ShapeBasedEmblemDetector()
        
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi.copy()
        
        h, w = gray.shape
        if h < 20 or w < 20:
            return False, 0, {"error": "ROI too small"}
        
        details = {}
        scores = []
        
        # Process multiple binary versions
        binaries = detector.preprocess_for_shape(roi)
        
        best_score = 0
        best_details = {}
        
        for binary in binaries:
            score = 0
            
            # 1. Shape features
            shape_features = detector.compute_shape_features(binary)
            
            # 2. Symmetry
            symmetry = detector.compute_symmetry(binary)
            
            # 3. Edge density
            edge_density = detector.compute_edge_density(gray)
            
            # 4. Texture
            texture = detector.compute_texture_features(gray)
            
            # Scoring based on Pharaoh emblem characteristics
            
            # Aspect ratio: Pharaoh is taller than wide (1.2 - 2.5)
            aspect = shape_features["largest_contour_aspect"]
            if 0.8 <= aspect <= 3.0:
                score += 0.15
            if 1.0 <= aspect <= 2.5:
                score += 0.10
            
            # Symmetry: Pharaoh mask is symmetric (>0.6)
            if symmetry > 0.5:
                score += 0.15
            if symmetry > 0.7:
                score += 0.10
            
            # Edge density: Has details but not noise (0.03 - 0.25)
            if 0.02 <= edge_density <= 0.30:
                score += 0.15
            if 0.05 <= edge_density <= 0.20:
                score += 0.05
            
            # Contour area: Should have significant content (0.1 - 0.7)
            area = shape_features["total_contour_area"]
            if 0.05 <= area <= 0.80:
                score += 0.10
            
            # Solidity: Pharaoh shape is fairly solid (0.4 - 0.9)
            solidity = shape_features["solidity"]
            if 0.3 <= solidity <= 0.95:
                score += 0.10
            
            # Complexity: Moderate complexity (not too simple, not chaotic)
            complexity = shape_features["complexity"]
            if 15 <= complexity <= 200:
                score += 0.10
            
            if score > best_score:
                best_score = score
                best_details = {
                    "aspect_ratio": round(aspect, 2),
                    "symmetry": round(symmetry, 2),
                    "edge_density": round(edge_density, 3),
                    "contour_area": round(area, 3),
                    "solidity": round(solidity, 2),
                    "complexity": round(complexity, 1),
                    "score": round(score, 2),
                }
        
        valid = best_score >= 0.40
        
        return valid, best_score, best_details
    
    @staticmethod
    def detect_eagle_emblem(roi: np.ndarray) -> Tuple[bool, float, Dict]:
        """
        Detect Eagle emblem using shape features.
        
        Expected characteristics:
        - Wider shape or roughly square (wings spread)
        - Vertical symmetry (eagle is symmetric)
        - Contains horizontal stripe pattern (flag)
        - Moderate to high edge density
        - Complex shape with internal details
        """
        detector = ShapeBasedEmblemDetector()
        
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi.copy()
        
        h, w = gray.shape
        if h < 20 or w < 20:
            return False, 0, {"error": "ROI too small"}
        
        # Process multiple binary versions
        binaries = detector.preprocess_for_shape(roi)
        
        best_score = 0
        best_details = {}
        
        for binary in binaries:
            score = 0
            
            # 1. Shape features
            shape_features = detector.compute_shape_features(binary)
            
            # 2. Symmetry
            symmetry = detector.compute_symmetry(binary)
            
            # 3. Edge density
            edge_density = detector.compute_edge_density(gray)
            
            # 4. Texture - check for horizontal stripes (flag)
            texture = detector.compute_texture_features(gray)
            
            # 5. Horizontal line detection (flag stripes)
            horizontal_ratio = texture["horizontal_energy"] / (texture["vertical_energy"] + 1)
            
            # Scoring based on Eagle emblem characteristics
            
            # Aspect ratio: Eagle with wings is wider or square (0.5 - 1.8)
            aspect = shape_features["largest_contour_aspect"]
            if 0.4 <= aspect <= 2.0:
                score += 0.15
            if 0.6 <= aspect <= 1.5:
                score += 0.05
            
            # Symmetry: Eagle is symmetric (>0.5)
            if symmetry > 0.4:
                score += 0.15
            if symmetry > 0.6:
                score += 0.10
            
            # Edge density: Has details (0.03 - 0.30)
            if 0.02 <= edge_density <= 0.35:
                score += 0.15
            if 0.05 <= edge_density <= 0.25:
                score += 0.05
            
            # Contour area: Should have content (0.1 - 0.8)
            area = shape_features["total_contour_area"]
            if 0.05 <= area <= 0.85:
                score += 0.10
            
            # Solidity: Eagle has moderate solidity (0.3 - 0.85)
            solidity = shape_features["solidity"]
            if 0.25 <= solidity <= 0.90:
                score += 0.10
            
            # Horizontal stripes (flag) - more horizontal than vertical energy
            if horizontal_ratio > 0.7:
                score += 0.10
            
            # Multiple contours (eagle has internal details)
            if shape_features["contour_count"] >= 2:
                score += 0.05
            
            if score > best_score:
                best_score = score
                best_details = {
                    "aspect_ratio": round(aspect, 2),
                    "symmetry": round(symmetry, 2),
                    "edge_density": round(edge_density, 3),
                    "contour_area": round(area, 3),
                    "solidity": round(solidity, 2),
                    "horizontal_ratio": round(horizontal_ratio, 2),
                    "contour_count": shape_features["contour_count"],
                    "score": round(score, 2),
                }
        
        valid = best_score >= 0.40
        
        return valid, best_score, best_details


# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║                    IMAGE PREPROCESSOR                                          ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝

class ImagePreprocessor:
    """Image preprocessing for OCR."""
    
    @staticmethod
    def preprocess_all(image: np.ndarray) -> List[np.ndarray]:
        """Apply all preprocessing techniques."""
        results = []
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            results.append(image)
        else:
            gray = image.copy()
        
        results.append(gray)
        
        # Otsu
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        results.append(otsu)
        results.append(cv2.bitwise_not(otsu))
        
        # Adaptive
        adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 11, 2)
        results.append(adaptive)
        
        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        results.append(enhanced)
        _, clahe_otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        results.append(clahe_otsu)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        results.append(denoised)
        
        # Sharpen
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(gray, -1, kernel)
        results.append(sharpened)
        
        # Fixed thresholds
        for thresh in [100, 127, 150, 180]:
            _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
            results.append(binary)
        
        return results
    
    @staticmethod
    def upscale_if_small(image: np.ndarray, min_height: int = 50) -> np.ndarray:
        """Upscale image if too small."""
        h = image.shape[0]
        if h < min_height:
            scale = max(2, (min_height // h) + 1)
            return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        return image
    
    @staticmethod
    def deskew(image: np.ndarray) -> np.ndarray:
        """Deskew image if tilted."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        try:
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=50, maxLineGap=10)
            
            if lines is not None and len(lines) > 0:
                angles = []
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                    if abs(angle) < 30:
                        angles.append(angle)
                
                if angles:
                    median_angle = np.median(angles)
                    if abs(median_angle) > 0.5:
                        h, w = image.shape[:2]
                        center = (w // 2, h // 2)
                        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                        return cv2.warpAffine(image, M, (w, h),
                                              flags=cv2.INTER_CUBIC,
                                              borderMode=cv2.BORDER_REPLICATE)
        except:
            pass
        
        return image


# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║                    DUAL OCR ENGINE                                             ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝

class DualOCREngine:
    """Dual OCR using EasyOCR and Tesseract."""
    
    def __init__(self):
        self.preprocessor = ImagePreprocessor()
    
    def read_text(self, image: np.ndarray, max_preprocess: int = 8) -> str:
        """Read text using multiple preprocessing methods."""
        all_texts = []
        
        preprocessed = self.preprocessor.preprocess_all(image)[:max_preprocess]
        
        # EasyOCR
        if EASYOCR_AVAILABLE and easyocr_reader:
            for proc_img in preprocessed:
                try:
                    if len(proc_img.shape) == 2:
                        proc_color = cv2.cvtColor(proc_img, cv2.COLOR_GRAY2BGR)
                    else:
                        proc_color = proc_img
                    
                    results = easyocr_reader.readtext(proc_color)
                    for _, text, conf in results:
                        text = text.strip()
                        if text and conf > 0.05:
                            all_texts.append(text)
                except:
                    pass
        
        # Tesseract
        if TESSERACT_AVAILABLE:
            configs = [
                '--psm 6 -l ara+eng',
                '--psm 3 -l ara+eng',
                '--psm 4 -l ara+eng',
                '--psm 11 -l ara+eng',
            ]
            
            for proc_img in preprocessed[:4]:
                for config in configs[:3]:
                    try:
                        text = pytesseract.image_to_string(proc_img, config=config)
                        text = text.strip()
                        if text:
                            all_texts.append(text)
                    except:
                        pass
        
        return ' '.join(all_texts)
    
    def read_region(self, image: np.ndarray, x1: float, y1: float, x2: float, y2: float,
                    padding_pct: float = 0.02) -> str:
        """Read text from specific region."""
        h, w = image.shape[:2]
        
        pad_x = int(w * padding_pct)
        pad_y = int(h * padding_pct)
        
        px1 = max(0, int(w * x1) - pad_x)
        py1 = max(0, int(h * y1) - pad_y)
        px2 = min(w, int(w * x2) + pad_x)
        py2 = min(h, int(h * y2) + pad_y)
        
        roi = image[py1:py2, px1:px2]
        
        if roi.shape[0] < 10 or roi.shape[1] < 10:
            return ""
        
        roi = self.preprocessor.upscale_if_small(roi)
        
        return self.read_text(roi)


# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║                    MAIN VALIDATOR                                              ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝

class EgyptianIDValidator:
    """Egyptian National ID Back Validator with Shape-Based Detection."""
    
    REGIONS = {
        'row1_full': (0.10, 0.0, 0.90, 0.18),
        'issue_date': (0.10, 0.0, 0.40, 0.18),
        'national_id': (0.30, 0.0, 0.90, 0.18),
        'occupation': (0.12, 0.12, 0.88, 0.35),
        'status_full': (0.10, 0.28, 0.90, 0.52),
        'gender': (0.55, 0.28, 0.90, 0.52),
        'religion': (0.30, 0.28, 0.65, 0.52),
        'marital': (0.08, 0.28, 0.40, 0.52),
        'validity': (0.08, 0.42, 0.92, 0.62),
        'full_text': (0.08, 0.0, 0.92, 0.62),
        'pharaoh': (0.0, 0.0, 0.18, 0.52),
        'eagle': (0.82, 0.0, 1.0, 0.52),
        'barcode': (0.02, 0.52, 0.98, 0.99),
    }
    
    def __init__(self, expiry_warning_days: int = 30):
        self.standards = STANDARDS
        self.ocr = DualOCREngine()
        self.preprocessor = ImagePreprocessor()
        self.emblem_detector = ShapeBasedEmblemDetector()
        self.expiry_warning_days = expiry_warning_days
        self.today = CURRENT_DATE
    
    def _convert_arabic_nums(self, text: str) -> str:
        for ar, en in self.standards["arabic_numerals"].items():
            text = text.replace(ar, en)
        return text
    
    def _normalize_arabic(self, text: str) -> str:
        replacements = [
            ('ة', 'ه'), ('ى', 'ي'), ('أ', 'ا'), ('إ', 'ا'), ('آ', 'ا'),
            ('ؤ', 'و'), ('ئ', 'ي'),
        ]
        for old, new in replacements:
            text = text.replace(old, new)
        return text
    
    def _find_in_text(self, text: str, values: List[str]) -> Optional[str]:
        text_norm = self._normalize_arabic(text)
        for value in values:
            value_norm = self._normalize_arabic(value)
            if value in text or value_norm in text_norm:
                return value
        return None
    
    def _get_all_values(self, field: str) -> List[str]:
        if field not in self.standards:
            return []
        data = self.standards[field]
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            all_vals = []
            for vals in data.values():
                if isinstance(vals, list):
                    all_vals.extend(vals)
            return all_vals
        return []
    
    # ─────────────────────────────────────────────────────────────────────────
    # FIELD EXTRACTION (same as before)
    # ─────────────────────────────────────────────────────────────────────────
    
    def extract_national_id(self, text: str) -> Tuple[Optional[str], Optional[Dict]]:
        text = self._convert_arabic_nums(text)
        clean = re.sub(r'[^\d\s]', ' ', text)
        
        patterns = [r'\b(\d{14})\b', r'(\d[\d\s]{12,20}\d)']
        
        for pattern in patterns:
            matches = re.findall(pattern, clean)
            for match in matches:
                num = re.sub(r'\s', '', match)
                if len(num) == 14 and num[0] in self.standards["national_id"]["century_codes"]:
                    details = self._parse_national_id(num)
                    if details.get("valid"):
                        return num, details
        
        return None, None
    
    def _parse_national_id(self, id_num: str) -> Dict:
        std = self.standards["national_id"]
        details = {"raw": id_num, "valid": False, "errors": []}
        
        if len(id_num) != 14 or not id_num.isdigit():
            details["errors"].append("Invalid length or format")
            return details
        
        century = id_num[0]
        year = id_num[1:3]
        month = id_num[3:5]
        day = id_num[5:7]
        gov_code = id_num[7:9]
        gender_digit = id_num[12]
        
        if century not in std["century_codes"]:
            details["errors"].append(f"Invalid century: {century}")
            return details
        
        full_year = (1900 if century == "2" else 2000) + int(year)
        
        try:
            birth_month = int(month)
            birth_day = int(day)
            
            if not (1 <= birth_month <= 12):
                details["errors"].append(f"Invalid month: {month}")
                return details
            
            if not (1 <= birth_day <= 31):
                details["errors"].append(f"Invalid day: {day}")
                return details
            
            birth_date = date(full_year, birth_month, birth_day)
            age = (self.today - birth_date).days // 365
            
            if not (0 <= age <= 120):
                details["errors"].append(f"Invalid age: {age}")
                return details
            
            details["birth_date"] = birth_date.strftime("%Y-%m-%d")
            details["age"] = age
            
        except ValueError as e:
            details["errors"].append(f"Invalid date: {e}")
            return details
        
        if gov_code in std["governorates"]:
            details["governorate"] = std["governorates"][gov_code]
        else:
            details["governorate"] = "Unknown"
        
        details["gender_from_id"] = "male" if int(gender_digit) % 2 == 1 else "female"
        details["valid"] = True
        return details
    
    def extract_gender(self, text: str) -> Optional[str]:
        all_genders = self._get_all_values("gender")
        return self._find_in_text(text, all_genders)
    
    def extract_religion(self, text: str) -> Optional[str]:
        all_religions = self._get_all_values("religion")
        return self._find_in_text(text, all_religions)
    
    def extract_marital(self, text: str) -> Optional[str]:
        all_marital = self._get_all_values("marital")
        return self._find_in_text(text, all_marital)
    
    def extract_occupation(self, text: str) -> Optional[str]:
        return self._find_in_text(text, self.standards["occupation"])
    
    def extract_validity_phrase(self, text: str) -> Optional[str]:
        return self._find_in_text(text, self.standards["validity_phrases"])
    
    def extract_dates(self, text: str) -> Dict[str, Optional[str]]:
        text = self._convert_arabic_nums(text)
        result = {"issue_date": None, "expiry_date": None, "all_dates": []}
        
        # Full dates
        for match in re.finditer(r'(\d{4})[/\-\.](\d{1,2})[/\-\.](\d{1,2})', text):
            y, m, d = int(match.group(1)), int(match.group(2)), int(match.group(3))
            if 2000 <= y <= 2050 and 1 <= m <= 12 and 1 <= d <= 31:
                result["all_dates"].append({
                    "string": match.group(0), "year": y, "month": m, "day": d, "type": "full"
                })
        
        # Partial dates
        for match in re.finditer(r'(\d{4})[/\-\.](\d{1,2})(?![/\-\.]\d)', text):
            y, m = int(match.group(1)), int(match.group(2))
            if 2000 <= y <= 2050 and 1 <= m <= 12:
                date_str = match.group(0)
                if not any(d["string"].startswith(date_str) for d in result["all_dates"]):
                    result["all_dates"].append({
                        "string": date_str, "year": y, "month": m, "day": None, "type": "partial"
                    })
        
        result["all_dates"].sort(key=lambda d: (d["year"], d["month"], d.get("day", 1)))
        
        if len(result["all_dates"]) >= 2:
            result["issue_date"] = result["all_dates"][0]["string"]
            result["expiry_date"] = result["all_dates"][-1]["string"]
        elif len(result["all_dates"]) == 1:
            d = result["all_dates"][0]
            if d["year"] >= self.today.year:
                result["expiry_date"] = d["string"]
            else:
                result["issue_date"] = d["string"]
        
        return result
    
    def extract_expiry_date(self, text: str) -> Tuple[Optional[date], Optional[str]]:
        text = self._convert_arabic_nums(text)
        
        patterns = [
            r'حت[ىي]\s*(\d{4}[/\-\.]\d{1,2}[/\-\.]\d{1,2})',
            r'حت[ىي]\s*(\d{4}[/\-\.]\d{1,2})',
            r'سارية?\s*(?:حت[ىي])?\s*(\d{4}[/\-\.]\d{1,2}[/\-\.]\d{1,2})',
            r'سارية?\s*(?:حت[ىي])?\s*(\d{4}[/\-\.]\d{1,2})',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                date_str = match.group(1)
                parsed = self._parse_date(date_str)
                if parsed:
                    return parsed, date_str
        
        return None, None
    
    def _parse_date(self, date_str: str) -> Optional[date]:
        if not date_str:
            return None
        
        date_str = self._convert_arabic_nums(date_str)
        date_str = re.sub(r'[^\d/\-\.]', '', date_str)
        
        match = re.match(r'(\d{4})[/\-\.](\d{1,2})[/\-\.](\d{1,2})', date_str)
        if match:
            try:
                return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
            except:
                pass
        
        match = re.match(r'(\d{4})[/\-\.](\d{1,2})', date_str)
        if match:
            try:
                y, m = int(match.group(1)), int(match.group(2))
                if m == 12:
                    return date(y + 1, 1, 1) - timedelta(days=1)
                else:
                    return date(y, m + 1, 1) - timedelta(days=1)
            except:
                pass
        
        return None
    
    def check_expiry(self, expiry_date: Optional[date]) -> ExpiryInfo:
        if not expiry_date:
            return ExpiryInfo(status=ExpiryStatus.UNKNOWN, message="⚪ Could not determine expiry date")
        
        days = (expiry_date - self.today).days
        
        if days < 0:
            return ExpiryInfo(
                status=ExpiryStatus.EXPIRED,
                expiry_date=expiry_date,
                expiry_date_str=expiry_date.strftime('%Y/%m/%d'),
                days_until_expiry=days,
                is_expired=True,
                message=f"⛔ EXPIRED - {abs(days)} days ago ({expiry_date.strftime('%Y/%m/%d')})"
            )
        elif days <= self.expiry_warning_days:
            return ExpiryInfo(
                status=ExpiryStatus.EXPIRING_SOON,
                expiry_date=expiry_date,
                expiry_date_str=expiry_date.strftime('%Y/%m/%d'),
                days_until_expiry=days,
                is_expired=False,
                message=f"⚠️ EXPIRING SOON - {days} days left ({expiry_date.strftime('%Y/%m/%d')})"
            )
        else:
            return ExpiryInfo(
                status=ExpiryStatus.VALID,
                expiry_date=expiry_date,
                expiry_date_str=expiry_date.strftime('%Y/%m/%d'),
                days_until_expiry=days,
                is_expired=False,
                message=f"✅ VALID - {days} days remaining ({expiry_date.strftime('%Y/%m/%d')})"
            )
    
    # ─────────────────────────────────────────────────────────────────────────
    # STRUCTURAL CHECKS
    # ─────────────────────────────────────────────────────────────────────────
    
    def check_aspect_ratio(self, image: np.ndarray) -> Tuple[bool, Dict]:
        h, w = image.shape[:2]
        ratio = w / h if h > 0 else 0
        std = self.standards["card"]
        valid = std["aspect_ratio_min"] <= ratio <= std["aspect_ratio_max"]
        return valid, {"ratio": round(ratio, 3)}
    
    def check_barcode(self, image: np.ndarray) -> Tuple[bool, Dict]:
        region = self.REGIONS['barcode']
        h, w = image.shape[:2]
        
        roi = image[int(h * region[1]):int(h * region[3]),
                    int(w * region[0]):int(w * region[2])]
        
        if roi.shape[0] < 20 or roi.shape[1] < 50:
            return False, {"error": "ROI too small"}
        
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        total = roi.shape[0] * roi.shape[1]
        black_ratio = np.sum(binary == 0) / total
        
        sv = np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3))
        sh = np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3))
        vh_ratio = np.sum(sv) / (np.sum(sh) + 1)
        
        transitions = []
        step = max(1, binary.shape[1] // 50)
        for c in range(0, binary.shape[1], step):
            t = np.sum(np.abs(np.diff(binary[:, c].astype(int))) > 100)
            transitions.append(t)
        avg_trans = np.mean(transitions) if transitions else 0
        
        score = 0
        if 0.10 <= black_ratio <= 0.60:
            score += 0.35
        if vh_ratio > 0.8:
            score += 0.35
        if avg_trans > 3:
            score += 0.30
        
        valid = score >= 0.5
        
        return valid, {
            "black_ratio": round(black_ratio, 3),
            "vh_ratio": round(vh_ratio, 2),
            "avg_transitions": round(avg_trans, 1),
            "score": round(score, 2)
        }
    
    def check_pharaoh_emblem(self, image: np.ndarray) -> Tuple[bool, Dict]:
        """Check Pharaoh emblem using shape-based detection."""
        region = self.REGIONS['pharaoh']
        h, w = image.shape[:2]
        
        roi = image[int(h * region[1]):int(h * region[3]),
                    int(w * region[0]):int(w * region[2])]
        
        valid, score, details = self.emblem_detector.detect_pharaoh_emblem(roi)
        return valid, details
    
    def check_eagle_emblem(self, image: np.ndarray) -> Tuple[bool, Dict]:
        """Check Eagle emblem using shape-based detection."""
        region = self.REGIONS['eagle']
        h, w = image.shape[:2]
        
        roi = image[int(h * region[1]):int(h * region[3]),
                    int(w * region[0]):int(w * region[2])]
        
        valid, score, details = self.emblem_detector.detect_eagle_emblem(roi)
        return valid, details
    
    def check_background(self, image: np.ndarray) -> Tuple[bool, Dict]:
        h, w = image.shape[:2]
        roi = image[int(h * 0.20):int(h * 0.50), int(w * 0.25):int(w * 0.75)]
        
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
        
        brightness = np.mean(gray)
        valid = brightness > 100
        
        return valid, {"brightness": round(float(brightness), 1)}
    
    # ─────────────────────────────────────────────────────────────────────────
    # MAIN VALIDATION
    # ─────────────────────────────────────────────────────────────────────────
    
    def validate(self, image_path: str) -> ValidationResult:
        checks_passed = []
        checks_failed = []
        fields = ExtractedFields()
        expiry_info = ExpiryInfo()
        details = {"check_date": self.today.strftime("%Y-%m-%d")}
        
        # Load image
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Could not load image")
            
            image = self.preprocessor.deskew(image)
            h, w = image.shape[:2]
            details["image_size"] = {"width": w, "height": h}
            
        except Exception as e:
            return ValidationResult(
                status=ValidationStatus.INVALID, confidence=0.0,
                checks_passed=[], checks_failed=["load_failed"],
                fields=fields, expiry_info=expiry_info,
                details={"error": str(e)}
            )
        
        # ═══════════════════════════════════════════════════════════════════
        # STRUCTURAL CHECKS
        # ═══════════════════════════════════════════════════════════════════
        print("    Checking structure...")
        
        ar_valid, ar_details = self.check_aspect_ratio(image)
        details["aspect_ratio"] = ar_details
        (checks_passed if ar_valid else checks_failed).append("aspect_ratio")
        
        bc_valid, bc_details = self.check_barcode(image)
        details["barcode"] = bc_details
        (checks_passed if bc_valid else checks_failed).append("barcode_present")
        
        # Shape-based emblem detection
        ph_valid, ph_details = self.check_pharaoh_emblem(image)
        details["pharaoh"] = ph_details
        (checks_passed if ph_valid else checks_failed).append("pharaoh_emblem")
        
        ea_valid, ea_details = self.check_eagle_emblem(image)
        details["eagle"] = ea_details
        (checks_passed if ea_valid else checks_failed).append("eagle_emblem")
        
        bg_valid, bg_details = self.check_background(image)
        details["background"] = bg_details
        (checks_passed if bg_valid else checks_failed).append("background_color")
        
        # ═══════════════════════════════════════════════════════════════════
        # OCR
        # ═══════════════════════════════════════════════════════════════════
        print("    Running OCR...")
        
        ocr_texts = {}
        for region_name, coords in self.REGIONS.items():
            if region_name in ['pharaoh', 'eagle', 'barcode']:
                continue
            ocr_texts[region_name] = self.ocr.read_region(image, *coords)
        
        all_text = ' '.join(ocr_texts.values())
        fields.raw_ocr_text = all_text
        details["ocr_text_length"] = len(all_text)
        
        # ═══════════════════════════════════════════════════════════════════
        # FIELD EXTRACTION
        # ═══════════════════════════════════════════════════════════════════
        print("    Extracting fields...")
        
        # National ID
        combined_id = f"{ocr_texts.get('national_id', '')} {ocr_texts.get('row1_full', '')} {all_text}"
        national_id, id_details = self.extract_national_id(combined_id)
        fields.national_id = national_id
        
        if national_id and id_details:
            fields.governorate = id_details.get("governorate")
            fields.birth_date = id_details.get("birth_date")
            fields.age = id_details.get("age")
            fields.gender_from_id = id_details.get("gender_from_id")
            checks_passed.append("national_id_format")
        else:
            checks_failed.append("national_id_format")
        
        # Gender
        gender_text = f"{ocr_texts.get('gender', '')} {ocr_texts.get('status_full', '')} {all_text}"
        fields.gender = self.extract_gender(gender_text)
        (checks_passed if fields.gender else checks_failed).append("gender_valid")
        
        # Religion
        religion_text = f"{ocr_texts.get('religion', '')} {ocr_texts.get('status_full', '')} {all_text}"
        fields.religion = self.extract_religion(religion_text)
        (checks_passed if fields.religion else checks_failed).append("religion_valid")
        
        # Marital
        marital_text = f"{ocr_texts.get('marital', '')} {ocr_texts.get('status_full', '')} {all_text}"
        fields.marital_status = self.extract_marital(marital_text)
        (checks_passed if fields.marital_status else checks_failed).append("marital_status_valid")
        
        # Occupation
        occupation_text = f"{ocr_texts.get('occupation', '')} {all_text}"
        fields.occupation = self.extract_occupation(occupation_text)
        (checks_passed if fields.occupation else checks_failed).append("occupation_present")
        
        # ═══════════════════════════════════════════════════════════════════
        # DATES
        # ═══════════════════════════════════════════════════════════════════
        print("    Extracting dates...")
        
        validity_text = f"{ocr_texts.get('validity', '')} {all_text}"
        fields.validity_phrase = self.extract_validity_phrase(validity_text)
        (checks_passed if fields.validity_phrase else checks_failed).append("validity_phrase_present")
        
        date_text = f"{ocr_texts.get('validity', '')} {ocr_texts.get('row1_full', '')} {all_text}"
        dates = self.extract_dates(date_text)
        fields.issue_date = dates.get("issue_date")
        (checks_passed if fields.issue_date else checks_failed).append("issue_date_valid")
        
        expiry_date, expiry_str = self.extract_expiry_date(validity_text)
        if expiry_date:
            fields.expiry_date = expiry_str
            fields.expiry_date_parsed = expiry_date
        elif dates.get("expiry_date"):
            fields.expiry_date = dates.get("expiry_date")
            fields.expiry_date_parsed = self._parse_date(dates.get("expiry_date"))
        
        (checks_passed if fields.expiry_date else checks_failed).append("expiry_date_valid")
        
        # Expiry check
        expiry_info = self.check_expiry(fields.expiry_date_parsed)
        
        if expiry_info.is_expired:
            checks_failed.append("card_not_expired")
        elif expiry_info.status != ExpiryStatus.UNKNOWN:
            checks_passed.append("card_not_expired")
        
        # ═══════════════════════════════════════════════════════════════════
        # CALCULATE SCORE
        # ═══════════════════════════════════════════════════════════════════
        
        weights = {
            "barcode_present": 0.12,
            "national_id_format": 0.15,
            "gender_valid": 0.10,
            "religion_valid": 0.10,
            "marital_status_valid": 0.10,
            "expiry_date_valid": 0.10,
            "issue_date_valid": 0.05,
            "occupation_present": 0.05,
            "validity_phrase_present": 0.05,
            "pharaoh_emblem": 0.05,
            "eagle_emblem": 0.05,
            "background_color": 0.04,
            "aspect_ratio": 0.04,
        }
        
        total_weight = sum(weights.values())
        earned_weight = sum(weights.get(c, 0) for c in checks_passed)
        confidence = earned_weight / total_weight
        
        key_fields = ["gender_valid", "religion_valid", "marital_status_valid"]
        key_count = sum(1 for f in key_fields if f in checks_passed)
        
        has_barcode = "barcode_present" in checks_passed
        has_national_id = "national_id_format" in checks_passed
        
        if expiry_info.is_expired:
            status = ValidationStatus.EXPIRED
        elif confidence >= 0.55 and (has_barcode or has_national_id) and key_count >= 2:
            status = ValidationStatus.VALID
        elif confidence >= 0.30 and (has_barcode or has_national_id or key_count >= 1):
            status = ValidationStatus.NEEDS_REVIEW
        else:
            status = ValidationStatus.INVALID
        
        return ValidationResult(
            status=status,
            confidence=confidence,
            checks_passed=checks_passed,
            checks_failed=checks_failed,
            fields=fields,
            expiry_info=expiry_info,
            details=details
        )


# ============ DRIVER VALIDATOR ============
class DriverValidator:
    def __init__(self, expiry_warning_days: int = 30):
        self.validator = EgyptianIDValidator(expiry_warning_days=expiry_warning_days)
    
    def validate(self, path: str) -> dict:
        result = self.validator.validate(path)
        f = result.fields
        e = result.expiry_info
        
        return {
            'file': os.path.basename(path),
            'status': result.status.value.upper(),
            'confidence': round(result.confidence * 100, 1),
            'checks_passed': result.checks_passed,
            'checks_failed': result.checks_failed,
            'fields': {
                'national_id': f.national_id,
                'governorate': f.governorate,
                'birth_date': f.birth_date,
                'age': f.age,
                'gender': f.gender,
                'gender_from_id': f.gender_from_id,
                'religion': f.religion,
                'marital_status': f.marital_status,
                'occupation': f.occupation,
                'issue_date': f.issue_date,
                'expiry_date': f.expiry_date,
            },
            'expiry': {
                'status': e.status.value,
                'date': e.expiry_date_str,
                'days_remaining': e.days_until_expiry,
                'is_expired': e.is_expired,
                'message': e.message,
            },
            'details': result.details,
        }


# ============ PROCESS FOLDER ============
def process_folder(folder: str, expiry_warning_days: int = 30):
    exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp', '*.tiff']
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(folder, e)))
        files.extend(glob.glob(os.path.join(folder, e.upper())))
    files = list(set(files))
    
    if not files:
        print(f"No images in {folder}")
        return []
    
    print(f"\n{'=' * 80}")
    print("EGYPTIAN NATIONAL ID VALIDATOR (Shape-Based)")
    print(f"{'=' * 80}")
    print(f"📁 Folder: {folder}")
    print(f"🖼️  Images: {len(files)}")
    print(f"📅 Check Date: {CURRENT_DATE}")
    print(f"{'=' * 80}\n")
    
    validator = DriverValidator(expiry_warning_days)
    results = []
    counts = {'VALID': 0, 'INVALID': 0, 'NEEDS_REVIEW': 0, 'EXPIRED': 0}
    
    for i, f in enumerate(sorted(files), 1):
        print(f"\n[{i}/{len(files)}] {os.path.basename(f)}")
        print("-" * 70)
        
        r = validator.validate(f)
        results.append(r)
        counts[r['status']] += 1
        
        icons = {'VALID': '✅', 'NEEDS_REVIEW': '🔍', 'INVALID': '❌', 'EXPIRED': '⛔'}
        print(f"  {icons.get(r['status'], '❓')} {r['status']} | Confidence: {r['confidence']}%")
        print(f"  📅 Expiry: {r['expiry']['message']}")
        
        flds = r['fields']
        if flds['national_id']:
            print(f"  🆔 National ID: {flds['national_id']}")
            if flds['governorate']:
                print(f"      Governorate: {flds['governorate']}")
            if flds['age']:
                print(f"      Age: {flds['age']} (Birth: {flds['birth_date']})")
        if flds['gender']:
            print(f"  👤 Gender: {flds['gender']}")
        if flds['religion']:
            print(f"  🕌 Religion: {flds['religion']}")
        if flds['marital_status']:
            print(f"  💍 Marital: {flds['marital_status']}")
        if flds['occupation']:
            print(f"  💼 Occupation: {flds['occupation']}")
        if flds['issue_date']:
            print(f"  📆 Issue: {flds['issue_date']}")
        if flds['expiry_date']:
            print(f"  📆 Expiry: {flds['expiry_date']}")
        
        if r['checks_failed']:
            print(f"  ⚠️ Failed: {', '.join(r['checks_failed'])}")
    
    # Summary
    print(f"\n{'=' * 80}")
    print("📊 SUMMARY")
    print(f"{'=' * 80}")
    n = len(files)
    for status, count in counts.items():
        icon = {'VALID': '✅', 'NEEDS_REVIEW': '🔍', 'INVALID': '❌', 'EXPIRED': '⛔'}.get(status, '')
        print(f"  {icon} {status}: {count} ({count/n*100:.1f}%)")
    
    # Save report
    import csv
    csv_path = os.path.join(folder, f"report_{CURRENT_DATE.strftime('%Y%m%d')}.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as file:
        w = csv.writer(file)
        w.writerow(['File', 'Status', 'Confidence', 'National ID', 'Governorate', 'Age',
                    'Gender', 'Religion', 'Marital', 'Occupation', 'Issue Date', 'Expiry Date',
                    'Days Remaining', 'Expired', 'Passed', 'Failed'])
        for r in results:
            fl = r['fields']
            ex = r['expiry']
            w.writerow([
                r['file'], r['status'], r['confidence'],
                fl['national_id'] or '', fl['governorate'] or '', fl['age'] or '',
                fl['gender'] or '', fl['religion'] or '', fl['marital_status'] or '',
                fl['occupation'] or '', fl['issue_date'] or '', fl['expiry_date'] or '',
                ex['days_remaining'] if ex['days_remaining'] else '', 'YES' if ex['is_expired'] else 'NO',
                len(r['checks_passed']), len(r['checks_failed'])
            ])
    print(f"\n📄 Report: {csv_path}\n")
    
    return results


# ============ MAIN ============
if __name__ == "__main__":
    FOLDER = r"C:\Users\asus\Downloads\id_checker"
    
    if os.path.exists(FOLDER):
        process_folder(FOLDER, expiry_warning_days=30)
    else:
        print(f"❌ Folder not found: {FOLDER}")