# production_egyptian_id_verifier_enhanced.py
"""
ENHANCED Egyptian National ID Verification System - COMPLETE STANDALONE VERSION
Based on detailed analysis of actual ID specimens

FIXES APPLIED:
- Issue 3: Adaptive color detection with white balance correction
- Issue 4: Global singleton OCR engine initialization
- Bug fix: cv2.circle mask dtype issue
"""

import cv2
import numpy as np
from datetime import datetime
import os
import re
from typing import Dict, Optional, Tuple, List
import glob
from dataclasses import dataclass, field
from enum import Enum
import threading

# ==================== ENHANCED CONFIGURATION ====================
class EnhancedConfig:
    """Enhanced system configuration based on actual ID analysis"""
    
    # Tesseract path
    TESSERACT_PATH = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    
    # ===== PHYSICAL SPECIFICATIONS =====
    ASPECT_RATIO_TARGET = 1.586
    ASPECT_RATIO_TOLERANCE = 0.18
    
    # ===== LAYOUT SPECIFICATIONS =====
    LAYOUT = {
        'photo_region': {
            'x_start': 0.0, 'x_end': 0.32,
            'y_start': 0.08, 'y_end': 0.72
        },
        'emblem_region': {
            'x_start': 0.35, 'x_end': 0.75,
            'y_start': 0.0, 'y_end': 0.25
        },
        'watermark_region': {
            'x_start': 0.20, 'x_end': 0.95,
            'y_start': 0.15, 'y_end': 0.75
        },
        'header_region': {
            'x_start': 0.0, 'x_end': 1.0,
            'y_start': 0.0, 'y_end': 0.18
        },
        'id_number_region': {
            'x_start': 0.12, 'x_end': 0.88,
            'y_start': 0.62, 'y_end': 0.88
        },
        'security_strip': {
            'x_start': 0.0, 'x_end': 1.0,
            'y_start': 0.85, 'y_end': 1.0
        }
    }
    
    # ===== ADAPTIVE COLOR ANALYSIS (FIX FOR ISSUE 3) =====
    # Base reference colors - will be adapted based on lighting
    COLOR_REFERENCES = {
        'background_beige': {
            # Wider ranges to accommodate lighting variation
            'hsv_center': [25, 65, 195],  # Center point
            'hsv_tolerance': [15, 55, 80],  # Tolerance in each channel
            'min_coverage': 0.10,  # Lowered threshold
            'weight': 0.12
        },
        'background_teal': {
            'hsv_center': [90, 100, 155],
            'hsv_tolerance': [20, 80, 90],
            'min_coverage': 0.08,
            'weight': 0.10
        },
        'security_blue': {
            'hsv_center': [110, 145, 155],
            'hsv_tolerance': [20, 110, 100],
            'min_coverage': 0.03,
            'weight': 0.08
        },
        'emblem_gold': {
            'hsv_center': [28, 170, 175],
            'hsv_tolerance': [15, 90, 90],
            'min_coverage': 0.01,
            'weight': 0.06
        },
        'text_fields_white': {
            'hsv_center': [90, 15, 215],
            'hsv_tolerance': [90, 25, 50],
            'min_coverage': 0.05,
            'weight': 0.04
        }
    }
    
    # Lighting adaptation settings
    LIGHTING_ADAPTATION = {
        'enable_white_balance': True,
        'enable_histogram_equalization': True,
        'saturation_boost_range': (0.9, 1.3),
        'value_normalization': True
    }
    
    # ===== TEXT KEYWORDS =====
    ARABIC_KEYWORDS = {
        'header': ['جمهورية', 'مصر', 'العربية', 'بطاقة', 'تحقيق', 'الشخصية'],
        'fields': ['الرقم', 'الاسم', 'العنوان', 'المحافظة', 'تاريخ', 'الميلاد']
    }
    
    ENGLISH_KEYWORDS = {
        'standard': ['egypt', 'arab', 'republic', 'national', 'id', 'card']
    }
    
    # Keywords that indicate this is NOT a National ID (e.g., Driving License)
    DRIVING_LICENSE_KEYWORDS = [
        'رخصة', 'قيادة', 'مرور', 'تسيير', 'خاصة', 'مهنية', 'دراجة', 'مركبة',
        'driving', 'license', 'traffic'
    ]
    
    # ===== FEATURE WEIGHTS =====
    WEIGHTS = {
        'aspect_ratio': 0.06,
        'layout_structure': 0.10,
        'photo_left_side': 0.14,
        'pyramids_sphinx': 0.09,
        'eagle_emblem': 0.11,
        'arabic_header': 0.13,
        'color_scheme': 0.10,
        'security_pattern': 0.07,
        'id_number_valid': 0.20
    }
    
    # ===== THRESHOLDS =====
    CONFIDENCE_THRESHOLD = 0.48
    # Higher threshold required if no valid 14-digit ID number is found
    # This prevents look-alike cards (like driving licenses) from passing weak checks
    CONFIDENCE_THRESHOLD_NO_ID = 0.65 
    ID_NUMBER_OVERRIDE = 0.92
    HIGH_CONFIDENCE = 0.70
    
    # ===== OCR SETTINGS =====
    USE_EASYOCR = True
    USE_TESSERACT = True
    OCR_MIN_CONFIDENCE = 0.30
    
    # ===== DETECTION PARAMETERS =====
    FACE_DETECTION = {
        'scaleFactor': 1.05,
        'minNeighbors': 3,
        'minSize': (20, 20),
        'maxSize': (180, 180)
    }
    
    CIRCLE_DETECTION = {
        'dp': 1,
        'minDist': 20,
        'param1': 50,
        'param2': 22,
        'minRadius': 15,
        'maxRadius': 80
    }
    
    EDGE_DETECTION = {
        'canny_low': 30,
        'canny_high': 120,
        'hough_threshold': 28,
        'min_line_length': 35,
        'max_line_gap': 18
    }


# ==================== DATACLASS ====================
@dataclass
class FeatureResult:
    """Result of a single feature check"""
    passed: bool
    score: float
    message: str
    details: Dict = field(default_factory=dict)


# ==================== LIGHTING CONDITION ESTIMATOR (FIX FOR ISSUE 3) ====================
class LightingConditionEstimator:
    """Estimates and compensates for different lighting conditions"""
    
    class LightingType(Enum):
        DAYLIGHT = "daylight"
        TUNGSTEN = "tungsten"  # Warm yellow
        FLUORESCENT = "fluorescent"  # Cool green tint
        LED_WARM = "led_warm"
        LED_COOL = "led_cool"
        UNKNOWN = "unknown"
    
    @staticmethod
    def estimate_lighting(image: np.ndarray) -> Tuple['LightingConditionEstimator.LightingType', Dict]:
        """Analyze image to estimate lighting conditions"""
        if len(image.shape) != 3:
            return LightingConditionEstimator.LightingType.UNKNOWN, {}
        
        # Convert to different color spaces for analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Analyze color temperature using LAB
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        # b_channel: negative = blue (cool), positive = yellow (warm)
        avg_b = np.mean(b_channel)
        # a_channel: negative = green, positive = magenta
        avg_a = np.mean(a_channel)
        
        # Analyze value distribution
        h, s, v = cv2.split(hsv)
        avg_saturation = np.mean(s)
        avg_value = np.mean(v)
        
        # Determine lighting type
        lighting_info = {
            'avg_b_channel': float(avg_b),
            'avg_a_channel': float(avg_a),
            'avg_saturation': float(avg_saturation),
            'avg_value': float(avg_value),
            'color_temp_shift': float(avg_b - 128),  # Deviation from neutral
        }
        
        # Classification logic
        if avg_b > 145:  # Strong yellow cast
            lighting_type = LightingConditionEstimator.LightingType.TUNGSTEN
        elif avg_b < 115:  # Strong blue cast
            lighting_type = LightingConditionEstimator.LightingType.LED_COOL
        elif avg_a < 120:  # Green tint
            lighting_type = LightingConditionEstimator.LightingType.FLUORESCENT
        elif 125 <= avg_b <= 135 and 125 <= avg_a <= 135:
            lighting_type = LightingConditionEstimator.LightingType.DAYLIGHT
        else:
            lighting_type = LightingConditionEstimator.LightingType.LED_WARM
        
        return lighting_type, lighting_info
    
    @staticmethod
    def apply_white_balance(image: np.ndarray) -> np.ndarray:
        """Apply automatic white balance correction using Gray World algorithm"""
        if len(image.shape) != 3:
            return image
        
        result = image.copy().astype(np.float32)
        
        # Gray World assumption: average color should be gray
        avg_b = np.mean(result[:, :, 0])
        avg_g = np.mean(result[:, :, 1])
        avg_r = np.mean(result[:, :, 2])
        
        # Calculate the average gray value
        avg_gray = (avg_b + avg_g + avg_r) / 3
        
        # Scale each channel
        if avg_b > 0:
            result[:, :, 0] = result[:, :, 0] * (avg_gray / avg_b)
        if avg_g > 0:
            result[:, :, 1] = result[:, :, 1] * (avg_gray / avg_g)
        if avg_r > 0:
            result[:, :, 2] = result[:, :, 2] * (avg_gray / avg_r)
        
        # Clip and convert back
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        return result
    
    @staticmethod
    def apply_advanced_white_balance(image: np.ndarray) -> np.ndarray:
        """Apply white balance using white patch algorithm combined with gray world"""
        if len(image.shape) != 3:
            return image
        
        # First apply gray world
        balanced = LightingConditionEstimator.apply_white_balance(image)
        
        # Then apply a mild white patch correction
        # Find the brightest pixels (top 1%)
        gray = cv2.cvtColor(balanced, cv2.COLOR_BGR2GRAY)
        threshold = np.percentile(gray, 99)
        bright_mask = gray >= threshold
        
        if np.sum(bright_mask) > 10:
            result = balanced.astype(np.float32)
            
            # Get average of bright pixels
            bright_b = np.mean(result[:, :, 0][bright_mask])
            bright_g = np.mean(result[:, :, 1][bright_mask])
            bright_r = np.mean(result[:, :, 2][bright_mask])
            
            max_bright = max(bright_b, bright_g, bright_r)
            
            if max_bright > 0:
                # Apply mild correction (blend with original)
                scale_b = max_bright / bright_b if bright_b > 0 else 1
                scale_g = max_bright / bright_g if bright_g > 0 else 1
                scale_r = max_bright / bright_r if bright_r > 0 else 1
                
                # Blend scales toward 1.0 to avoid over-correction
                blend = 0.5
                scale_b = 1 + (scale_b - 1) * blend
                scale_g = 1 + (scale_g - 1) * blend
                scale_r = 1 + (scale_r - 1) * blend
                
                result[:, :, 0] *= scale_b
                result[:, :, 1] *= scale_g
                result[:, :, 2] *= scale_r
                
                balanced = np.clip(result, 0, 255).astype(np.uint8)
        
        return balanced
    
    @staticmethod
    def normalize_image(image: np.ndarray) -> np.ndarray:
        """Normalize image for consistent color analysis"""
        if len(image.shape) != 3:
            return image
        
        # Apply white balance
        normalized = LightingConditionEstimator.apply_advanced_white_balance(image)
        
        # Optionally normalize value channel
        hsv = cv2.cvtColor(normalized, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Apply CLAHE to value channel for consistent brightness
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        v = clahe.apply(v)
        
        hsv = cv2.merge([h, s, v])
        normalized = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        return normalized


# ==================== ADAPTIVE COLOR DETECTOR (FIX FOR ISSUE 3) ====================
class AdaptiveColorDetector:
    """Adaptive color detection that handles various lighting conditions"""
    
    def __init__(self, config: EnhancedConfig):
        self.config = config
        self.lighting_estimator = LightingConditionEstimator()
    
    def get_adaptive_color_ranges(self, image: np.ndarray) -> Dict:
        """Calculate adaptive color ranges based on image lighting"""
        lighting_type, lighting_info = self.lighting_estimator.estimate_lighting(image)
        
        # Start with reference colors
        adapted_ranges = {}
        
        for color_name, color_spec in self.config.COLOR_REFERENCES.items():
            center = np.array(color_spec['hsv_center'], dtype=np.float32)
            tolerance = np.array(color_spec['hsv_tolerance'], dtype=np.float32)
            
            # Adjust based on lighting type
            if lighting_type == LightingConditionEstimator.LightingType.TUNGSTEN:
                # Warm light: shift hue toward yellow, increase tolerance
                if color_name in ['background_beige', 'emblem_gold']:
                    center[0] = min(center[0] + 5, 179)  # Shift hue
                tolerance *= 1.2  # Increase tolerance
                
            elif lighting_type == LightingConditionEstimator.LightingType.LED_COOL:
                # Cool light: shift hue toward blue
                if color_name in ['background_teal', 'security_blue']:
                    center[0] = max(center[0] - 5, 0)
                tolerance *= 1.2
                
            elif lighting_type == LightingConditionEstimator.LightingType.FLUORESCENT:
                # Green tint: adjust all colors
                tolerance *= 1.3  # More tolerance for green cast
            
            # Apply saturation/value adjustments based on overall image
            if lighting_info.get('avg_saturation', 128) < 80:
                # Low saturation image: widen saturation tolerance
                tolerance[1] *= 1.5
            
            if lighting_info.get('avg_value', 128) < 100:
                # Dark image: widen value tolerance downward
                tolerance[2] *= 1.3
            elif lighting_info.get('avg_value', 128) > 180:
                # Bright image: widen value tolerance upward
                tolerance[2] *= 1.2
            
            # Calculate final ranges
            lower = np.clip(center - tolerance, [0, 0, 0], [179, 255, 255]).astype(np.int32)
            upper = np.clip(center + tolerance, [0, 0, 0], [179, 255, 255]).astype(np.int32)
            
            adapted_ranges[color_name] = {
                'hsv_lower': lower.tolist(),
                'hsv_upper': upper.tolist(),
                'min_coverage': color_spec['min_coverage'] * 0.8,  # Slightly lower threshold
                'weight': color_spec['weight']
            }
        
        return adapted_ranges
    
    def analyze_colors(self, image: np.ndarray, use_normalization: bool = True) -> Dict:
        """Analyze colors with adaptive detection"""
        if len(image.shape) != 3:
            return {'error': 'Grayscale image', 'colors_detected': [], 'total_score': 0}
        
        # Optionally normalize image first
        if use_normalization and self.config.LIGHTING_ADAPTATION['enable_white_balance']:
            normalized_image = self.lighting_estimator.normalize_image(image)
        else:
            normalized_image = image
        
        # Get adaptive ranges
        adaptive_ranges = self.get_adaptive_color_ranges(image)
        
        # Convert to HSV
        hsv = cv2.cvtColor(normalized_image, cv2.COLOR_BGR2HSV)
        
        color_scores = {}
        colors_detected = []
        
        for color_name, color_spec in adaptive_ranges.items():
            mask = cv2.inRange(hsv,
                               np.array(color_spec['hsv_lower']),
                               np.array(color_spec['hsv_upper']))
            coverage = np.sum(mask > 0) / mask.size
            
            color_scores[color_name] = {
                'coverage': float(coverage),
                'passed': coverage >= color_spec['min_coverage'],
                'min_required': color_spec['min_coverage']
            }
            
            if coverage >= color_spec['min_coverage']:
                colors_detected.append(color_name.replace('_', ' '))
        
        # Calculate total score
        total_score = 0.0
        for color_name, result in color_scores.items():
            spec = adaptive_ranges[color_name]
            if result['passed']:
                total_score += spec['weight']
            elif result['coverage'] >= spec['min_coverage'] * 0.5:
                total_score += spec['weight'] * 0.5
        
        max_possible = sum(spec['weight'] for spec in adaptive_ranges.values())
        normalized_score = total_score / max_possible if max_possible > 0 else 0
        
        return {
            'colors_detected': colors_detected,
            'color_scores': color_scores,
            'total_score': total_score,
            'normalized_score': normalized_score,
            'adaptive_ranges': adaptive_ranges
        }
    
    def detect_color_relationships(self, image: np.ndarray) -> Dict:
        """Detect relative color relationships rather than absolute values"""
        if len(image.shape) != 3:
            return {'valid': False, 'reason': 'Grayscale image'}
        
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Analyze hue histogram
        hue_hist = cv2.calcHist([h], [0], None, [180], [0, 180]).flatten()
        hist_sum = hue_hist.sum()
        if hist_sum > 0:
            hue_hist = hue_hist / hist_sum  # Normalize
        
        # Look for expected peaks
        # Egyptian ID typically has: beige (15-35), teal (75-105), blue (95-125)
        
        beige_presence = float(np.sum(hue_hist[15:35]))
        teal_presence = float(np.sum(hue_hist[75:105]))
        blue_presence = float(np.sum(hue_hist[95:125]))
        
        # Check for multi-modal distribution (indicates ID card colors)
        peaks = []
        for i in range(5, 175):
            if hue_hist[i] > 0.02:
                left_max = hue_hist[max(0, i-5):i].max() if i > 0 else 0
                right_max = hue_hist[i+1:min(180, i+6)].max() if i < 179 else 0
                if hue_hist[i] > left_max and hue_hist[i] > right_max:
                    peaks.append(i)
        
        # Valid ID should have at least 2 distinct color peaks
        valid_color_distribution = len(peaks) >= 2
        
        return {
            'valid': valid_color_distribution,
            'beige_presence': beige_presence,
            'teal_presence': teal_presence,
            'blue_presence': blue_presence,
            'hue_peaks': peaks,
            'num_peaks': len(peaks)
        }


# ==================== GLOBAL OCR ENGINE SINGLETON (FIX FOR ISSUE 4) ====================
class OCREngineSingleton:
    """
    Thread-safe singleton OCR engine.
    Initializes OCR models once at module load, not per-request.
    """
    _instance = None
    _lock = threading.Lock()
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        # Only initialize once
        if OCREngineSingleton._initialized:
            return
        
        with OCREngineSingleton._lock:
            if OCREngineSingleton._initialized:
                return
            
            self.engines = []
            self.reader = None  # EasyOCR reader
            self._tesseract_available = False
            
            self._init_tesseract()
            self._init_easyocr()
            
            if not self.engines:
                print("[WARN] WARNING: No OCR engines available!")
            
            OCREngineSingleton._initialized = True
    
    def _init_tesseract(self):
        if not EnhancedConfig.USE_TESSERACT:
            return
        
        try:
            import pytesseract
            
            if os.path.exists(EnhancedConfig.TESSERACT_PATH):
                pytesseract.pytesseract.tesseract_cmd = EnhancedConfig.TESSERACT_PATH
            
            version = pytesseract.get_tesseract_version()
            self.engines.append('tesseract')
            self._tesseract_available = True
            print(f"[OK] Tesseract OCR v{version}")
        except Exception as e:
            print(f"[WARN] Tesseract not available: {str(e)[:50]}")
    
    def _init_easyocr(self):
        if not EnhancedConfig.USE_EASYOCR:
            return
        
        try:
            import easyocr
            print("[OK] Loading EasyOCR (Arabic + English)... This may take a moment on first load.")
            self.reader = easyocr.Reader(['ar', 'en'], gpu=False, verbose=False)
            self.engines.append('easyocr')
            print("[OK] EasyOCR ready")
        except Exception as e:
            print(f"[WARN] EasyOCR not available: {str(e)[:50]}")
    
    def is_available(self) -> bool:
        """Check if any OCR engine is available"""
        return len(self.engines) > 0
    
    def extract_text(self, image: np.ndarray) -> str:
        """Extract text using all available engines"""
        all_text = []
        
        if 'easyocr' in self.engines and self.reader is not None:
            text = self._extract_easyocr(image)
            all_text.append(text)
        
        if 'tesseract' in self.engines:
            text = self._extract_tesseract(image)
            all_text.append(text)
        
        return '\n'.join(filter(None, all_text))
    
    def _extract_easyocr(self, image: np.ndarray) -> str:
        try:
            results = self.reader.readtext(image, detail=1)
            texts = [text for (bbox, text, conf) in results if conf > 0.25]
            return '\n'.join(texts)
        except Exception as e:
            print(f"EasyOCR error: {e}")
            return ""
    
    def _extract_tesseract(self, image: np.ndarray) -> str:
        try:
            import pytesseract
            
            preprocessed = self._preprocess_for_ocr(image)
            all_text = []
            
            for name, img in preprocessed[:4]:
                try:
                    text = pytesseract.image_to_string(img, lang='ara+eng', config='--oem 3 --psm 6')
                    all_text.append(text)
                except:
                    pass
                
                try:
                    text = pytesseract.image_to_string(img, lang='eng', 
                                                      config='--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789')
                    all_text.append(text)
                except:
                    pass
            
            return '\n'.join(filter(None, all_text))
        except Exception as e:
            print(f"Tesseract error: {e}")
            return ""
    
    def _preprocess_for_ocr(self, image: np.ndarray) -> List[Tuple[str, np.ndarray]]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        results = []
        results.append(("original", gray))
        
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        results.append(("otsu", binary))
        
        adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 11, 2)
        results.append(("adaptive", adaptive))
        
        denoised = cv2.fastNlMeansDenoising(gray, h=10)
        _, denoised_binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        results.append(("denoised", denoised_binary))
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        results.append(("clahe", enhanced))
        
        kernel = np.ones((2, 2), np.uint8)
        morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        results.append(("morph", morph))
        
        return results


# ==================== GLOBAL OCR INSTANCE ====================
# Initialize OCR engine once at module load time
# This avoids reloading models on every request
def get_ocr_engine() -> OCREngineSingleton:
    """Get the global OCR engine singleton"""
    return OCREngineSingleton()


# Pre-initialize on module load (optional - can be lazy)
_global_ocr_engine: Optional[OCREngineSingleton] = None


def initialize_ocr_engine():
    """Explicitly initialize the OCR engine (call once at startup)"""
    global _global_ocr_engine
    if _global_ocr_engine is None:
        print("\n" + "="*50)
        print("Initializing OCR Engine (one-time load)...")
        print("="*50)
        _global_ocr_engine = get_ocr_engine()
        print("="*50 + "\n")
    return _global_ocr_engine


# ==================== ID VALIDATOR ====================
class EgyptianIDValidator:
    """Validate 14-digit Egyptian National ID format"""
    
    GOVERNORATES = {
        '01': 'Cairo', '02': 'Alexandria', '03': 'Port Said', '04': 'Suez',
        '11': 'Damietta', '12': 'Dakahlia', '13': 'Ash Sharqia', '14': 'Kaliobeya',
        '15': 'Kafr El-Sheikh', '16': 'Gharbia', '17': 'Menoufia', '18': 'Beheira',
        '19': 'Ismailia', '21': 'Giza', '22': 'Beni Suef', '23': 'Fayoum',
        '24': 'Menia', '25': 'Assiut', '26': 'Sohag', '27': 'Qena',
        '28': 'Aswan', '29': 'Luxor', '31': 'Red Sea', '32': 'New Valley',
        '33': 'Matrouh', '34': 'North Sinai', '35': 'South Sinai', '88': 'Foreign'
    }
    
    def validate(self, id_number: str) -> Dict:
        """Complete ID validation"""
        result = {
            'valid': False,
            'id_number': id_number,
            'errors': [],
            'info': {}
        }
        
        # Clean ID
        id_clean = id_number.replace(' ', '').replace('-', '').replace('_', '')
        id_clean = id_clean.replace('O', '0').replace('o', '0')
        id_clean = id_clean.replace('I', '1').replace('l', '1').replace('|', '1')
        id_clean = ''.join(c for c in id_clean if c.isdigit())
        
        if len(id_clean) != 14:
            result['errors'].append(f"Must be 14 digits, got {len(id_clean)}")
            return result
        
        result['id_number'] = id_clean
        
        if id_clean[0] not in ['2', '3']:
            result['errors'].append("Must start with 2 or 3")
            return result
        
        try:
            century = '19' if id_clean[0] == '2' else '20'
            year = int(century + id_clean[1:3])
            month = int(id_clean[3:5])
            day = int(id_clean[5:7])
            
            if not (1 <= month <= 12):
                result['errors'].append(f"Invalid month: {month}")
                return result
            
            if not (1 <= day <= 31):
                result['errors'].append(f"Invalid day: {day}")
                return result
            
            birth_date = datetime(year, month, day)
            
            if birth_date > datetime.now():
                result['errors'].append("Birth date in future")
                return result
            
            age = (datetime.now() - birth_date).days // 365
            if age < 0 or age > 120:
                result['errors'].append(f"Unrealistic age: {age}")
                return result
            
        except ValueError as e:
            result['errors'].append(f"Invalid date: {e}")
            return result
        
        gov_code = id_clean[7:9]
        governorate = self.GOVERNORATES.get(gov_code)
        
        if not governorate:
            result['errors'].append(f"Invalid governorate code: {gov_code}")
            return result
        
        gender = "Male" if int(id_clean[-1]) % 2 == 1 else "Female"
        
        result['valid'] = True
        result['info'] = {
            'birth_date': birth_date.strftime('%Y-%m-%d'),
            'age': age,
            'governorate': governorate,
            'governorate_code': gov_code,
            'gender': gender,
            'sequence': id_clean[9:13],
            'check_digit': id_clean[-1]
        }
        
        return result


# ==================== ENHANCED FEATURE DETECTOR ====================
class EnhancedEgyptianIDFeatureDetector:
    """Enhanced detector with precise layout verification and adaptive color detection"""
    
    def __init__(self, ocr_engine: Optional[OCREngineSingleton] = None):
        """
        Initialize detector with optional shared OCR engine.
        
        Args:
            ocr_engine: Pre-initialized OCR engine singleton. If None, will get/create global instance.
        """
        self.validator = EgyptianIDValidator()
        self.config = EnhancedConfig()
        
        # Use provided OCR engine or get global singleton
        self.ocr = ocr_engine if ocr_engine is not None else get_ocr_engine()
        
        # Initialize adaptive color detector
        self.color_detector = AdaptiveColorDetector(self.config)
        self.lighting_estimator = LightingConditionEstimator()
    
    def verify_all_features(self, image: np.ndarray) -> Dict:
        """Run all enhanced feature checks"""
        
        print("\n" + "="*70)
        print("[SEARCH] ENHANCED EGYPTIAN NATIONAL ID FEATURE VERIFICATION")
        print("="*70)
        
        # Estimate and report lighting conditions
        lighting_type, lighting_info = self.lighting_estimator.estimate_lighting(image)
        print(f"[INFO] Detected lighting: {lighting_type.value}")
        print(f"   Color temperature shift: {lighting_info.get('color_temp_shift', 0):.1f}")
        
        results = {}
        
        results['aspect_ratio'] = self._check_aspect_ratio(image)
        results['layout_structure'] = self._verify_layout_structure(image)
        results['photo_left_side'] = self._detect_photo_left(image)
        results['pyramids_sphinx'] = self._detect_pyramids_sphinx(image)
        results['eagle_emblem'] = self._detect_eagle_emblem(image)
        results['arabic_header'] = self._detect_arabic_header(image)
        results['color_scheme'] = self._verify_color_scheme_adaptive(image)
        results['security_pattern'] = self._detect_security_pattern_adaptive(image)
        results['arabic_header'] = self._detect_arabic_header(image)
        results['color_scheme'] = self._verify_color_scheme_adaptive(image)
        results['security_pattern'] = self._detect_security_pattern_adaptive(image)
        results['id_number_valid'] = self._extract_and_validate_id(image)
        
        # Check for Driving License specific keywords to explicitly reject
        is_driving_license = self._detect_driving_license(image)
        if is_driving_license['detected']:
            print(f"[WARN] Detected Driving License keywords: {is_driving_license['keywords']}")
        
        # Print results
        for feature_name, feature_result in results.items():
            icon = "[OK]" if feature_result.passed else "[X]"
            weight = self.config.WEIGHTS[feature_name]
            print(f"{icon} {feature_name.replace('_', ' ').title():25} "
                  f"[{feature_result.score:.2f}] (weight: {weight:.2f}) - {feature_result.message}")
        
        # Calculate confidence
        confidence = sum(
            results[feature].score * self.config.WEIGHTS[feature]
            for feature in self.config.WEIGHTS.keys()
        )
        
        # Decision logic
        print("\n" + "="*70)
        
        # Determine which threshold to use
        if results['id_number_valid'].score >= 0.8:
            # If we have a valid ID number, use standard threshold
            active_threshold = self.config.CONFIDENCE_THRESHOLD
            threshold_name = "Standard"
        else:
            # If NO valid ID number, require much higher visual confidence
            active_threshold = self.config.CONFIDENCE_THRESHOLD_NO_ID
            threshold_name = "Strict (No ID Number)"
            
        if results['id_number_valid'].score >= self.config.ID_NUMBER_OVERRIDE:
            is_egyptian_id = True
            reason = "[PASS] Valid Egyptian ID number verified"
        elif is_driving_license['detected']:
             is_egyptian_id = False
             reason = f"[X] Rejected: Detected Driving License ({', '.join(is_driving_license['keywords'])})"
        elif confidence >= self.config.HIGH_CONFIDENCE:
            is_egyptian_id = True
            reason = f"[PASS] High confidence match ({confidence*100:.1f}%)"
        elif confidence >= active_threshold:
            is_egyptian_id = True
            reason = f"[PASS] Features verified ({confidence*100:.1f}% > {active_threshold*100:.0f}% {threshold_name})"
        else:
            is_egyptian_id = False
            reason = f"[X] Low confidence ({confidence*100:.1f}% < {active_threshold*100:.0f}% {threshold_name})"
            failed = [name for name, res in results.items() if not res.passed]
            if failed:
                reason += f"\n   Failed: {', '.join(failed[:4])}"
        
        print(reason)
        print(f"Overall Confidence: {confidence*100:.1f}%")
        print("="*70 + "\n")
        
        return {
            'is_egyptian_national_id': is_egyptian_id,
            'confidence': confidence,
            'reason': reason,
            'lighting_conditions': {
                'type': lighting_type.value,
                'info': lighting_info
            },
            'features': {name: {
                'passed': res.passed,
                'score': res.score,
                'message': res.message,
                'details': res.details
            } for name, res in results.items()},
            'extracted_data': results['id_number_valid'].details if results['id_number_valid'].details else {}
        }
    
    def _get_region(self, image: np.ndarray, region_name: str) -> np.ndarray:
        """Extract region based on layout specification"""
        h, w = image.shape[:2]
        region = self.config.LAYOUT[region_name]
        
        y1 = int(h * region['y_start'])
        y2 = int(h * region['y_end'])
        x1 = int(w * region['x_start'])
        x2 = int(w * region['x_end'])
        
        return image[y1:y2, x1:x2]
    
    def _check_aspect_ratio(self, image: np.ndarray) -> FeatureResult:
        h, w = image.shape[:2]
        aspect = w / h
        target = self.config.ASPECT_RATIO_TARGET
        distance = abs(aspect - target) / target
        
        if distance < 0.03:
            score = 1.0
        elif distance < 0.07:
            score = 0.85
        elif distance < 0.12:
            score = 0.65
        elif distance < self.config.ASPECT_RATIO_TOLERANCE:
            score = 0.45
        else:
            score = 0.20
        
        passed = score >= 0.40
        message = f"{aspect:.3f} (target: {target:.3f}, error: {distance*100:.1f}%)"
        
        return FeatureResult(passed, score, message, {'aspect': aspect})
    
    def _verify_layout_structure(self, image: np.ndarray) -> FeatureResult:
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        checks = {}
        
        # Photo on LEFT
        photo_region = self._get_region(gray, 'photo_region')
        photo_variance = np.var(photo_region)
        checks['photo_left'] = photo_variance > 650
        
        # Text on RIGHT
        right_region = gray[:int(h*0.6), int(w*0.40):]
        right_variance = np.var(right_region)
        checks['text_right'] = 300 < right_variance < 1500
        
        # Header at TOP
        top_region = gray[:int(h*0.15), :]
        top_edges = cv2.Canny(top_region, 30, 100)
        top_edge_density = np.sum(top_edges > 0) / top_edges.size
        checks['header_top'] = top_edge_density > 0.02
        
        # Security at BOTTOM - using adaptive color detection
        bottom_region = self._get_region(image, 'security_strip')
        if len(bottom_region.shape) == 3:
            # Use adaptive detection instead of fixed ranges
            color_analysis = self.color_detector.analyze_colors(bottom_region)
            blue_detected = 'security blue' in color_analysis.get('colors_detected', [])
            
            # Fallback to simple check
            if not blue_detected:
                hsv_bottom = cv2.cvtColor(bottom_region, cv2.COLOR_BGR2HSV)
                # Use wider range
                blue_mask = cv2.inRange(hsv_bottom, 
                                       np.array([85, 20, 40]),  # Wider range
                                       np.array([135, 255, 255]))
                blue_ratio = np.sum(blue_mask > 0) / blue_mask.size
                checks['security_bottom'] = blue_ratio > 0.04
            else:
                checks['security_bottom'] = True
        else:
            checks['security_bottom'] = False
        
        # Asymmetry check
        left_half = gray[:, :w//2]
        right_half = gray[:, w//2:]
        if left_half.shape == right_half.shape:
            diff = np.mean(cv2.absdiff(left_half, right_half))
            checks['asymmetric'] = diff > 25
        else:
            checks['asymmetric'] = True
        
        passed_count = sum(checks.values())
        score = passed_count / len(checks)
        passed = score >= 0.6
        
        message = f"{passed_count}/{len(checks)} layout checks passed"
        
        return FeatureResult(passed, score, message, checks)
    
    def _detect_photo_left(self, image: np.ndarray) -> FeatureResult:
        photo_region = self._get_region(image, 'photo_region')
        
        if len(photo_region.shape) == 3:
            gray_photo = cv2.cvtColor(photo_region, cv2.COLOR_BGR2GRAY)
        else:
            gray_photo = photo_region
        
        # Face detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(
            gray_photo, 
            scaleFactor=self.config.FACE_DETECTION['scaleFactor'],
            minNeighbors=self.config.FACE_DETECTION['minNeighbors'],
            minSize=self.config.FACE_DETECTION['minSize'],
            maxSize=self.config.FACE_DETECTION['maxSize']
        )
        
        variance = np.var(gray_photo)
        
        edges = cv2.Canny(gray_photo, 40, 120)
        edge_density = np.sum(edges > 0) / edges.size
        
        if len(faces) > 0:
            score = 1.0
            message = f"{len(faces)} face(s) detected"
            passed = True
        elif variance > 800 and edge_density > 0.06:
            score = 0.8
            message = f"High-detail photo region (var:{variance:.0f})"
            passed = True
        elif variance > 650:
            score = 0.6
            message = f"Possible photo (variance: {variance:.0f})"
            passed = True
        else:
            score = 0.3
            message = "No photo detected"
            passed = False
        
        return FeatureResult(passed, score, message,
                            {'faces': len(faces), 'variance': float(variance), 'edge_density': float(edge_density)})
    
    def _detect_pyramids_sphinx(self, image: np.ndarray) -> FeatureResult:
        watermark_region = self._get_region(image, 'watermark_region')
        
        if len(watermark_region.shape) == 3:
            gray = cv2.cvtColor(watermark_region, cv2.COLOR_BGR2GRAY)
        else:
            gray = watermark_region
        
        edges = cv2.Canny(gray, self.config.EDGE_DETECTION['canny_low'],
                         self.config.EDGE_DETECTION['canny_high'])
        
        lines = cv2.HoughLinesP(edges, 1, np.pi/180,
                               threshold=self.config.EDGE_DETECTION['hough_threshold'],
                               minLineLength=self.config.EDGE_DETECTION['min_line_length'],
                               maxLineGap=self.config.EDGE_DETECTION['max_line_gap'])
        
        if lines is None:
            return FeatureResult(False, 0.2, "No geometric structure", {})
        
        diagonal_lines = []
        horizontal_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            
            if 30 < angle < 60 or 120 < angle < 150:
                diagonal_lines.append(angle)
            elif angle < 20 or angle > 160:
                horizontal_lines.append(angle)
        
        diagonal_count = len(diagonal_lines)
        horizontal_count = len(horizontal_lines)
        
        if diagonal_count >= 10 and horizontal_count >= 3:
            score = 0.95
            message = f"Strong pyramid pattern ({diagonal_count}D, {horizontal_count}H)"
            passed = True
        elif diagonal_count >= 6:
            score = 0.75
            message = f"Pyramid structure ({diagonal_count} diagonals)"
            passed = True
        elif diagonal_count >= 3:
            score = 0.50
            message = f"Weak pattern ({diagonal_count} lines)"
            passed = True
        else:
            score = 0.25
            message = f"Insufficient structure"
            passed = False
        
        return FeatureResult(passed, score, message,
                            {'diagonals': diagonal_count, 'horizontals': horizontal_count})
    
    def _detect_eagle_emblem(self, image: np.ndarray) -> FeatureResult:
        emblem_region = self._get_region(image, 'emblem_region')
        
        if len(emblem_region.shape) == 3:
            gray = cv2.cvtColor(emblem_region, cv2.COLOR_BGR2GRAY)
        else:
            gray = emblem_region
        
        circles = cv2.HoughCircles(
            gray, 
            cv2.HOUGH_GRADIENT,
            dp=self.config.CIRCLE_DETECTION['dp'],
            minDist=self.config.CIRCLE_DETECTION['minDist'],
            param1=self.config.CIRCLE_DETECTION['param1'],
            param2=self.config.CIRCLE_DETECTION['param2'],
            minRadius=self.config.CIRCLE_DETECTION['minRadius'],
            maxRadius=self.config.CIRCLE_DETECTION['maxRadius']
        )
        
        has_circle = circles is not None and len(circles[0]) > 0
        
        # Gold color check with adaptive detection
        gold_ratio = 0.0
        if len(emblem_region.shape) == 3:
            # Normalize for lighting
            normalized = self.lighting_estimator.normalize_image(emblem_region)
            hsv_emblem = cv2.cvtColor(normalized, cv2.COLOR_BGR2HSV)
            
            # Wider gold range
            gold_mask = cv2.inRange(hsv_emblem,
                                   np.array([10, 60, 80]),  # Wider range
                                   np.array([45, 255, 255]))
            gold_ratio = float(np.sum(gold_mask > 0) / gold_mask.size)
        
        # Symmetry check
        is_symmetrical = False
        if gray.shape[1] > 50:
            mid = gray.shape[1] // 2
            left = gray[:, :mid]
            right = cv2.flip(gray[:, mid:mid*2], 1) if mid*2 <= gray.shape[1] else cv2.flip(gray[:, mid:], 1)
            
            min_w = min(left.shape[1], right.shape[1])
            if min_w > 10:
                left_crop = left[:, :min_w]
                right_crop = right[:, :min_w]
                
                if left_crop.shape == right_crop.shape:
                    diff = np.mean(cv2.absdiff(left_crop, right_crop))
                    is_symmetrical = diff < 40
        
        if has_circle and (gold_ratio > 0.02 or is_symmetrical):
            score = 1.0
            message = "Eagle emblem detected"
            passed = True
        elif has_circle:
            score = 0.80
            message = "Circular emblem found"
            passed = True
        elif gold_ratio > 0.04:
            score = 0.65
            message = f"Gold emblem ({gold_ratio:.2%})"
            passed = True
        elif is_symmetrical:
            score = 0.55
            message = "Symmetrical pattern"
            passed = True
        else:
            score = 0.25
            message = "No emblem"
            passed = False
        
        return FeatureResult(passed, score, message,
                            {'circle': has_circle, 'gold_ratio': gold_ratio, 'symmetry': is_symmetrical})
    
    def _detect_arabic_header(self, image: np.ndarray) -> FeatureResult:
        header_region = self._get_region(image, 'header_region')
        
        text = self.ocr.extract_text(header_region)
        text_lower = text.lower()
        
        arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
        
        arabic_found = [kw for kw in self.config.ARABIC_KEYWORDS['header'] if kw in text]
        english_found = [kw for kw in self.config.ENGLISH_KEYWORDS['standard'] if kw in text_lower]
        
        total_keywords = len(arabic_found) + len(english_found)
        
        if arabic_chars > 30 and total_keywords >= 4:
            score = 1.0
            message = f"Full Arabic header ({total_keywords} keywords)"
            passed = True
        elif arabic_chars > 20 and total_keywords >= 2:
            score = 0.80
            message = f"Arabic header ({arabic_chars} chars, {total_keywords} kw)"
            passed = True
        elif arabic_chars > 10 or total_keywords >= 2:
            score = 0.60
            message = f"Partial header ({arabic_chars} chars)"
            passed = True
        elif arabic_chars > 3:
            score = 0.35
            message = f"Weak Arabic ({arabic_chars} chars)"
            passed = False
        else:
            score = 0.15
            message = "No Arabic header"
            passed = False
        
        return FeatureResult(passed, score, message,
                            {'arabic_chars': arabic_chars, 'keywords': total_keywords})
    
    def _verify_color_scheme_adaptive(self, image: np.ndarray) -> FeatureResult:
        """Adaptive color scheme verification"""
        if len(image.shape) != 3:
            return FeatureResult(False, 0.0, "Grayscale image", {})
        
        # Use adaptive color detection
        color_analysis = self.color_detector.analyze_colors(image, use_normalization=True)
        
        # Also check color relationships
        color_relationships = self.color_detector.detect_color_relationships(image)
        
        colors_detected = color_analysis['colors_detected']
        normalized_score = color_analysis['normalized_score']
        
        # Boost score if color relationships look valid
        if color_relationships['valid'] and color_relationships['num_peaks'] >= 2:
            normalized_score = min(normalized_score * 1.2, 1.0)
        
        passed = normalized_score >= 0.35  # Lowered threshold for adaptive detection
        message = f"{len(colors_detected)} colors: {', '.join(colors_detected[:3]) if colors_detected else 'none'}"
        
        details = {
            'color_scores': color_analysis['color_scores'],
            'color_relationships': color_relationships,
            'lighting_adapted': True
        }
        
        return FeatureResult(passed, normalized_score, message, details)
    
    def _detect_security_pattern_adaptive(self, image: np.ndarray) -> FeatureResult:
        """Adaptive security pattern detection"""
        security_region = self._get_region(image, 'security_strip')
        
        if len(security_region.shape) != 3:
            return FeatureResult(False, 0.0, "Grayscale image", {})
        
        # Normalize the security region for lighting
        normalized_region = self.lighting_estimator.normalize_image(security_region)
        hsv_security = cv2.cvtColor(normalized_region, cv2.COLOR_BGR2HSV)
        
        # Use wider, adaptive blue range
        # Blue can shift significantly under different lighting
        blue_ranges = [
            ([85, 25, 40], [135, 255, 255]),   # Standard blue
            ([90, 15, 30], [125, 200, 255]),   # Desaturated blue (LED)
            ([80, 30, 50], [140, 255, 255]),   # Wide range (tungsten)
        ]
        
        max_blue_ratio = 0.0
        for lower, upper in blue_ranges:
            blue_mask = cv2.inRange(hsv_security, np.array(lower), np.array(upper))
            ratio = np.sum(blue_mask > 0) / blue_mask.size
            max_blue_ratio = max(max_blue_ratio, ratio)
        
        gray_security = cv2.cvtColor(security_region, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_security, 30, 100)
        edge_density = float(np.sum(edges > 0) / edges.size)
        
        # Also check for pattern regularity (security features often have regular patterns)
        pattern_score = self._check_pattern_regularity(gray_security)
        
        if max_blue_ratio > 0.15 and edge_density > 0.04:
            score = 0.95
            message = f"Strong security ({max_blue_ratio:.1%} blue)"
            passed = True
        elif max_blue_ratio > 0.08:
            score = 0.75
            message = f"Security detected ({max_blue_ratio:.1%})"
            passed = True
        elif max_blue_ratio > 0.04 or edge_density > 0.06 or pattern_score > 0.5:
            score = 0.50
            message = "Weak security pattern"
            passed = True
        else:
            score = 0.20
            message = "No security pattern"
            passed = False
        
        return FeatureResult(passed, score, message,
                            {'blue_ratio': float(max_blue_ratio), 'edge_density': edge_density,
                             'pattern_score': float(pattern_score)})
    
    def _check_pattern_regularity(self, gray_image: np.ndarray) -> float:
        """Check for regular patterns in security features"""
        if gray_image.size < 100:
            return 0.0
        
        try:
            # Use FFT to detect regular patterns
            f_transform = np.fft.fft2(gray_image.astype(np.float32))
            f_shift = np.fft.fftshift(f_transform)
            magnitude = np.abs(f_shift)
            
            # Normalize
            max_val = magnitude.max()
            if max_val > 0:
                magnitude = magnitude / max_val
            
            # Create mask as uint8 (FIXED: was float64 which caused cv2.circle error)
            mask = np.ones(magnitude.shape, dtype=np.uint8)
            
            # Calculate center coordinates as integers
            center_y = int(magnitude.shape[0] // 2)
            center_x = int(magnitude.shape[1] // 2)
            
            # Exclude center region (DC component)
            cv2.circle(mask, (center_x, center_y), 5, 0, -1)
            
            # Apply mask to magnitude
            masked_magnitude = magnitude * mask.astype(np.float32)
            
            # Count significant peaks
            max_masked = masked_magnitude.max()
            if max_masked > 0:
                threshold = 0.3 * max_masked
                peaks = int(np.sum(masked_magnitude > threshold))
            else:
                peaks = 0
            
            # Normalize score (more peaks = more regular pattern)
            score = min(peaks / 50.0, 1.0)
            
            return score
        
        except Exception as e:
            # If FFT analysis fails, return neutral score
            print(f"Pattern analysis warning: {e}")
            return 0.3
    
    def _extract_and_validate_id(self, image: np.ndarray) -> FeatureResult:
        # Extract from ID region
        id_region = self._get_region(image, 'id_number_region')
        region_text = self.ocr.extract_text(id_region)
        
        # Also full image
        full_text = self.ocr.extract_text(image)
        
        all_text = region_text + '\n' + full_text
        
        # Clean and find 14-digit sequences
        cleaned = re.sub(r'[^0-9]', '', all_text)
        cleaned = cleaned.replace('O', '0').replace('o', '0').replace('I', '1').replace('l', '1')
        
        potential_ids = re.findall(r'\d{14}', cleaned)
        potential_ids = list(set(potential_ids))
        
        for pid in potential_ids:
            if pid[0] in ['2', '3']:
                validation = self.validator.validate(pid)
                
                if validation['valid']:
                    return FeatureResult(
                        True, 1.0,
                        f"[OK] Valid ID: {pid}",
                        validation
                    )
        
        if potential_ids:
            return FeatureResult(False, 0.35, f"{len(potential_ids)} numbers, none valid", {})
        else:
            return FeatureResult(False, 0.0, "No 14-digit number", {})

    def _detect_driving_license(self, image: np.ndarray) -> Dict:
        """Explicitly check for driving license keywords"""
        # Check header region and full image
        regions_to_check = [
            self._get_region(image, 'header_region'),
            image
        ]
        
        found_keywords = []
        
        for roi in regions_to_check:
            text = self.ocr.extract_text(roi)
            # Remove spaces for better keyword matching in Arabic
            text_cleaned = text.replace(' ', '')
            
            for kw in self.config.DRIVING_LICENSE_KEYWORDS:
                if kw in text or kw in text_cleaned:
                    if kw not in found_keywords:
                        found_keywords.append(kw)
        
        return {
            'detected': len(found_keywords) > 0,
            'keywords': found_keywords
        }


# ==================== DOCUMENT DETECTOR ====================
class DocumentDetector:
    """Detect and extract document from image"""
    
    def __init__(self):
        self.min_area = 4000
        self.aspect_ratio_range = (1.2, 2.1)
        self.target_width = 850
        self.target_height = 536
    
    def detect_and_extract(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        result = self._try_contour_detection(image)
        if result is not None:
            return result
        
        result = self._try_enhanced_detection(image)
        if result is not None:
            return result
        
        h, w = image.shape[:2]
        aspect = w / h
        if 1.35 < aspect < 1.85 and w > 350:
            print("   ℹ Image appears pre-cropped")
            resized = cv2.resize(image, (self.target_width, self.target_height))
            return resized, image
        
        return None, None
    
    def _try_contour_detection(self, image: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 25, 150)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated = cv2.dilate(edges, kernel, iterations=2)
        
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        return self._find_best_contour(contours, image)
    
    def _try_enhanced_detection(self, image: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        return self._try_contour_detection(enhanced)
    
    def _find_best_contour(self, contours: list, image: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        for contour in contours[:25]:
            area = cv2.contourArea(contour)
            
            if area < self.min_area:
                continue
            
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            
            if 4 <= len(approx) <= 12:
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = w / float(h)
                
                if self.aspect_ratio_range[0] <= aspect_ratio <= self.aspect_ratio_range[1]:
                    if len(approx) == 4:
                        corners = approx
                    else:
                        corners = np.array([
                            [x, y], [x + w, y],
                            [x + w, y + h], [x, y + h]
                        ], dtype=np.float32).reshape(-1, 1, 2).astype(np.int32)
                    
                    warped = self._perspective_transform(image, corners)
                    annotated = image.copy()
                    cv2.drawContours(annotated, [corners], -1, (0, 255, 0), 3)
                    
                    return warped, annotated
        
        return None
    
    def _perspective_transform(self, image: np.ndarray, corners: np.ndarray) -> np.ndarray:
        pts = corners.reshape(4, 2).astype('float32')
        rect = np.zeros((4, 2), dtype='float32')
        
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        
        dst = np.array([
            [0, 0],
            [self.target_width - 1, 0],
            [self.target_width - 1, self.target_height - 1],
            [0, self.target_height - 1]
        ], dtype='float32')
        
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (self.target_width, self.target_height))
        
        return warped


# ==================== PIPELINE ====================
class EgyptianIDVerificationPipeline:
    """Complete production pipeline with shared OCR engine"""
    
    def __init__(self, ocr_engine: Optional[OCREngineSingleton] = None):
        """
        Initialize pipeline with optional pre-loaded OCR engine.
        
        Args:
            ocr_engine: Pre-initialized OCR engine. If None, will initialize globally.
        """
        self.detector = DocumentDetector()
        
        # Get or create OCR engine ONCE
        self.ocr_engine = ocr_engine if ocr_engine is not None else initialize_ocr_engine()
        
        # Pass shared OCR engine to verifier
        self.verifier = EnhancedEgyptianIDFeatureDetector(ocr_engine=self.ocr_engine)
    
    def process_image(self, image_path: str, save_output: bool = True) -> Dict:
        print(f"\n{'='*70}")
        print(f"[FILE] Processing: {os.path.basename(image_path)}")
        print(f"{'='*70}\n")
        
        image = cv2.imread(image_path)
        if image is None:
            print("[X] Cannot read image\n")
            return {'success': False, 'error': 'Cannot read image', 'is_egyptian_id': False}
        
        print(f"[OK] Image loaded: {image.shape[1]}x{image.shape[0]} pixels\n")
        
        print("Step 1: Document Detection")
        print("-" * 70)
        extracted, annotated = self.detector.detect_and_extract(image)
        
        if extracted is None:
            print("[X] No document detected\n")
            return {'success': False, 'error': 'No document detected', 'is_egyptian_id': False}
        
        print(f"[OK] Document extracted: {extracted.shape[1]}x{extracted.shape[0]} pixels\n")
        
        print("Step 2: Egyptian National ID Verification")
        print("-" * 70)
        verification = self.verifier.verify_all_features(extracted)
        
        if save_output:
            self._save_results(image_path, extracted, annotated, verification)
        
        return {
            'success': True,
            'is_egyptian_id': verification['is_egyptian_national_id'],
            'confidence': verification['confidence'],
            'verification': verification,
            'file': os.path.basename(image_path)
        }
    
    def _save_results(self, image_path: str, extracted: np.ndarray,
                     annotated: Optional[np.ndarray], verification: Dict):
        output_dir = r'C:\Users\asus\Downloads\id_checker\output'
        os.makedirs(output_dir, exist_ok=True)
        
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        cv2.imwrite(os.path.join(output_dir, f'{base_name}_extracted.jpg'), extracted)
        
        if annotated is not None:
            cv2.imwrite(os.path.join(output_dir, f'{base_name}_annotated.jpg'), annotated)
        
        result_img = extracted.copy()
        
        is_valid = verification['is_egyptian_national_id']
        status_text = "EGYPTIAN ID" if is_valid else "NOT EGYPTIAN ID"
        color = (0, 255, 0) if is_valid else (0, 0, 255)
        
        cv2.putText(result_img, status_text, (15, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(result_img, f"Confidence: {verification['confidence']*100:.1f}%",
                   (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Add lighting info
        lighting = verification.get('lighting_conditions', {}).get('type', 'unknown')
        cv2.putText(result_img, f"Lighting: {lighting}",
                   (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        if 'id_number' in verification['extracted_data']:
            data = verification['extracted_data']
            cv2.putText(result_img, f"ID: {data['id_number']}",
                       (15, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            if 'info' in data:
                info = data['info']
                cv2.putText(result_img,
                           f"{info['age']}y | {info['gender']} | {info['governorate']}",
                           (15, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        
        cv2.imwrite(os.path.join(output_dir, f'{base_name}_result.jpg'), result_img)
    
    def process_folder(self, folder_path: str) -> List[Dict]:
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        image_files = []
        
        for ext in extensions:
            image_files.extend(glob.glob(os.path.join(folder_path, ext)))
            image_files.extend(glob.glob(os.path.join(folder_path, ext.upper())))
        
        image_files = sorted(list(set(image_files)))
        
        if not image_files:
            print(f"[X] No images found in {folder_path}")
            return []
        
        print(f"\n{'='*70}")
        print(f"[SEARCH] Found {len(image_files)} unique images")
        print(f"{'='*70}")
        
        results = []
        for img_path in image_files:
            result = self.process_image(img_path, save_output=True)
            results.append(result)
        
        return results


# ==================== WEB SERVER INTEGRATION EXAMPLE ====================
class IDVerificationService:
    """
    Service class for web server integration.
    Demonstrates proper singleton usage for production environments.
    """
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        with self._lock:
            if self._initialized:
                return
            
            print("Initializing ID Verification Service...")
            # Initialize OCR once
            self.ocr_engine = initialize_ocr_engine()
            # Create pipeline with shared OCR
            self.pipeline = EgyptianIDVerificationPipeline(ocr_engine=self.ocr_engine)
            self._initialized = True
            print("ID Verification Service ready!")
    
    def verify_image(self, image_path: str) -> Dict:
        """Thread-safe image verification"""
        return self.pipeline.process_image(image_path, save_output=False)
    
    def verify_image_array(self, image: np.ndarray) -> Dict:
        """Verify image from numpy array (useful for web uploads)"""
        # Detect document
        extracted, _ = self.pipeline.detector.detect_and_extract(image)
        
        if extracted is None:
            return {'success': False, 'error': 'No document detected', 'is_egyptian_id': False}
        
        # Verify features
        verification = self.pipeline.verifier.verify_all_features(extracted)
        
        return {
            'success': True,
            'is_egyptian_id': verification['is_egyptian_national_id'],
            'confidence': verification['confidence'],
            'verification': verification
        }


# ==================== MAIN ====================
def main():
    folder_path = r'C:\Users\asus\Downloads\id_checker'
    
    print("\n" + "="*70)
    print("EGYPTIAN ID VERIFIER - ENHANCED VERSION")
    print("   With Adaptive Color Detection & Optimized OCR Loading")
    print("="*70)
    print(f"[DIR] Target folder: {folder_path}\n")
    
    try:
        # Initialize OCR engine ONCE at startup
        ocr_engine = initialize_ocr_engine()
        
        # Create pipeline with shared OCR engine
        pipeline = EgyptianIDVerificationPipeline(ocr_engine=ocr_engine)
    except Exception as e:
        print(f"[X] Failed to initialize: {e}")
        import traceback
        traceback.print_exc()
        return
    
    results = pipeline.process_folder(folder_path)
    
    if not results:
        return
    
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70 + "\n")
    
    total = len(results)
    detected = sum(1 for r in results if r['success'])
    valid_ids = sum(1 for r in results if r.get('is_egyptian_id', False))
    
    print(f"Total images: {total}")
    print(f"Documents detected: {detected}/{total}")
    print(f"[PASS] Valid Egyptian IDs: {valid_ids}/{detected}\n")
    
    print("Results:")
    print("-" * 70)
    
    for r in sorted(results, key=lambda x: x.get('confidence', 0), reverse=True):
        name = r.get('file', 'Unknown')
        
        if not r['success']:
            print(f"[X] {name:30} - {r.get('error', 'Error')}")
        elif r['is_egyptian_id']:
            conf = r['confidence'] * 100
            lighting = r.get('verification', {}).get('lighting_conditions', {}).get('type', 'unknown')
            print(f"[PASS] {name:30} - EGYPTIAN ID ({conf:.1f}%) [lighting: {lighting}]")
            
            if 'verification' in r and 'extracted_data' in r['verification']:
                data = r['verification']['extracted_data']
                if 'id_number' in data and 'info' in data:
                    info = data['info']
                    print(f"   {'':30}   ID: {data['id_number']}")
                    print(f"   {'':30}   {info['age']}y | {info['gender']} | {info['governorate']}")
        else:
            conf = r['confidence'] * 100
            print(f"[X] {name:30} - NOT Egyptian ID ({conf:.1f}%)")
    
    print(f"\n{'='*70}")
    print(f"[Saved] Saved to: {folder_path}\\output")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
