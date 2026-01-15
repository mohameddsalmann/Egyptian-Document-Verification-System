import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import re
import os
from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Dict
from enum import Enum
from pathlib import Path


class ValidationStatus(Enum):
    VALID = "valid"
    INVALID = "invalid"
    NEEDS_REVIEW = "needs_review"


@dataclass
class ValidationResult:
    status: ValidationStatus
    confidence: float
    messages: list
    detected_fields: dict
    filename: str = ""
    score_breakdown: dict = field(default_factory=dict)
    debug_info: dict = field(default_factory=dict)


class EgyptianDrivingLicenseValidator:
    """
    Enhanced validator with better OCR preprocessing and debug output.
    """
    
    EXPECTED_ASPECT_RATIO = 1.57
    ASPECT_RATIO_TOLERANCE = 0.20
    
    MIN_WIDTH = 300
    MIN_HEIGHT = 180
    
    # Simplified critical keywords (more variations for OCR errors)
    CRITICAL_KEYWORDS = [
        "ادارة", "اداره",           # Administration
        "مرور",                      # Traffic
        "رخصة", "رخصه",             # License
        "قيادة", "قياده",           # Driving
    ]
    
    SECONDARY_KEYWORDS = [
        "مصرى", "مصري",             # Egyptian
        "Egyptian",
        "تاريخ",                     # Date
        "نهاية", "نهايه",           # End/Expiry
        "التحرير",                   # Issue
        "الترخيص",                   # License
        "وحدة", "وحده",             # Unit
    ]
    
    EGYPTIAN_CITIES = [
        "القاهرة", "القاهره",
        "الاسكندرية", "الاسكندريه",
        "البحيرة", "البحيره",
        "الجيزة", "الجيزه",
        "المقطم",
        "عين الصيرة", "عين الصيره",
        "كوم حماده", "كوم حمادة",
        "سيدي جابر",
        "ابيس",
    ]
    
    VEHICLE_CLASSES = ['B', 'C', 'CE', 'DE', 'D']
    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    
    def __init__(self, tesseract_path: Optional[str] = None, debug: bool = True):
        self.debug = debug
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        
        # Check Tesseract availability
        try:
            langs = pytesseract.get_languages()
            print(f"Tesseract languages available: {langs}")
            if 'ara' not in langs:
                print("⚠ WARNING: Arabic language pack not installed!")
                print("  Install with: apt-get install tesseract-ocr-ara")
                print("  Or download from Tesseract GitHub for Windows")
        except Exception as e:
            print(f"⚠ Tesseract check failed: {e}")
    
    def preprocess_image(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Create multiple preprocessed versions for better OCR.
        Returns list of processed images to try.
        """
        processed = []
        
        # Original
        processed.append(image.copy())
        
        # 1. Grayscale + Contrast Enhancement
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 2. Adaptive thresholding
        adaptive = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        processed.append(cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR))
        
        # 3. OTSU thresholding
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        processed.append(cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR))
        
        # 4. Contrast enhancement using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        processed.append(cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR))
        
        # 5. Sharpening
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(image, -1, kernel)
        processed.append(sharpened)
        
        # 6. Denoise + threshold
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        _, denoised_thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        processed.append(cv2.cvtColor(denoised_thresh, cv2.COLOR_GRAY2BGR))
        
        # 7. Resize if image is small
        height, width = image.shape[:2]
        if width < 800:
            scale = 800 / width
            resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            processed.append(resized)
        
        return processed
    
    def extract_text_multi(self, image: np.ndarray) -> str:
        """
        Extract text using multiple preprocessing methods and combine results.
        """
        all_text = []
        processed_images = self.preprocess_image(image)
        
        configs = [
            ('ara+eng', '--psm 3'),   # Fully automatic
            ('ara+eng', '--psm 6'),   # Uniform block of text
            ('ara+eng', '--psm 4'),   # Single column
            ('ara', '--psm 6'),       # Arabic only
            ('eng', '--psm 6'),       # English only
        ]
        
        for idx, proc_img in enumerate(processed_images[:4]):  # Limit to first 4
            for lang, config in configs[:3]:  # Limit configs
                try:
                    pil_img = Image.fromarray(cv2.cvtColor(proc_img, cv2.COLOR_BGR2RGB))
                    text = pytesseract.image_to_string(pil_img, lang=lang, config=config)
                    if text.strip():
                        all_text.append(text)
                except Exception:
                    continue
        
        # Combine all extracted text
        combined = " ".join(all_text)
        return combined
    
    def validate_folder(self, folder_path: str) -> List[ValidationResult]:
        """Validate all images in folder with debug output."""
        results = []
        folder = Path(folder_path)
        
        if not folder.exists():
            print(f"Error: Folder '{folder_path}' does not exist!")
            return results
        
        image_files = [
            f for f in folder.iterdir() 
            if f.is_file() and f.suffix.lower() in self.IMAGE_EXTENSIONS
        ]
        
        if not image_files:
            print(f"No image files found in '{folder_path}'")
            return results
        
        print(f"\nFound {len(image_files)} image(s)")
        print("=" * 70)
        
        for idx, image_path in enumerate(image_files, 1):
            print(f"\n[{idx}/{len(image_files)}] {image_path.name}")
            print("-" * 60)
            
            result = self.validate(str(image_path))
            result.filename = image_path.name
            results.append(result)
            
            self._print_result(result)
        
        return results
    
    def _print_result(self, result: ValidationResult):
        """Print detailed result with debug info."""
        icons = {
            ValidationStatus.VALID: "✓",
            ValidationStatus.INVALID: "✗", 
            ValidationStatus.NEEDS_REVIEW: "?"
        }
        
        print(f"\nStatus: {icons[result.status]} {result.status.value.upper()}")
        print(f"Confidence: {result.confidence * 100:.1f}%")
        
        print("\nScore Breakdown:")
        for check, score in result.score_breakdown.items():
            bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))
            status = "✓" if score >= 0.5 else "✗"
            print(f"  {status} {check:25} [{bar}] {score*100:5.1f}%")
        
        # Debug: Show what was detected
        if self.debug and result.debug_info:
            print("\n[DEBUG] Detected Text Samples:")
            if 'extracted_text_sample' in result.debug_info:
                sample = result.debug_info['extracted_text_sample'][:300]
                print(f"  {sample}...")
            
            if 'found_keywords' in result.debug_info:
                print(f"\n[DEBUG] Found Keywords: {result.debug_info['found_keywords']}")
            
            if 'found_numbers' in result.debug_info:
                print(f"[DEBUG] Found Numbers: {result.debug_info['found_numbers']}")
    
    def validate(self, image_path: str) -> ValidationResult:
        """Main validation with improved OCR."""
        messages = []
        detected_fields = {}
        score_breakdown = {}
        debug_info = {}
        
        # Load image
        try:
            image = cv2.imread(image_path)
            if image is None:
                return ValidationResult(
                    status=ValidationStatus.INVALID,
                    confidence=0.0,
                    messages=["Could not load image"],
                    detected_fields={},
                    score_breakdown={},
                    debug_info={}
                )
        except Exception as e:
            return ValidationResult(
                status=ValidationStatus.INVALID,
                confidence=0.0,
                messages=[f"Load error: {str(e)}"],
                detected_fields={},
                score_breakdown={},
                debug_info={}
            )
        
        # Extract all text first (used by multiple checks)
        full_text = self.extract_text_multi(image)
        debug_info['extracted_text_sample'] = full_text[:500] if full_text else "NO TEXT"
        
        # === CHECK 1: Dimensions ===
        dim_score, dim_msg, dim_det = self._check_dimensions(image)
        score_breakdown['1_Dimensions'] = dim_score
        messages.append(dim_msg)
        detected_fields.update(dim_det)
        
        # === CHECK 2: Critical Keywords ===
        kw_score, kw_msg, kw_det, kw_debug = self._check_keywords(full_text)
        score_breakdown['2_Keywords'] = kw_score
        messages.append(kw_msg)
        detected_fields.update(kw_det)
        debug_info['found_keywords'] = kw_debug
        
        # === CHECK 3: License Number ===
        lic_score, lic_msg, lic_det, lic_debug = self._check_license_number(full_text, image)
        score_breakdown['3_License_Number'] = lic_score
        messages.append(lic_msg)
        detected_fields.update(lic_det)
        debug_info['found_numbers'] = lic_debug
        
        # === CHECK 4: Dates ===
        date_score, date_msg, date_det = self._check_dates(full_text)
        score_breakdown['4_Dates'] = date_score
        messages.append(date_msg)
        detected_fields.update(date_det)
        
        # === CHECK 5: Photo Region ===
        photo_score, photo_msg = self._check_photo_region(image)
        score_breakdown['5_Photo'] = photo_score
        messages.append(photo_msg)
        
        # === CHECK 6: Barcode ===
        bar_score, bar_msg = self._check_barcode(image)
        score_breakdown['6_Barcode'] = bar_score
        messages.append(bar_msg)
        
        # === CHECK 7: Color Scheme ===
        color_score, color_msg = self._check_color_scheme(image)
        score_breakdown['7_Color'] = color_score
        messages.append(color_msg)
        
        # === CHECK 8: Vehicle Class ===
        vc_score, vc_msg = self._check_vehicle_class(full_text, image)
        score_breakdown['8_Vehicle_Class'] = vc_score
        messages.append(vc_msg)
        
        # === WEIGHTED CONFIDENCE ===
        weights = {
            '1_Dimensions': 0.10,
            '2_Keywords': 0.25,
            '3_License_Number': 0.20,
            '4_Dates': 0.10,
            '5_Photo': 0.10,
            '6_Barcode': 0.10,
            '7_Color': 0.10,
            '8_Vehicle_Class': 0.05,
        }
        
        confidence = sum(score_breakdown[k] * weights[k] for k in score_breakdown)
        
        # === FLEXIBLE VALIDATION RULES ===
        has_keywords = kw_det.get('keyword_count', 0) >= 2
        has_license = lic_det.get('has_license_number', False)
        has_dates = date_det.get('date_count', 0) >= 1
        has_structure = (photo_score >= 0.5 and bar_score >= 0.4)
        has_color = color_score >= 0.5
        
        critical_passed = sum([has_keywords, has_license, has_dates, has_structure, has_color])
        
        # More lenient thresholds
        if confidence >= 0.55 and critical_passed >= 3:
            status = ValidationStatus.VALID
        elif confidence >= 0.40 and critical_passed >= 2:
            status = ValidationStatus.NEEDS_REVIEW
        elif confidence >= 0.35 and (has_keywords or has_license):
            status = ValidationStatus.NEEDS_REVIEW
        else:
            status = ValidationStatus.INVALID
        
        return ValidationResult(
            status=status,
            confidence=round(confidence, 2),
            messages=[m for m in messages if m],
            detected_fields=detected_fields,
            score_breakdown=score_breakdown,
            debug_info=debug_info
        )
    
    def _check_dimensions(self, image: np.ndarray) -> Tuple[float, str, dict]:
        """Check image dimensions."""
        height, width = image.shape[:2]
        details = {'width': width, 'height': height}
        
        if width < self.MIN_WIDTH or height < self.MIN_HEIGHT:
            return 0.3, f"Small image: {width}x{height}", details
        
        aspect = width / height
        details['aspect_ratio'] = round(aspect, 2)
        
        diff = abs(aspect - self.EXPECTED_ASPECT_RATIO)
        
        if diff <= 0.10:
            return 1.0, "Dimensions: Perfect", details
        elif diff <= self.ASPECT_RATIO_TOLERANCE:
            return 0.7, "Dimensions: Good", details
        elif diff <= 0.35:
            return 0.4, "Dimensions: Acceptable", details
        else:
            return 0.2, f"Dimensions: Ratio {aspect:.2f} off", details
    
    def _check_keywords(self, text: str) -> Tuple[float, str, dict, list]:
        """Check for Arabic/English keywords."""
        details = {'keyword_count': 0}
        found = []
        
        text_lower = text.lower()
        
        # Check critical keywords
        critical_found = 0
        for kw in self.CRITICAL_KEYWORDS:
            if kw in text:
                critical_found += 1
                found.append(f"CRITICAL: {kw}")
        
        # Check secondary keywords
        secondary_found = 0
        for kw in self.SECONDARY_KEYWORDS:
            if kw in text or kw.lower() in text_lower:
                secondary_found += 1
                found.append(f"SECONDARY: {kw}")
        
        # Check city names
        city_found = False
        for city in self.EGYPTIAN_CITIES:
            if city in text:
                city_found = True
                found.append(f"CITY: {city}")
                break
        
        # Check for "Egyptian" in English
        if 'egyptian' in text_lower:
            secondary_found += 1
            found.append("ENGLISH: Egyptian")
        
        total = critical_found + secondary_found
        details['keyword_count'] = total
        details['critical_count'] = critical_found
        details['has_city'] = city_found
        
        # Scoring
        if critical_found >= 3 or (critical_found >= 2 and secondary_found >= 2):
            score = 1.0
        elif critical_found >= 2 or (critical_found >= 1 and secondary_found >= 2):
            score = 0.8
        elif critical_found >= 1 or secondary_found >= 3:
            score = 0.6
        elif secondary_found >= 2:
            score = 0.4
        elif secondary_found >= 1 or city_found:
            score = 0.3
        else:
            score = 0.0
        
        msg = f"Keywords: {critical_found} critical, {secondary_found} secondary"
        return score, msg, details, found
    
    def _check_license_number(self, text: str, image: np.ndarray) -> Tuple[float, str, dict, list]:
        """Check for license number format."""
        details = {'has_license_number': False}
        found_numbers = []
        
        # Find all number sequences
        all_numbers = re.findall(r'\d+', text)
        long_numbers = [n for n in all_numbers if len(n) >= 10]
        found_numbers = long_numbers[:5]  # First 5 for debug
        
        # Check for 14-digit numbers (Egyptian DL format)
        exact_14 = re.findall(r'\b\d{14}\b', text)
        
        # Check for 13-digit (OCR might miss one)
        near_14 = re.findall(r'\b\d{13,15}\b', text)
        
        if exact_14:
            details['has_license_number'] = True
            details['license_format'] = '14-digit'
            return 1.0, "License#: Valid 14-digit found", details, found_numbers
        elif near_14:
            details['has_license_number'] = True
            details['license_format'] = '13-15 digit'
            return 0.8, "License#: Near-valid format found", details, found_numbers
        elif long_numbers:
            # Has long numbers, might be partial
            max_len = max(len(n) for n in long_numbers)
            if max_len >= 10:
                details['has_license_number'] = True
                details['license_format'] = f'{max_len}-digit partial'
                return 0.5, f"License#: {max_len}-digit number found", details, found_numbers
        
        return 0.0, "License#: No valid number found", details, found_numbers
    
    def _check_dates(self, text: str) -> Tuple[float, str, dict]:
        """Check for date patterns."""
        details = {'date_count': 0}
        
        # Various date patterns
        patterns = [
            r'20[0-3][0-9][/-][0-1]?[0-9][/-][0-3]?[0-9]',  # 2020/01/15
            r'[0-3]?[0-9][/-][0-1]?[0-9][/-]20[0-3][0-9]',  # 15/01/2020
            r'[٢٠٢][٠-٩]{3}[/-][٠-٩]{1,2}[/-][٠-٩]{1,2}',  # Arabic numerals
        ]
        
        dates_found = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            dates_found.extend(matches)
        
        # Also look for year patterns
        years = re.findall(r'20[1-3][0-9]', text)
        
        details['date_count'] = len(dates_found)
        details['year_count'] = len(years)
        
        if len(dates_found) >= 2:
            return 1.0, f"Dates: {len(dates_found)} dates found", details
        elif len(dates_found) == 1:
            return 0.7, "Dates: 1 date found", details
        elif len(years) >= 2:
            return 0.5, f"Dates: {len(years)} years found", details
        elif len(years) >= 1:
            return 0.3, "Dates: 1 year found", details
        else:
            return 0.0, "Dates: None found", details
    
    def _check_photo_region(self, image: np.ndarray) -> Tuple[float, str]:
        """Check for photo in left region."""
        height, width = image.shape[:2]
        
        # Left 35% of image
        photo_region = image[int(height*0.08):int(height*0.78), 0:int(width*0.35)]
        
        gray = cv2.cvtColor(photo_region, cv2.COLOR_BGR2GRAY)
        
        # Face detection
        try:
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) > 0:
                return 1.0, "Photo: Face detected"
        except:
            pass
        
        # Check for content variation (indicates a photo exists)
        std_dev = np.std(gray)
        mean_val = np.mean(gray)
        
        # Check for rectangular content area
        edges = cv2.Canny(gray, 30, 100)
        edge_density = np.sum(edges > 0) / edges.size
        
        if std_dev > 40 and edge_density > 0.05:
            return 0.8, "Photo: Content detected"
        elif std_dev > 25:
            return 0.5, "Photo: Likely present"
        elif std_dev > 15:
            return 0.3, "Photo: Possibly present"
        
        return 0.1, "Photo: Not detected"
    
    def _check_barcode(self, image: np.ndarray) -> Tuple[float, str]:
        """Check for barcode at bottom."""
        height, width = image.shape[:2]
        
        # Bottom 25% of image
        barcode_region = image[int(height*0.75):height, int(width*0.1):int(width*0.9)]
        
        gray = cv2.cvtColor(barcode_region, cv2.COLOR_BGR2GRAY)
        
        # Detect vertical lines
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Gradient to find vertical edges
        grad_x = cv2.Sobel(thresh, cv2.CV_64F, 1, 0, ksize=3)
        grad_x = np.abs(grad_x)
        
        # Count strong vertical transitions
        vertical_lines = np.sum(grad_x > 50)
        total_pixels = grad_x.size
        line_ratio = vertical_lines / total_pixels
        
        # Also check standard deviation (barcode has high contrast)
        std_dev = np.std(gray)
        
        if line_ratio > 0.15 and std_dev > 60:
            return 1.0, "Barcode: Detected"
        elif line_ratio > 0.08 or std_dev > 50:
            return 0.7, "Barcode: Likely present"
        elif line_ratio > 0.04 or std_dev > 35:
            return 0.4, "Barcode: Possibly present"
        
        return 0.1, "Barcode: Not detected"
    
    def _check_color_scheme(self, image: np.ndarray) -> Tuple[float, str]:
        """Check for yellowish/beige color typical of Egyptian DL."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Yellow/beige range
        lower_yellow = np.array([15, 15, 120])
        upper_yellow = np.array([50, 180, 255])
        
        # Light/cream colors
        lower_light = np.array([0, 0, 170])
        upper_light = np.array([180, 50, 255])
        
        # Gray/neutral (some licenses have gray tones)
        lower_gray = np.array([0, 0, 100])
        upper_gray = np.array([180, 30, 220])
        
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        light_mask = cv2.inRange(hsv, lower_light, upper_light)
        gray_mask = cv2.inRange(hsv, lower_gray, upper_gray)
        
        total = image.shape[0] * image.shape[1]
        yellow_ratio = np.sum(yellow_mask > 0) / total
        light_ratio = np.sum(light_mask > 0) / total
        gray_ratio = np.sum(gray_mask > 0) / total
        
        combined = yellow_ratio + light_ratio * 0.5 + gray_ratio * 0.3
        
        if combined > 0.35:
            return 1.0, "Color: Matches Egyptian DL"
        elif combined > 0.20:
            return 0.7, "Color: Acceptable"
        elif combined > 0.10:
            return 0.4, "Color: Partial match"
        
        return 0.2, "Color: Different scheme"
    
    def _check_vehicle_class(self, text: str, image: np.ndarray) -> Tuple[float, str]:
        """Check for vehicle class indicator."""
        text_upper = text.upper()
        
        # Check in text
        for vc in self.VEHICLE_CLASSES:
            # Look for standalone class letters
            if re.search(rf'\b{vc}\b', text_upper):
                return 1.0, f"Vehicle Class: {vc} found"
        
        # Check bottom-left region specifically
        height, width = image.shape[:2]
        class_region = image[int(height*0.65):int(height*0.95), 0:int(width*0.20)]
        
        try:
            pil_img = Image.fromarray(cv2.cvtColor(class_region, cv2.COLOR_BGR2RGB))
            class_text = pytesseract.image_to_string(pil_img, lang='eng', config='--psm 6')
            
            for vc in self.VEHICLE_CLASSES:
                if vc in class_text.upper():
                    return 1.0, f"Vehicle Class: {vc} found"
        except:
            pass
        
        # Check for car icon shapes
        gray = cv2.cvtColor(class_region, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 3:
            return 0.5, "Vehicle Class: Icons detected"
        
        return 0.2, "Vehicle Class: Not found"


def print_final_report(results: List[ValidationResult]):
    """Print summary report."""
    print("\n" + "=" * 70)
    print("                         FINAL REPORT")
    print("=" * 70)
    
    valid = sum(1 for r in results if r.status == ValidationStatus.VALID)
    review = sum(1 for r in results if r.status == ValidationStatus.NEEDS_REVIEW)
    invalid = sum(1 for r in results if r.status == ValidationStatus.INVALID)
    
    print(f"\nTotal: {len(results)}")
    print(f"  ✓ Valid:        {valid}")
    print(f"  ? Needs Review: {review}")
    print(f"  ✗ Invalid:      {invalid}")
    
    if results:
        avg = sum(r.confidence for r in results) / len(results)
        print(f"  Avg Confidence: {avg*100:.1f}%")
    
    print("\n" + "-" * 70)
    
    for r in sorted(results, key=lambda x: -x.confidence):
        icon = {"valid": "✓", "needs_review": "?", "invalid": "✗"}[r.status.value]
        print(f"{icon} {r.filename:40} {r.confidence*100:5.1f}% - {r.status.value}")


def main():
    FOLDER_PATH = r"C:\Users\asus\Downloads\id_checker"
    
    print("\n" + "=" * 70)
    print("     EGYPTIAN DRIVING LICENSE VALIDATOR v3.0")
    print("     Debug Mode - Improved OCR")
    print("=" * 70)
    print(f"\nFolder: {FOLDER_PATH}")
    
    # For Windows, you might need to specify Tesseract path:
    # validator = EgyptianDrivingLicenseValidator(
    #     tesseract_path=r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    #     debug=True
    # )
    
    validator = EgyptianDrivingLicenseValidator(debug=True)
    results = validator.validate_folder(FOLDER_PATH)
    
    if results:
        print_final_report(results)
    
    return results


if __name__ == "__main__":
    results = main()
