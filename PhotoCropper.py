import cv2
import os
import sys
import numpy as np
from datetime import datetime

def enhance_image(image):
    """Improve image quality for better face detection"""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    return cv2.filter2D(enhanced, -1, kernel)

def detect_face(image):
    """Detect faces with improved parameters"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detectors = [
        cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'),
        cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
    ]
    
    faces = []
    for detector in detectors:
        faces.extend(detector.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=7,
            minSize=(200, 200),  # Increased minimum face size
            flags=cv2.CASCADE_SCALE_IMAGE
        ))
    
    if faces:
        # Return largest face with padding
        face = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)[0]
        x, y, w, h = face
        # Add 25% padding around detected face
        pad = int(w * 0.25)
        x = max(0, x - pad)
        y = max(0, y - pad)
        w = min(image.shape[1] - x, w + 2*pad)
        h = min(image.shape[0] - y, h + 2*pad)
        return (x, y, w, h)
    return None

def calculate_crop(image, face, ratio_type='standard'):
    """
    Calculate crop with slightly less headroom
    ratio_type: 'standard' (2:2.3) or 'square' (1:1)
    """
    x, y, w, h = face
    img_h, img_w = image.shape[:2]
    
    # Set ratios and face position - less headroom
    if ratio_type == 'standard':
        target_ratio = 2 / 2.3
        face_height_ratio = 0.65  # Face takes 65% of photo height
        headroom_ratio = 0.13     # Reduced from 0.2 - less space above head
    else:  # square
        target_ratio = 1
        face_height_ratio = 0.6   # Face takes 60% of photo height
        headroom_ratio = 0.22     # Reduced from 0.25 - less space above head
    
    # Calculate dimensions
    target_height = int(h / face_height_ratio)
    target_width = int(target_height * target_ratio)
    
    # Calculate crop coordinates with reduced headroom
    face_center_x = x + w // 2
    head_top = y  # Top of head position
    
    crop_x1 = max(0, face_center_x - target_width // 2)
    crop_y1 = max(0, head_top - int(target_height * headroom_ratio))
    
    crop_x2 = min(img_w, crop_x1 + target_width)
    crop_y2 = min(img_h, crop_y1 + target_height)
    
    # Adjust if at image boundaries
    if crop_x2 - crop_x1 < target_width:
        if crop_x1 == 0:
            crop_x2 = min(img_w, target_width)
        else:
            crop_x1 = max(0, img_w - target_width)
    
    if crop_y2 - crop_y1 < target_height:
        if crop_y1 == 0:
            crop_y2 = min(img_h, target_height)
        else:
            crop_y1 = max(0, img_h - target_height)
    
    # Verify minimum dimensions (400px for better quality)
    min_dim = 400
    if (crop_x2 - crop_x1) < min_dim or (crop_y2 - crop_y1) < min_dim:
        return None
    
    return (crop_x1, crop_y1, crop_x2, crop_y2)

def process_image(image_path, output_folder, counter, ratio_type):
    """Process single image with reduced headroom"""
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error reading image: {image_path}")
            return False
        
        enhanced = enhance_image(image)
        face = detect_face(enhanced)
        if face is None:
            print(f"No face detected in {os.path.basename(image_path)}")
            return False
        
        crop = calculate_crop(image, face, ratio_type)
        if not crop:
            print(f"Invalid crop for {os.path.basename(image_path)}")
            return False
        
        x1, y1, x2, y2 = crop
        
        # Crop and resize with high quality
        cropped = image[y1:y2, x1:x2]
        if ratio_type == 'standard':
            output_size = (600, 690)  # 2:2.3 ratio
        else:
            output_size = (600, 600)  # Square
        
        resized = cv2.resize(cropped, output_size, interpolation=cv2.INTER_CUBIC)
        
        # Save result
        original_name = os.path.splitext(os.path.basename(image_path))[0]
        output_filename = f"{original_name}.jpg"
        output_path = os.path.join(output_folder, output_filename)
        cv2.imwrite(output_path, resized, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        return True
    
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return False

def main():
    if len(sys.argv) < 4:
        print("Usage: python passport_cropper.py <input_folder> <output_folder> <ratio_type>")
        print("ratio_type: 'standard' (2x2.3cm) or 'square' (1x1)")
        print("Example: python passport_cropper.py ./input ./output standard")
        sys.exit(1)
    
    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
    ratio_type = sys.argv[3].lower()
    
    if ratio_type not in ('standard', 'square'):
        print("Invalid ratio type. Use 'standard' or 'square'")
        sys.exit(1)
    
    if not os.path.exists(input_folder):
        print(f"Input folder not found: {input_folder}")
        sys.exit(1)
    
    os.makedirs(output_folder, exist_ok=True)
    
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    images = [f for f in os.listdir(input_folder) if f.lower().endswith(supported_formats)]
    
    if not images:
        print("No supported images found")
        sys.exit(1)
    
    success_count = 0
    for i, img in enumerate(images, 1):
        img_path = os.path.join(input_folder, img)
        print(f"\nProcessing {i}/{len(images)}: {img}")
        
        if process_image(img_path, output_folder, i, ratio_type):
            success_count += 1
            print(f"Successfully created passport photo with reduced headroom")
        else:
            print(f"Failed to process {img}")
    
    print(f"\nCompleted. Successfully processed {success_count}/{len(images)} images.")

if __name__ == "__main__":
    main()
