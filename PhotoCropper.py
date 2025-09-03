import streamlit as st
import cv2
import numpy as np
import zipfile
import io

# Check OpenCV version and backend
st.sidebar.info(f"OpenCV version: {cv2.__version__}")

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
    
    # Try to load Haar cascades
    try:
        detectors = [
            cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'),
            cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
        ]
    except:
        st.error("Error loading face detection models. Please ensure OpenCV is properly installed.")
        return None
    
    faces = []
    for detector in detectors:
        detected_faces = detector.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=7,
            minSize=(100, 100),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        if len(detected_faces) > 0:
            faces.extend(detected_faces)
    
    if faces:
        # Return largest face with padding
        face = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)[0]
        x, y, w, h = face
        # Add 20% padding around detected face
        pad_w = int(w * 0.2)
        pad_h = int(h * 0.2)
        x = max(0, x - pad_w)
        y = max(0, y - pad_h)
        w = min(image.shape[1] - x, w + 2*pad_w)
        h = min(image.shape[0] - y, h + 2*pad_h)
        return (x, y, w, h)
    return None

def calculate_crop(image, face, ratio_type='standard'):
    """Calculate crop with proper headroom while maintaining aspect ratio"""
    x, y, w, h = face
    img_h, img_w = image.shape[:2]
    
    # Set target ratios
    if ratio_type == 'standard':
        target_ratio = 2.0 / 2.3
    else:  # square
        target_ratio = 1.0
    
    # Calculate the center of the face
    face_center_x = x + w // 2
    face_center_y = y + h // 2
    
    # Calculate crop dimensions based on face size
    if ratio_type == 'standard':
        target_height = int(h / 0.7)
    else:
        target_height = int(h / 0.65)
    
    target_width = int(target_height * target_ratio)
    
    # Calculate crop coordinates centered on face
    crop_x1 = max(0, face_center_x - target_width // 2)
    crop_y1 = max(0, face_center_y - target_height // 2)
    crop_x2 = min(img_w, crop_x1 + target_width)
    crop_y2 = min(img_h, crop_y1 + target_height)
    
    # Adjust boundaries
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
    
    # Final dimensions
    final_width = crop_x2 - crop_x1
    final_height = crop_y2 - crop_y1
    
    # Verify minimum dimensions
    min_dim = 300
    if final_width < min_dim or final_height < min_dim:
        return None
    
    return (crop_x1, crop_y1, crop_x2, crop_y2)

def process_uploaded_files(uploaded_files, ratio_type):
    """Process all uploaded files and return results"""
    processed_images = {}
    
    for uploaded_file in uploaded_files:
        try:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            if image is None:
                processed_images[uploaded_file.name] = {
                    'success': False,
                    'error': "Invalid image file"
                }
                continue
            
            enhanced = enhance_image(image)
            face = detect_face(enhanced)
            
            if face:
                crop = calculate_crop(image, face, ratio_type)
                if crop:
                    x1, y1, x2, y2 = crop
                    cropped = image[y1:y2, x1:x2]
                    
                    if ratio_type == 'standard':
                        output_size = (600, 690)
                    else:
                        output_size = (600, 600)
                    
                    resized = cv2.resize(cropped, output_size, interpolation=cv2.INTER_CUBIC)
                    display_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                    
                    _, buffer = cv2.imencode('.jpg', resized)
                    processed_images[uploaded_file.name] = {
                        'image_bytes': buffer.tobytes(),
                        'display_image': display_image,
                        'success': True
                    }
                else:
                    processed_images[uploaded_file.name] = {
                        'success': False,
                        'error': "Could not crop image"
                    }
            else:
                processed_images[uploaded_file.name] = {
                    'success': False,
                    'error': "No face detected"
                }
                
        except Exception as e:
            processed_images[uploaded_file.name] = {
                'success': False,
                'error': f"Error: {str(e)}"
            }
    
    return processed_images

def main():
    st.set_page_config(page_title="Passport Photo Cropper", page_icon="📸")
    st.title("📸 Passport Photo Cropper")
    
    st.sidebar.header("Settings")
    ratio_type = st.sidebar.radio("Select photo type:", 
                                 ("Standard (2x2.3cm)", "Square (1x1)"),
                                 index=0)
    ratio_type = "standard" if ratio_type == "Standard (2x2.3cm)" else "square"
    
    st.sidebar.info("""
    **Instructions:**
    1. Upload one or more photos
    2. Click 'Crop Photos'
    3. Download your processed passport photos
    """)
    
    uploaded_files = st.file_uploader("Upload images", 
                                     type=["jpg", "jpeg", "png"], 
                                     accept_multiple_files=True)
    
    # Initialize session state
    if 'processed' not in st.session_state:
        st.session_state.processed = False
    if 'processed_images' not in st.session_state:
        st.session_state.processed_images = {}
    
    if uploaded_files:
        st.success(f"📁 {len(uploaded_files)} file(s) uploaded successfully!")
        
        if not st.session_state.processed:
            if st.button("🚀 Crop Photos", type="primary", use_container_width=True):
                with st.spinner("Processing images... This may take a few seconds"):
                    st.session_state.processed_images = process_uploaded_files(uploaded_files, ratio_type)
                    st.session_state.processed = True
                st.rerun()
    
    if st.session_state.processed and st.session_state.processed_images:
        processed_images = st.session_state.processed_images
        successful_images = {k: v for k, v in processed_images.items() if v['success']}
        failed_images = {k: v for k, v in processed_images.items() if not v['success']}
        
        st.subheader("📊 Results")
        st.success(f"✅ Successfully processed: {len(successful_images)} image(s)")
        if failed_images:
            st.warning(f"❌ Failed to process: {len(failed_images)} image(s)")
        
        if successful_images:
            st.subheader("📷 Processed Photos")
            cols = st.columns(2)
            for i, (name, data) in enumerate(successful_images.items()):
                with cols[i % 2]:
                    st.image(data['display_image'], caption=name, use_container_width=True)
            
            # Create download ZIP
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                for name, data in successful_images.items():
                    zip_file.writestr(name, data['image_bytes'])
            
            zip_buffer.seek(0)
            
            st.download_button(
                label="📥 Download All Photos",
                data=zip_buffer,
                file_name="passport_photos.zip",
                mime="application/zip",
                use_container_width=True
            )
        
        if failed_images:
            st.subheader("❌ Failed Images")
            for name, data in failed_images.items():
                st.error(f"**{name}**: {data['error']}")
        
        if st.button("🔄 Process New Photos", use_container_width=True):
            st.session_state.processed = False
            st.session_state.processed_images = {}
            st.rerun()
    
    elif not uploaded_files:
        st.info("👆 Upload photos to get started! Supported formats: JPG, JPEG, PNG")

if __name__ == "__main__":
    main()
