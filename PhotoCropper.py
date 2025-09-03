import streamlit as st
import numpy as np
import zipfile
import io
import os

# Try to import OpenCV with fallback
try:
    import cv2
    OPENCV_AVAILABLE = True
    st.sidebar.success("✅ OpenCV loaded successfully")
except ImportError:
    OPENCV_AVAILABLE = False
    st.sidebar.error("❌ OpenCV not available - using fallback methods")

# Fallback functions if OpenCV is not available
if not OPENCV_AVAILABLE:
    st.error("""
    **OpenCV is not available on this system.**
    
    This app requires OpenCV for image processing. Please:
    1. Check that opencv-python-headless is in your requirements.txt
    2. Ensure all dependencies are properly installed
    3. Try using a different OpenCV version
    """)
    st.stop()

# Display versions for debugging
st.sidebar.info(f"OpenCV version: {cv2.__version__}")
st.sidebar.info(f"NumPy version: {np.__version__}")

def enhance_image(image):
    """Improve image quality for better face detection"""
    try:
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge channels and convert back to BGR
        lab = cv2.merge((l, a, b))
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Apply sharpening filter
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        return cv2.filter2D(enhanced, -1, kernel)
        
    except Exception as e:
        st.error(f"Error enhancing image: {e}")
        return image

def detect_face(image):
    """Detect faces with improved parameters"""
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Load Haar cascades
        try:
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
        except:
            # Fallback: try to load from absolute path
            try:
                face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            except:
                st.error("Could not load face detection model")
                return None
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        if len(faces) > 0:
            # Return the largest face
            faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
            x, y, w, h = faces[0]
            
            # Add padding
            padding = 20
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(image.shape[1] - x, w + 2 * padding)
            h = min(image.shape[0] - y, h + 2 * padding)
            
            return (x, y, w, h)
        
        return None
        
    except Exception as e:
        st.error(f"Error detecting face: {e}")
        return None

def calculate_crop(image, face, ratio_type='standard'):
    """Calculate crop with proper headroom"""
    try:
        x, y, w, h = face
        img_h, img_w = image.shape[:2]
        
        # Set aspect ratio
        if ratio_type == 'standard':
            target_ratio = 2.0 / 2.3
        else:  # square
            target_ratio = 1.0
        
        # Calculate crop based on face size
        face_center_x = x + w // 2
        face_center_y = y + h // 2
        
        # Determine crop size
        crop_height = int(h * 1.8)  # Include headroom
        crop_width = int(crop_height * target_ratio)
        
        # Calculate crop coordinates
        crop_x1 = max(0, face_center_x - crop_width // 2)
        crop_y1 = max(0, face_center_y - crop_height // 3)  # More space above
        crop_x2 = min(img_w, crop_x1 + crop_width)
        crop_y2 = min(img_h, crop_y1 + crop_height)
        
        # Adjust if out of bounds
        if crop_x2 - crop_x1 < crop_width:
            crop_x1 = max(0, crop_x2 - crop_width)
        if crop_y2 - crop_y1 < crop_height:
            crop_y1 = max(0, crop_y2 - crop_height)
        
        return (crop_x1, crop_y1, crop_x2, crop_y2)
        
    except Exception as e:
        st.error(f"Error calculating crop: {e}")
        return None

def process_uploaded_files(uploaded_files, ratio_type):
    """Process all uploaded files"""
    processed_images = {}
    
    for uploaded_file in uploaded_files:
        try:
            # Read and decode image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            if image is None:
                processed_images[uploaded_file.name] = {
                    'success': False,
                    'error': "Invalid image file"
                }
                continue
            
            # Enhance image
            enhanced = enhance_image(image)
            
            # Detect face
            face = detect_face(enhanced)
            
            if not face:
                processed_images[uploaded_file.name] = {
                    'success': False,
                    'error': "No face detected"
                }
                continue
            
            # Calculate crop
            crop = calculate_crop(image, face, ratio_type)
            
            if not crop:
                processed_images[uploaded_file.name] = {
                    'success': False,
                    'error': "Could not calculate crop"
                }
                continue
            
            # Apply crop
            x1, y1, x2, y2 = crop
            cropped = image[y1:y2, x1:x2]
            
            # Resize to standard size
            if ratio_type == 'standard':
                output_size = (600, 690)  # 2:2.3 ratio
            else:
                output_size = (600, 600)  # Square
            
            resized = cv2.resize(cropped, output_size, interpolation=cv2.INTER_CUBIC)
            
            # Convert for display
            display_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            
            # Encode for download
            _, buffer = cv2.imencode('.jpg', resized, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            processed_images[uploaded_file.name] = {
                'image_bytes': buffer.tobytes(),
                'display_image': display_image,
                'success': True
            }
            
        except Exception as e:
            processed_images[uploaded_file.name] = {
                'success': False,
                'error': f"Processing error: {str(e)}"
            }
    
    return processed_images

def main():
    st.set_page_config(
        page_title="Passport Photo Cropper",
        page_icon="📸",
        layout="wide"
    )
    
    st.title("📸 Passport Photo Cropper")
    st.write("Automatically crop and resize photos for passport applications")
    
    # Settings
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Settings")
        ratio_type = st.radio(
            "Photo Type:",
            ["Standard (2x2.3cm)", "Square (1x1)"],
            index=0
        )
        ratio_type = "standard" if "Standard" in ratio_type else "square"
        
        st.info("""
        **Instructions:**
        1. Upload photos with clear front-facing faces
        2. Click 'Process Photos'
        3. Download your cropped photos
        """)
    
    with col2:
        st.subheader("Upload Photos")
        uploaded_files = st.file_uploader(
            "Choose images",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            help="Select one or more photos to process"
        )
    
    # Initialize session state
    if 'processed' not in st.session_state:
        st.session_state.processed = False
    if 'processed_images' not in st.session_state:
        st.session_state.processed_images = {}
    
    # Process files
    if uploaded_files:
        st.success(f"📁 Found {len(uploaded_files)} file(s)")
        
        if not st.session_state.processed:
            if st.button("🚀 Process Photos", type="primary", use_container_width=True):
                with st.spinner("Processing images..."):
                    progress_bar = st.progress(0)
                    
                    processed_images = process_uploaded_files(uploaded_files, ratio_type)
                    st.session_state.processed_images = processed_images
                    st.session_state.processed = True
                    
                    progress_bar.progress(100)
                    st.rerun()
    
    # Display results
    if st.session_state.processed and st.session_state.processed_images:
        processed_images = st.session_state.processed_images
        successful = [k for k, v in processed_images.items() if v['success']]
        failed = [k for k, v in processed_images.items() if not v['success']]
        
        st.subheader("Results")
        
        if successful:
            st.success(f"✅ Processed {len(successful)} photo(s) successfully")
            
            # Display processed images
            st.subheader("Processed Photos")
            cols = st.columns(2)
            
            for i, filename in enumerate(successful):
                data = processed_images[filename]
                with cols[i % 2]:
                    st.image(
                        data['display_image'],
                        caption=filename,
                        use_container_width=True
                    )
            
            # Download button
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                for filename in successful:
                    data = processed_images[filename]
                    zip_file.writestr(filename, data['image_bytes'])
            
            zip_buffer.seek(0)
            
            st.download_button(
                label="📥 Download All Photos",
                data=zip_buffer,
                file_name="passport_photos.zip",
                mime="application/zip",
                use_container_width=True
            )
        
        if failed:
            st.warning(f"❌ Failed to process {len(failed)} photo(s)")
            
            with st.expander("Show failed photos"):
                for filename in failed:
                    st.error(f"{filename}: {processed_images[filename]['error']}")
        
        # Reset button
        if st.button("🔄 Process New Photos", use_container_width=True):
            st.session_state.processed = False
            st.session_state.processed_images = {}
            st.rerun()
    
    elif not uploaded_files:
        st.info("👆 Upload photos above to get started")

if __name__ == "__main__":
    main()
