import streamlit as st
import numpy as np
import zipfile
import io
import os
from PIL import Image

# Set page config must be the first Streamlit command
st.set_page_config(
    page_title="Passport Photo Cropper",
    page_icon="📸",
    layout="wide"
)

# Try to import OpenCV with fallback
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    st.error("❌ OpenCV not available - using fallback methods")
    st.stop()

# Display versions for debugging
st.sidebar.success("✅ OpenCV loaded successfully")
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
    """Calculate crop with precise headroom control"""
    try:
        x, y, w, h = face
        img_h, img_w = image.shape[:2]
        
        # Set aspect ratio
        if ratio_type == 'standard':
            target_ratio = 2.0 / 2.3
        else:  # square
            target_ratio = 1.0
        
        # Calculate the center of the face
        face_center_x = x + w // 2
        face_center_y = y + h // 2
        
        # For passport photos, standard guidelines:
        # - Face should be 70-80% of the photo height
        # - Significant headroom above (about 25-30% of total height)
        
        if ratio_type == 'standard':
            # Face height = 70% of total height
            target_height = int(h / 0.7)
            # Headroom = 25% of total height above the head
            headroom = int(target_height * 0.25)
        else:
            # Face height = 65% of total height for square
            target_height = int(h / 0.65)
            # Headroom = 30% of total height above the head
            headroom = int(target_height * 0.3)
        
        target_width = int(target_height * target_ratio)
        
        # Calculate crop coordinates
        crop_x1 = max(0, face_center_x - target_width // 2)
        
        # Position the crop so there's proper headroom above the face
        # The top of the head should be at (headroom) pixels from the top
        crop_y1 = max(0, y - headroom)
        
        crop_x2 = min(img_w, crop_x1 + target_width)
        crop_y2 = min(img_h, crop_y1 + target_height)
        
        # Adjust if we're at image boundaries
        if crop_x2 - crop_x1 < target_width:
            if crop_x1 == 0:
                crop_x2 = min(img_w, crop_x1 + target_width)
            else:
                crop_x1 = max(0, crop_x2 - target_width)
        
        if crop_y2 - crop_y1 < target_height:
            if crop_y1 == 0:
                crop_y2 = min(img_h, crop_y1 + target_height)
            else:
                crop_y1 = max(0, crop_y2 - target_height)
        
        # Final check to ensure we have proper headroom
        final_headroom = y - crop_y1
        if final_headroom < headroom * 0.5:  # At least 50% of desired headroom
            st.warning(f"Limited headroom available for {uploaded_file.name}")
        
        # Verify minimum dimensions
        min_dim = 300
        if crop_x2 - crop_x1 < min_dim or crop_y2 - crop_y1 < min_dim:
            return None
        
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
            
            # Convert for display (BGR to RGB) and to PIL Image
            rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
            
            # Encode for download
            _, buffer = cv2.imencode('.jpg', resized, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            processed_images[uploaded_file.name] = {
                'image_bytes': buffer.tobytes(),
                'display_image': pil_image,  # Store as PIL Image
                'success': True
            }
            
        except Exception as e:
            processed_images[uploaded_file.name] = {
                'success': False,
                'error': f"Processing error: {str(e)}"
            }
    
    return processed_images

def main():
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
            if st.button("🚀 Process Photos", type="primary"):
                with st.spinner("Processing images..."):
                    processed_images = process_uploaded_files(uploaded_files, ratio_type)
                    st.session_state.processed_images = processed_images
                    st.session_state.processed = True
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
                    try:
                        # Use use_column_width instead of use_container_width
                        st.image(
                            data['display_image'],  # PIL Image
                            caption=filename,
                            use_column_width=True  # Changed parameter name
                        )
                    except Exception as e:
                        st.error(f"Error displaying image {filename}: {e}")
                        st.write(f"Processed: {filename}")
            
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
                mime="application/zip"
            )
        
        if failed:
            st.warning(f"❌ Failed to process {len(failed)} photo(s)")
            
            with st.expander("Show failed photos"):
                for filename in failed:
                    st.error(f"{filename}: {processed_images[filename]['error']}")
        
        # Reset button
        if st.button("🔄 Process New Photos"):
            st.session_state.processed = False
            st.session_state.processed_images = {}
            st.rerun()
    
    elif not uploaded_files:
        st.info("👆 Upload photos above to get started")

if __name__ == "__main__":
    main()


