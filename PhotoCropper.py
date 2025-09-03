import streamlit as st
import cv2
import numpy as np
import os
from datetime import datetime
import tempfile
import zipfile
import io


# Copy all your functions from app.py here
def enhance_image(image):
    """Improve image quality for better face detection"""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
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
            minSize=(100, 100),  # Reduced minimum face size for better detection
            flags=cv2.CASCADE_SCALE_IMAGE
        ))

    if faces:
        # Return largest face with padding
        face = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)[0]
        x, y, w, h = face
        # Add 20% padding around detected face
        pad_w = int(w * 0.2)
        pad_h = int(h * 0.2)
        x = max(0, x - pad_w)
        y = max(0, y - pad_h)
        w = min(image.shape[1] - x, w + 2 * pad_w)
        h = min(image.shape[0] - y, h + 2 * pad_h)
        return (x, y, w, h)
    return None


def calculate_crop(image, face, ratio_type='standard'):
    """
    Calculate crop with proper headroom while maintaining aspect ratio
    ratio_type: 'standard' (2:2.3) or 'square' (1:1)
    """
    x, y, w, h = face
    img_h, img_w = image.shape[:2]

    # Set target ratios
    if ratio_type == 'standard':
        target_ratio = 2.0 / 2.3  # Width/Height ratio for standard passport
    else:  # square
        target_ratio = 1.0

    # Calculate the center of the face
    face_center_x = x + w // 2
    face_center_y = y + h // 2

    # Calculate crop dimensions based on face size
    # For passport photos, the face should be about 70-80% of the image height
    if ratio_type == 'standard':
        target_height = int(h / 0.7)  # Face height is 70% of total height
    else:
        target_height = int(h / 0.65)  # Face height is 65% of total height for square

    target_width = int(target_height * target_ratio)

    # Calculate crop coordinates centered on face
    crop_x1 = max(0, face_center_x - target_width // 2)
    crop_y1 = max(0, face_center_y - target_height // 2)

    crop_x2 = min(img_w, crop_x1 + target_width)
    crop_y2 = min(img_h, crop_y1 + target_height)

    # Adjust if we're at image boundaries
    if crop_x2 - crop_x1 < target_width:
        # If we need more width, adjust accordingly
        if crop_x1 == 0:
            crop_x2 = min(img_w, target_width)
        else:
            crop_x1 = max(0, img_w - target_width)

    if crop_y2 - crop_y1 < target_height:
        # If we need more height, adjust accordingly
        if crop_y1 == 0:
            crop_y2 = min(img_h, target_height)
        else:
            crop_y1 = max(0, img_h - target_height)

    # Final adjustment to ensure we have the exact aspect ratio
    final_width = crop_x2 - crop_x1
    final_height = crop_y2 - crop_y1

    # If the aspect ratio doesn't match, adjust the larger dimension
    current_ratio = final_width / final_height

    if abs(current_ratio - target_ratio) > 0.05:  # Only adjust if significantly off
        if current_ratio > target_ratio:
            # Too wide, reduce width
            target_height = final_height
            target_width = int(target_height * target_ratio)
            crop_x1 = face_center_x - target_width // 2
            crop_x2 = crop_x1 + target_width
        else:
            # Too tall, reduce height
            target_width = final_width
            target_height = int(target_width / target_ratio)
            crop_y1 = face_center_y - target_height // 2
            crop_y2 = crop_y1 + target_height

    # Ensure we're still within image boundaries after adjustment
    crop_x1 = max(0, crop_x1)
    crop_y1 = max(0, crop_y1)
    crop_x2 = min(img_w, crop_x2)
    crop_y2 = min(img_h, crop_y2)

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

    for i, uploaded_file in enumerate(uploaded_files):
        try:
            # Read file bytes and store in memory
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            uploaded_file.seek(0)  # Reset file pointer for potential future reads

            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            if image is None:
                processed_images[uploaded_file.name] = {
                    'original_name': uploaded_file.name,
                    'success': False,
                    'error': "Invalid image file"
                }
                continue

            # Process using your existing functions
            enhanced = enhance_image(image)
            face = detect_face(enhanced)

            if face:
                crop = calculate_crop(image, face, ratio_type)
                if crop:
                    x1, y1, x2, y2 = crop
                    cropped = image[y1:y2, x1:x2]

                    # Define output size based on ratio type
                    if ratio_type == 'standard':
                        output_size = (600, 690)  # 2:2.3 ratio (600x690)
                    else:
                        output_size = (600, 600)  # Square ratio

                    # Resize while maintaining aspect ratio
                    resized = cv2.resize(cropped, output_size, interpolation=cv2.INTER_CUBIC)

                    # Convert to RGB for display
                    display_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

                    # Store processed image data
                    _, buffer = cv2.imencode('.jpg', resized)
                    processed_images[uploaded_file.name] = {
                        'original_name': uploaded_file.name,
                        'image_bytes': buffer.tobytes(),
                        'display_image': display_image,
                        'success': True
                    }
                else:
                    processed_images[uploaded_file.name] = {
                        'original_name': uploaded_file.name,
                        'success': False,
                        'error': "Could not crop image (face too small or at edge)"
                    }
            else:
                processed_images[uploaded_file.name] = {
                    'original_name': uploaded_file.name,
                    'success': False,
                    'error': "No face detected"
                }

        except Exception as e:
            processed_images[uploaded_file.name] = {
                'original_name': uploaded_file.name,
                'success': False,
                'error': f"Processing error: {str(e)}"
            }

    return processed_images


def main():
    st.title("Passport Photo Cropper")

    ratio_type = st.radio("Select photo type:", ("Standard (2x2.3cm)", "Square (1x1)"))
    ratio_type = "standard" if ratio_type == "Standard (2x2.3cm)" else "square"

    uploaded_files = st.file_uploader("Upload images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    # Initialize session state
    if 'processed' not in st.session_state:
        st.session_state.processed = False
    if 'processed_images' not in st.session_state:
        st.session_state.processed_images = {}
    if 'uploaded_files_info' not in st.session_state:
        st.session_state.uploaded_files_info = []

    # Store uploaded files info in session state
    if uploaded_files:
        st.session_state.uploaded_files_info = [
            {'name': f.name, 'size': f.size, 'type': f.type}
            for f in uploaded_files
        ]

    # Show crop button only if files are uploaded but not processed yet
    if uploaded_files and not st.session_state.processed:
        if st.button("🚀 Crop Photos", type="primary", use_container_width=True):
            with st.spinner("Processing images..."):
                st.session_state.processed_images = process_uploaded_files(uploaded_files, ratio_type)
                st.session_state.processed = True
            st.rerun()

    # Display results if processing is complete
    if st.session_state.processed and st.session_state.processed_images:
        processed_images = st.session_state.processed_images

        # Display results
        st.subheader("Processing Results")

        successful_images = {k: v for k, v in processed_images.items() if v['success']}

        # Display in a grid
        if successful_images:
            cols = st.columns(2)
            for i, (name, data) in enumerate(successful_images.items()):
                with cols[i % 2]:
                    st.image(data['display_image'], caption=f"Processed: {name}", use_container_width=True)

        # Show statistics
        st.info(f"Successfully processed {len(successful_images)} out of {len(processed_images)} images")

        # Create download options
        if successful_images:
            st.subheader("Download Options")

            # Create a ZIP file in memory
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                for name, data in successful_images.items():
                    # Add to ZIP with original filename
                    zip_file.writestr(name, data['image_bytes'])

            zip_buffer.seek(0)

            # Download all button
            st.download_button(
                label="📥 Download All Processed Photos",
                data=zip_buffer,
                file_name="passport_photos.zip",
                mime="application/zip",
                key="download_all",
                use_container_width=True
            )

        # Show errors if any
        failed_images = {k: v for k, v in processed_images.items() if not v['success']}
        if failed_images:
            st.subheader("Failed to Process")
            for name, data in failed_images.items():
                st.error(f"{name}: {data['error']}")

        # Add a reset button
        if st.button("🔄 Process New Photos", use_container_width=True):
            st.session_state.processed = False
            st.session_state.processed_images = {}
            st.session_state.uploaded_files_info = []
            st.rerun()

    # Show instructions if no files uploaded
    elif not uploaded_files and not st.session_state.uploaded_files_info:
        st.info("👆 Please upload images above to get started!")


if __name__ == "__main__":
    main()