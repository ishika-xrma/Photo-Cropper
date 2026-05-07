import streamlit as st
import cv2
import numpy as np
import io
import zipfile


def enhance_image(image):
    """Improve image quality for better face detection."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    return cv2.filter2D(enhanced, -1, kernel)


def detect_face(image):
    """Detect faces using two cascades; return the largest face with padding."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascade_paths = [
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml',
        cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml',
    ]

    # FIX: collect detections as plain Python lists to avoid numpy array extend() shape errors
    all_faces = []
    for path in cascade_paths:
        detector = cv2.CascadeClassifier(path)
        detections = detector.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=7,
            minSize=(100, 100),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        # detectMultiScale returns an empty tuple (not an array) when nothing found
        if len(detections) > 0:
            all_faces.extend(detections.tolist())   # FIX: convert to list before extending

    if not all_faces:
        return None

    # Pick the largest face
    x, y, w, h = sorted(all_faces, key=lambda f: f[2] * f[3], reverse=True)[0]

    # Add 20 % padding, clamped to image bounds
    img_h, img_w = image.shape[:2]
    pad_w = int(w * 0.2)
    pad_h = int(h * 0.2)
    x = max(0, x - pad_w)
    y = max(0, y - pad_h)
    w = min(img_w - x, w + 2 * pad_w)
    h = min(img_h - y, h + 2 * pad_h)
    return (x, y, w, h)


def calculate_crop(image, face, ratio_type='standard'):
    """
    Return (x1, y1, x2, y2) crop coordinates that respect the target aspect
    ratio while keeping the face centred.
    ratio_type: 'standard' (2 : 2.3) | 'square' (1 : 1)
    """
    x, y, w, h = face
    img_h, img_w = image.shape[:2]

    target_ratio = (2.0 / 2.3) if ratio_type == 'standard' else 1.0
    face_fraction = 0.70 if ratio_type == 'standard' else 0.65

    face_center_x = x + w // 2
    face_center_y = y + h // 2

    target_height = int(h / face_fraction)
    target_width = int(target_height * target_ratio)

    # Initial crop box centred on face
    crop_x1 = face_center_x - target_width // 2
    crop_y1 = face_center_y - target_height // 2
    crop_x2 = crop_x1 + target_width
    crop_y2 = crop_y1 + target_height

    # Shift box into image bounds without changing its size
    if crop_x1 < 0:
        crop_x2 -= crop_x1
        crop_x1 = 0
    if crop_y1 < 0:
        crop_y2 -= crop_y1
        crop_y1 = 0
    if crop_x2 > img_w:
        crop_x1 -= crop_x2 - img_w
        crop_x2 = img_w
    if crop_y2 > img_h:
        crop_y1 -= crop_y2 - img_h
        crop_y2 = img_h

    # Clamp again after shift (image may simply be too small)
    crop_x1 = max(0, crop_x1)
    crop_y1 = max(0, crop_y1)
    crop_x2 = min(img_w, crop_x2)
    crop_y2 = min(img_h, crop_y2)

    final_w = crop_x2 - crop_x1
    final_h = crop_y2 - crop_y1

    # Trim to exact aspect ratio if needed
    current_ratio = final_w / final_h if final_h else 0
    if abs(current_ratio - target_ratio) > 0.05:
        if current_ratio > target_ratio:
            # Too wide — shrink width
            new_w = int(final_h * target_ratio)
            offset = (final_w - new_w) // 2
            crop_x1 += offset
            crop_x2 = crop_x1 + new_w
        else:
            # Too tall — shrink height
            new_h = int(final_w / target_ratio)
            offset = (final_h - new_h) // 2
            crop_y1 += offset
            crop_y2 = crop_y1 + new_h

    # FIX: re-clamp after ratio trim so coords stay inside image
    crop_x1 = max(0, crop_x1)
    crop_y1 = max(0, crop_y1)
    crop_x2 = min(img_w, crop_x2)
    crop_y2 = min(img_h, crop_y2)

    final_w = crop_x2 - crop_x1
    final_h = crop_y2 - crop_y1

    if final_w < 300 or final_h < 300:
        return None

    return (crop_x1, crop_y1, crop_x2, crop_y2)


def process_uploaded_files(uploaded_files, ratio_type):
    """Process all uploaded files and return a results dict."""
    output_size = (600, 690) if ratio_type == 'standard' else (600, 600)
    results = {}

    progress = st.progress(0, text="Starting…")
    total = len(uploaded_files)

    for i, uploaded_file in enumerate(uploaded_files):
        progress.progress((i + 1) / total, text=f"Processing {uploaded_file.name} ({i+1}/{total})")

        try:
            # FIX: read bytes immediately while the pointer is at position 0
            raw_bytes = uploaded_file.read()
            file_bytes = np.frombuffer(raw_bytes, dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            if image is None:
                raise ValueError("Could not decode image — unsupported or corrupt file.")

            enhanced = enhance_image(image)
            face = detect_face(enhanced)

            if face is None:
                raise ValueError("No face detected.")

            crop = calculate_crop(image, face, ratio_type)
            if crop is None:
                raise ValueError("Could not crop image (face too small or too close to edge).")

            x1, y1, x2, y2 = crop
            cropped = image[y1:y2, x1:x2]
            resized = cv2.resize(cropped, output_size, interpolation=cv2.INTER_CUBIC)

            _, buffer = cv2.imencode('.jpg', resized, [cv2.IMWRITE_JPEG_QUALITY, 95])
            results[uploaded_file.name] = {
                'success': True,
                'image_bytes': buffer.tobytes(),
                # BGR → RGB for Streamlit display
                'display_image': cv2.cvtColor(resized, cv2.COLOR_BGR2RGB),
            }

        except Exception as exc:
            results[uploaded_file.name] = {
                'success': False,
                'error': str(exc),
            }

    progress.empty()
    return results


def build_zip(successful: dict) -> bytes:
    """Pack all processed images into an in-memory ZIP and return its bytes."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in successful.items():
            zf.writestr(name, data['image_bytes'])
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

def main():
    st.title("Passport Photo Cropper")

    ratio_label = st.radio("Select photo type:", ("Standard (2×2.3 cm)", "Square (1×1)"))
    ratio_type = "standard" if ratio_label == "Standard (2×2.3 cm)" else "square"

    uploaded_files = st.file_uploader(
        "Upload images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )

    # --- session-state bootstrap ---
    for key, default in [
        ("processed", False),
        ("processed_images", {}),
        ("last_file_ids", []),
        ("last_ratio", None),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    # FIX: reset processed flag when the file set or ratio changes so the
    #      user always sees the Crop button for a fresh batch / new setting.
    current_ids = [f.file_id for f in uploaded_files] if uploaded_files else []
    if current_ids != st.session_state.last_file_ids or ratio_type != st.session_state.last_ratio:
        st.session_state.processed = False
        st.session_state.processed_images = {}
        st.session_state.last_file_ids = current_ids
        st.session_state.last_ratio = ratio_type

    # --- Crop button ---
    if uploaded_files and not st.session_state.processed:
        if st.button("🚀 Crop Photos", type="primary", use_container_width=True):
            st.session_state.processed_images = process_uploaded_files(uploaded_files, ratio_type)
            st.session_state.processed = True
            st.rerun()

    # --- Results ---
    if st.session_state.processed and st.session_state.processed_images:
        results = st.session_state.processed_images
        successful = {k: v for k, v in results.items() if v['success']}
        failed = {k: v for k, v in results.items() if not v['success']}

        st.subheader("Processing Results")
        st.info(f"✅ {len(successful)} succeeded · ❌ {len(failed)} failed · {len(results)} total")

        if successful:
            cols = st.columns(2)
            for i, (name, data) in enumerate(successful.items()):
                with cols[i % 2]:
                    st.image(data['display_image'], caption=f"✅ {name}", use_container_width=True)

            st.subheader("Download")
            # FIX: build ZIP bytes once and store in session state so the
            #      download button always serves the correct, up-to-date file.
            if "zip_bytes" not in st.session_state or st.session_state.get("zip_stale"):
                st.session_state.zip_bytes = build_zip(successful)
                st.session_state.zip_stale = False

            st.download_button(
                label="📥 Download All as ZIP",
                data=st.session_state.zip_bytes,
                file_name="passport_photos.zip",
                mime="application/zip",
                use_container_width=True,
            )

        if failed:
            st.subheader("Failed Images")
            for name, data in failed.items():
                st.error(f"**{name}**: {data['error']}")

        if st.button("🔄 Process New Photos", use_container_width=True):
            for key in ("processed", "processed_images", "last_file_ids",
                        "last_ratio", "zip_bytes", "zip_stale"):
                st.session_state.pop(key, None)
            st.rerun()

    elif not uploaded_files:
        st.info("👆 Upload one or more images above to get started.")


if __name__ == "__main__":
    main()
