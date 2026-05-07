import streamlit as st
import cv2
import numpy as np
import io
import zipfile


def enhance_image(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=3.0,
        tileGridSize=(8, 8)
    )

    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))

    enhanced = cv2.cvtColor(
        lab,
        cv2.COLOR_LAB2BGR
    )

    kernel = np.array([
        [-1, -1, -1],
        [-1,  9, -1],
        [-1, -1, -1]
    ])

    return cv2.filter2D(
        enhanced,
        -1,
        kernel
    )


def detect_face(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    cascade_paths = [
        cv2.data.haarcascades +
        "haarcascade_frontalface_default.xml",

        cv2.data.haarcascades +
        "haarcascade_frontalface_alt2.xml",
    ]

    all_faces = []

    for path in cascade_paths:

        detector = cv2.CascadeClassifier(
            path
        )

        detections = detector.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=7,
            minSize=(100, 100),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        if len(detections) > 0:
            all_faces.extend(
                detections.tolist()
            )

    if not all_faces:
        return None

    x, y, w, h = sorted(
        all_faces,
        key=lambda f: f[2] * f[3],
        reverse=True
    )[0]

    img_h, img_w = image.shape[:2]

    pad_w = int(w * 0.2)
    pad_h = int(h * 0.2)

    x = max(0, x - pad_w)
    y = max(0, y - pad_h)

    w = min(
        img_w - x,
        w + 2 * pad_w
    )

    h = min(
        img_h - y,
        h + 2 * pad_h
    )

    return (x, y, w, h)


def calculate_crop(
    image,
    face,
    ratio_type
):

    x, y, w, h = face

    img_h, img_w = image.shape[:2]

    target_ratio = (
        2 / 2.3
        if ratio_type == "standard"
        else 1
    )

    face_fraction = (
        0.70
        if ratio_type == "standard"
        else 0.65
    )

    center_x = x + w // 2
    center_y = y + h // 2

    target_height = int(
        h / face_fraction
    )

    target_width = int(
        target_height *
        target_ratio
    )

    x1 = center_x - target_width // 2
    y1 = center_y - target_height // 2

    x2 = x1 + target_width
    y2 = y1 + target_height

    if x1 < 0:
        x2 -= x1
        x1 = 0

    if y1 < 0:
        y2 -= y1
        y1 = 0

    if x2 > img_w:
        x1 -= (x2 - img_w)
        x2 = img_w

    if y2 > img_h:
        y1 -= (y2 - img_h)
        y2 = img_h

    x1 = max(0, x1)
    y1 = max(0, y1)

    x2 = min(img_w, x2)
    y2 = min(img_h, y2)

    if (
        x2 - x1 < 300 or
        y2 - y1 < 300
    ):
        return None

    return (
        x1,
        y1,
        x2,
        y2
    )


def process_uploaded_files(
    uploaded_files,
    ratio_type
):

    output_size = (
        (600, 690)
        if ratio_type == "standard"
        else (600, 600)
    )

    results = {}

    progress = st.progress(
        0,
        text="Starting..."
    )

    total = len(uploaded_files)

    for i, file in enumerate(uploaded_files):

        progress.progress(
            (i + 1) / total,
            text=f"Processing {file.name}"
        )

        try:

            # FIXED
            raw_bytes = file.getvalue()

            file_bytes = np.frombuffer(
                raw_bytes,
                dtype=np.uint8
            )

            image = cv2.imdecode(
                file_bytes,
                cv2.IMREAD_COLOR
            )

            if image is None:
                raise ValueError(
                    "Invalid image"
                )

            enhanced = enhance_image(
                image
            )

            face = detect_face(
                enhanced
            )

            if face is None:
                raise ValueError(
                    "No face detected"
                )

            crop = calculate_crop(
                image,
                face,
                ratio_type
            )

            if crop is None:
                raise ValueError(
                    "Crop failed"
                )

            x1, y1, x2, y2 = crop

            cropped = image[
                y1:y2,
                x1:x2
            ]

            resized = cv2.resize(
                cropped,
                output_size,
                interpolation=cv2.INTER_CUBIC
            )

            _, buffer = cv2.imencode(
                ".jpg",
                resized,
                [
                    cv2.IMWRITE_JPEG_QUALITY,
                    95
                ]
            )

            results[file.name] = {
                "success": True,
                "image_bytes": buffer.tobytes(),
                "display_image": cv2.cvtColor(
                    resized,
                    cv2.COLOR_BGR2RGB
                )
            }

        except Exception as e:

            results[file.name] = {
                "success": False,
                "error": str(e)
            }

    progress.empty()

    return results


def build_zip(successful):

    buf = io.BytesIO()

    with zipfile.ZipFile(
        buf,
        "w",
        zipfile.ZIP_DEFLATED
    ) as zf:

        for name, data in successful.items():

            zf.writestr(
                name,
                data["image_bytes"]
            )

    return buf.getvalue()


def main():

    st.title(
        "Passport Photo Cropper"
    )

    ratio_label = st.radio(
        "Select photo type:",
        (
            "Standard (2×2.3 cm)",
            "Square (1×1)"
        )
    )

    ratio_type = (
        "standard"
        if "Standard" in ratio_label
        else "square"
    )

    uploaded_files = st.file_uploader(
        "Upload images",
        type=[
            "jpg",
            "jpeg",
            "png"
        ],
        accept_multiple_files=True
    )

    if "processed" not in st.session_state:
        st.session_state.processed = False

    if "processed_images" not in st.session_state:
        st.session_state.processed_images = {}

    if "last_files" not in st.session_state:
        st.session_state.last_files = []

    # FIXED
    current_files = [
        f"{f.name}_{f.size}"
        for f in uploaded_files
    ] if uploaded_files else []

    if current_files != st.session_state.last_files:

        st.session_state.processed = False

        st.session_state.processed_images = {}

        st.session_state.last_files = (
            current_files
        )

    if (
        uploaded_files and
        not st.session_state.processed
    ):

        if st.button(
            "🚀 Crop Photos",
            use_container_width=True
        ):

            st.session_state.processed_images = (
                process_uploaded_files(
                    uploaded_files,
                    ratio_type
                )
            )

            st.session_state.processed = True

            st.rerun()

    if st.session_state.processed:

        results = (
            st.session_state
            .processed_images
        )

        successful = {
            k: v
            for k, v
            in results.items()
            if v["success"]
        }

        failed = {
            k: v
            for k, v
            in results.items()
            if not v["success"]
        }

        if successful:

            cols = st.columns(2)

            for i, (
                name,
                data
            ) in enumerate(
                successful.items()
            ):

                with cols[i % 2]:

                    st.image(
                        data[
                            "display_image"
                        ],
                        caption=name,
                        use_container_width=True
                    )

            zip_bytes = build_zip(
                successful
            )

            st.download_button(
                "📥 Download ZIP",
                zip_bytes,
                "passport_photos.zip",
                "application/zip",
                use_container_width=True
            )

        if failed:

            for name, data in failed.items():

                st.error(
                    f"{name}: "
                    f"{data['error']}"
                )


if __name__ == "__main__":
    main()
