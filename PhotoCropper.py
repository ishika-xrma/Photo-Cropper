import streamlit as st
import cv2
import numpy as np
import zipfile
import tempfile
import os
from io import BytesIO


# ===== Your original functions =====

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
        [-1,-1,-1],
        [-1,9,-1],
        [-1,-1,-1]
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

    detectors = [
        cv2.CascadeClassifier(
            cv2.data.haarcascades +
            'haarcascade_frontalface_default.xml'
        ),

        cv2.CascadeClassifier(
            cv2.data.haarcascades +
            'haarcascade_frontalface_alt2.xml'
        )
    ]

    faces = []

    for detector in detectors:

        found = detector.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=7,
            minSize=(200, 200),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        faces.extend(found)

    if faces:

        face = sorted(
            faces,
            key=lambda x: x[2] * x[3],
            reverse=True
        )[0]

        x, y, w, h = face

        pad = int(w * 0.25)

        x = max(0, x - pad)
        y = max(0, y - pad)

        w = min(
            image.shape[1] - x,
            w + 2 * pad
        )

        h = min(
            image.shape[0] - y,
            h + 2 * pad
        )

        return (x, y, w, h)

    return None


def calculate_crop(image, face, ratio_type='standard'):

    x, y, w, h = face

    img_h, img_w = image.shape[:2]

    if ratio_type == 'standard':
        target_ratio = 2 / 2.3
        face_height_ratio = 0.65
        headroom_ratio = 0.13

    else:
        target_ratio = 1
        face_height_ratio = 0.6
        headroom_ratio = 0.22

    target_height = int(
        h / face_height_ratio
    )

    target_width = int(
        target_height * target_ratio
    )

    face_center_x = x + w // 2
    head_top = y

    crop_x1 = max(
        0,
        face_center_x - target_width // 2
    )

    crop_y1 = max(
        0,
        head_top - int(
            target_height *
            headroom_ratio
        )
    )

    crop_x2 = min(
        img_w,
        crop_x1 + target_width
    )

    crop_y2 = min(
        img_h,
        crop_y1 + target_height
    )

    if crop_x2 - crop_x1 < target_width:

        if crop_x1 == 0:
            crop_x2 = min(
                img_w,
                target_width
            )
        else:
            crop_x1 = max(
                0,
                img_w - target_width
            )

    if crop_y2 - crop_y1 < target_height:

        if crop_y1 == 0:
            crop_y2 = min(
                img_h,
                target_height
            )
        else:
            crop_y1 = max(
                0,
                img_h - target_height
            )

    return (
        crop_x1,
        crop_y1,
        crop_x2,
        crop_y2
    )


def process_uploaded_image(
    uploaded_file,
    ratio_type
):

    file_bytes = np.asarray(
        bytearray(
            uploaded_file.read()
        ),
        dtype=np.uint8
    )

    image = cv2.imdecode(
        file_bytes,
        1
    )

    enhanced = enhance_image(
        image
    )

    face = detect_face(
        enhanced
    )

    if face is None:
        return None

    crop = calculate_crop(
        image,
        face,
        ratio_type
    )

    x1, y1, x2, y2 = crop

    cropped = image[
        y1:y2,
        x1:x2
    ]

    if ratio_type == "standard":
        size = (600, 690)
    else:
        size = (600, 600)

    result = cv2.resize(
        cropped,
        size,
        interpolation=cv2.INTER_CUBIC
    )

    _, buffer = cv2.imencode(
        ".jpg",
        result,
        [cv2.IMWRITE_JPEG_QUALITY, 95]
    )

    return buffer.tobytes()


# ===== Streamlit UI =====

st.set_page_config(
    page_title="Passport Photo Cropper",
    page_icon="📸"
)

st.title("Passport Photo Cropper")

ratio = st.selectbox(
    "Select ratio",
    ["standard", "square"]
)

uploaded_files = st.file_uploader(
    "Upload images",
    accept_multiple_files=True,
    type=["jpg", "jpeg", "png"]
)

if uploaded_files:

    zip_buffer = BytesIO()

    with zipfile.ZipFile(
        zip_buffer,
        "w"
    ) as zip_file:

        for file in uploaded_files:

            result = process_uploaded_image(
                file,
                ratio
            )

            if result:

                zip_file.writestr(
                    f"{file.name}.jpg",
                    result
                )

                st.image(
                    result,
                    caption=file.name
                )

            else:

                st.error(
                    f"No face found in {file.name}"
                )

    st.download_button(
        "Download ZIP",
        zip_buffer.getvalue(),
        "passport_photos.zip",
        "application/zip"
    )
