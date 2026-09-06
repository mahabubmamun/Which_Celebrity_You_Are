import streamlit as st
import cv2
import numpy as np
import pickle
from PIL import Image
from keras_facenet import FaceNet
from mtcnn import MTCNN
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Which Celebrity Are You?",
    page_icon="🎭",
    layout="wide"
)


# --------------------------------------------------
# Load Models / Data
# --------------------------------------------------

@st.cache_resource
def load_models():

    embedder = FaceNet()
    detector = MTCNN()

    return embedder, detector


@st.cache_data
def load_embeddings():

    with open("embedding.pkl", "rb") as f:
        feature_list = pickle.load(f)

    with open("filenames.pkl", "rb") as f:
        filenames = pickle.load(f)

    return np.array(feature_list), filenames


embedder, detector = load_models()
feature_list, filenames = load_embeddings()


# --------------------------------------------------
# Face Detection
# --------------------------------------------------

def extract_face(image):

    # PIL -> numpy
    image = np.array(image)

    # Make sure RGB
    if image.shape[-1] == 4:
        image = image[:, :, :3]

    # Detect faces
    results = detector.detect_faces(image)

    if len(results) == 0:
        return None

    # Select largest face
    def face_area(face):
        x, y, w, h = face["box"]
        return w * h

    best_face = max(results, key=face_area)

    x, y, w, h = best_face["box"]

    # Prevent negative coordinates
    x = max(0, x)
    y = max(0, y)

    x2 = min(image.shape[1], x + w)
    y2 = min(image.shape[0], y + h)

    face = image[y:y2, x:x2]

    if face.size == 0:
        return None

    # Resize for FaceNet
    face = cv2.resize(face, (160, 160))

    return face


# --------------------------------------------------
# Generate FaceNet Embedding
# --------------------------------------------------

def get_embedding(face):

    face = np.expand_dims(face, axis=0)

    embedding = embedder.embeddings(face)[0]

    # L2 normalization
    embedding = embedding / np.linalg.norm(embedding)

    return embedding


# --------------------------------------------------
# Find Best Celebrity
# --------------------------------------------------

def find_best_match(embedding):

    similarity = cosine_similarity(
        embedding.reshape(1, -1),
        feature_list
    )[0]

    best_index = np.argmax(similarity)

    best_score = similarity[best_index]

    best_filename = filenames[best_index]

    return best_filename, best_score


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.title("🎭 Celebrity Matcher")

st.sidebar.write(
    """
    Upload a photo containing a face.

    The application will:
    
    1. Detect your face
    2. Generate a FaceNet embedding
    3. Compare it with celebrity embeddings
    4. Find the closest match
    """
)

st.sidebar.info(
    "For best results, upload a clear front-facing photo."
)


# --------------------------------------------------
# Main UI
# --------------------------------------------------

st.title("🎭 Which Celebrity Are You?")
st.subheader("Upload your photo and find your closest celebrity match!")

st.write(
    "Upload a clear photo containing one face."
)


uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)


# --------------------------------------------------
# Process Image
# --------------------------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    # Display uploaded image
    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Your Photo")

        st.image(
            image,
            use_container_width=True
        )


    # Button
    if st.button("🔍 Find My Celebrity", use_container_width=True):

        with st.spinner("Analyzing your face..."):

            # Detect face
            face = extract_face(image)

            if face is None:

                st.error(
                    "❌ No face detected. Please upload a clear photo."
                )

            else:

                # Generate embedding
                embedding = get_embedding(face)

                # Find match
                best_filename, best_score = find_best_match(
                    embedding
                )

                # Convert Windows path to Linux-compatible path
                best_filename = best_filename.replace("\\", "/")

                # Extract celebrity name
                celebrity_name = best_filename.split("/")[-2]


                # --------------------------------------------------
                # Result
                # --------------------------------------------------

                with col2:

                    st.subheader("Your Celebrity Match")

                    try:
                        celebrity_image = Image.open(best_filename)

                        st.image(
                            celebrity_image,
                            use_container_width=True
                        )

                        # image_path = Path(best_filename)

                        # st.write("Image path:", image_path)
                        # st.write("Image exists:", image_path.exists())

                        # if image_path.exists():

                        #     celebrity_image = Image.open(image_path)

                        #     st.image(
                        #         celebrity_image,
                        #         use_container_width=True
                        #     )

                        # else:

                        #     st.error(
                        #         f"Celebrity image not found: {image_path}"
                        #     )

                    except Exception as e:

                        st.error(f"Error loading celebrity image: {e}")


                st.success(
                    f"🎉 Your closest celebrity match is **{celebrity_name}**!"
                )

                st.metric(
                    "Similarity",
                    f"{best_score * 100:.2f}%"
                )


                # --------------------------------------------------
                # Confidence Message
                # --------------------------------------------------

                if best_score >= 0.70:

                    st.success(
                        "🔥 Very strong match!"
                    )

                elif best_score >= 0.55:

                    st.info(
                        "👍 Good match, but there is some uncertainty."
                    )

                else:

                    st.warning(
                        "⚠️ Low similarity. The result may not be reliable."
                    )