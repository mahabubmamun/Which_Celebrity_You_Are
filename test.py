# from tensorflow.keras.applications.vgg16 import preprocess_input
# from tensorflow.keras.applications import VGG16
# import numpy as np
# import pickle
# from tensorflow.keras.preprocessing import image
# from mtcnn import MTCNN
# import cv2
# from PIL import Image
# from sklearn.metrics.pairwise import cosine_similarity

# feature_list = np.array(pickle.load(open('embedding.pkl','rb')))
# filenames = pickle.load(open('filenames.pkl','rb'))

# model  = VGG16(weights='imagenet', include_top = False, input_shape = (224, 224, 3), pooling = 'avg')

# detector = MTCNN()

# # load image and face detect from saple
# sample_img = cv2.imread('sample\sahruk.jpg')
# results = detector.detect_faces(sample_img)
# x, y, width, height = results[0]['box']
# face = sample_img[y:y+height, x:x+width]

# # cv2.imshow('output', face)
# # cv2.waitKey(0)

# # extract features
# image = Image.fromarray(face)
# image = image.resize((224,224))

# face_array = np.asarray(image)
# face_array = face_array.astype('float32')
# expanded_img = np.expand_dims(face_array, axis = 0)
# preprocessed_img = preprocess_input(expanded_img)
# result = model.predict(preprocessed_img).flatten()

# # print(result)
# # print(result.shape)
# similarity = []

# for i in range(len(feature_list)):
#     similarity.append(cosine_similarity(result.reshape(1,-1), feature_list[i].reshape(1,-1))[0][0])
    
# index_pos = sorted(list(enumerate(similarity)), reverse=True, key = lambda x:x[1])[0][0]
# print(cosine_similarity(result.reshape(1,-1), feature_list[index_pos].reshape(1,-1)))

# temp_img = cv2.imread(filenames[index_pos])
# cv2.imshow('output', temp_img)
# cv2.waitKey(0)


import pickle
import numpy as np
import cv2

from mtcnn import MTCNN
from keras_facenet import FaceNet
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# 1. Load FaceNet
# =========================================================

embedder = FaceNet()

detector = MTCNN()


# =========================================================
# 2. Load stored embeddings
# =========================================================

feature_list = np.array(
    pickle.load(
        open("embedding.pkl", "rb")
    )
)

filenames = pickle.load(
    open("filenames.pkl", "rb")
)


print("Embeddings:", feature_list.shape)
print("Images:", len(filenames))


# =========================================================
# 3. Load test image
# =========================================================

sample_path = "sample/jerifa.jpg"

sample_img = cv2.imread(sample_path)

if sample_img is None:
    raise FileNotFoundError(
        f"Could not read image: {sample_path}"
    )


# =========================================================
# 4. BGR → RGB
# =========================================================

sample_rgb = cv2.cvtColor(
    sample_img,
    cv2.COLOR_BGR2RGB
)


# =========================================================
# 5. Detect faces
# =========================================================

results = detector.detect_faces(
    sample_rgb
)


if len(results) == 0:
    raise ValueError(
        "No face detected in test image."
    )


print("Faces detected:", len(results))


# =========================================================
# 6. Choose largest face
# =========================================================

best_face = max(
    results,
    key=lambda x: x["box"][2] * x["box"][3]
)

x, y, width, height = best_face["box"]


# Prevent negative coordinates
x = max(0, x)
y = max(0, y)


# Prevent out-of-bound coordinates
x2 = min(
    sample_rgb.shape[1],
    x + width
)

y2 = min(
    sample_rgb.shape[0],
    y + height
)


face = sample_rgb[
    y:y2,
    x:x2
]


if face.size == 0:
    raise ValueError(
        "Invalid face crop."
    )


# =========================================================
# 7. Resize to FaceNet input
# =========================================================

face = cv2.resize(
    face,
    (160, 160)
)


# =========================================================
# 8. Generate FaceNet embedding
# =========================================================

result = embedder.embeddings(
    np.expand_dims(face, axis=0)
)[0]


# =========================================================
# 9. L2 normalize
# =========================================================

result = result / np.linalg.norm(
    result
)


# =========================================================
# 10. Calculate cosine similarity
# =========================================================

similarity = cosine_similarity(
    result.reshape(1, -1),
    feature_list
)[0]


# =========================================================
# 11. Sort results
# =========================================================

top_indices = np.argsort(
    similarity
)[::-1]


# =========================================================
# 12. Show top 5 matches
# =========================================================

print("\nTop 5 matches:\n")

for rank, index in enumerate(
    top_indices[:5],
    start=1
):

    print(
        f"{rank}. "
        f"{filenames[index]} "
        f"--> {similarity[index]:.4f}"
    )


# =========================================================
# 13. Best match
# =========================================================

best_index = top_indices[0]

best_file = filenames[best_index]

best_score = similarity[best_index]


print("\n==============================")
print("BEST MATCH")
print("==============================")

print("Image:", best_file)
print("Similarity:", round(best_score, 4))


# =========================================================
# 14. Display detected face
# =========================================================

face_display = cv2.cvtColor(
    face,
    cv2.COLOR_RGB2BGR
)

cv2.imshow(
    "Detected Face",
    face_display
)


# =========================================================
# 15. Display best matching image
# =========================================================

matched_img = cv2.imread(
    best_file
)

cv2.imshow(
    "Best Match",
    matched_img
)


cv2.waitKey(0)
cv2.destroyAllWindows()