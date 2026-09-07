# 🎭 Which Celebrity Are You?

A face-matching web application that finds the celebrity who looks most similar to an uploaded photo using **FaceNet face embeddings** and **cosine similarity**.

The application detects a face from the uploaded image, generates its facial embedding, compares it with precomputed celebrity embeddings, and returns the closest celebrity match.

## 🚀 Live Demo

👉 [Try the App]([https://which-celebrity-your-are.streamlit.app](https://which-celebrity-you-are.streamlit.app/))

---

## ✨ Features

- 📷 Upload a JPG, JPEG, or PNG image
- 👤 Detect faces using MTCNN
- 🧠 Generate facial embeddings using FaceNet
- 🔍 Compare faces using cosine similarity
- 🎭 Find the closest celebrity match
- 🖼️ Display the matched celebrity image
- 📊 Show similarity score
- ⚡ Fast inference using precomputed celebrity embeddings
- 🌐 Deployed using Streamlit Community Cloud

---

## 🧠 How It Works

The application follows these steps:

```text
User uploads an image
        ↓
Face Detection using MTCNN
        ↓
Crop and resize detected face
        ↓
Generate FaceNet embedding
        ↓
Calculate cosine similarity
        ↓
Compare with celebrity embeddings
        ↓
Find highest similarity score
        ↓
Display celebrity name and image
