# Sentiment-analysis-project 🐱‍👤
live project link : https://sentiment-analysis-project-nf4hztwbxwjd6bicemssbz.streamlit.a/
gmail : borsefurgesh274@gmail.com

Below is a **professional, GitHub-ready `README.md` description** for your **Sentiment Analysis Streamlit project**, written clearly and simply (perfect for a fresher profile).

# 💬 Sentiment Analysis App (NLP)

A **Sentiment Analysis web application** built using **Streamlit** and **Hugging Face Transformers**.
This app analyzes user-entered text and predicts whether the sentiment is **Positive** or **Negative** along with a confidence score.

> 👨‍💻 **Created by:** Durgesh Borse

## 🚀 Live Demo

🔗 *Deploy on Streamlit Cloud and add your link here*

## 📌 Features

* 🧠 Uses **pre-trained Transformer model** for sentiment analysis
* ✍️ Accepts **real-time user text input**
* 😊 Displays **Positive / Negative sentiment**
* 📊 Shows **confidence score**
* 🎨 Clean and attractive UI using **custom CSS**
* ⚡ Fast performance with **model caching**

# 🛠️ Tech Stack

* **Python**
* **Streamlit** – Web application framework
* **Hugging Face Transformers** – NLP model
* **PyTorch** – Deep learning backend

# 📂 Project Structure

```bash
├── app.py              # Streamlit application
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/sentiment-analysis-streamlit.git
cd sentiment-analysis-streamlit
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Application

```bash
streamlit run app.py
```

---

## 🧪 Example Input

```text
I don't like this product
```

### Output

* **Sentiment:** NEGATIVE 😠
* **Confidence:** 99%

---

## 🧠 How It Works

* The app uses `pipeline("sentiment-analysis")` from Hugging Face
* A **pre-trained DistilBERT model** analyzes the input text
* The model predicts the sentiment label and confidence score
* Streamlit renders results interactively on the web interface

---

## 📈 Future Enhancements

* 🔹 Add **Neutral sentiment**
* 🔹 Analyze **multiple texts at once**
* 🔹 Export results to **CSV**
* 🔹 Add **confidence bar chart**
* 🔹 Support for **other languages**




