# fake_news_app.py
# Ammu's Fake News Detection Project 🚀

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ----------------------
# 1. Load Default Dataset
# ----------------------
@st.cache_data
def load_data():
    df = pd.read_csv("fake_or_real_news.csv")
    return df

df = load_data()

# ----------------------
# 2. Train Model
# ----------------------
X = df['text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)
acc = accuracy_score(y_test, y_pred)

# ----------------------
# 3. Streamlit UI
# ----------------------
st.title("📰 Fake News Detection App")
st.write("Enter a news headline/article, upload a CSV, or auto-fill a sample!")

st.info(f"Model trained with Accuracy: **{acc*100:.2f}%**")

# Show dataset preview
if st.checkbox("Show Default Dataset Preview"):
    st.dataframe(df.head())

# ----------------------
# Single News Prediction
# ----------------------
st.subheader("🔹 Check a Single News Article")

# Auto-fill sample
if st.button("Use Sample Text"):
    sample_text = "Breaking: Scientists discover water on Mars!"
else:
    sample_text = ""

user_input = st.text_area("🖊️ Enter News Text Here:", sample_text)

if st.button("Predict News"):
    if user_input.strip() == "":
        st.warning("Please enter some news text.")
    else:
        user_vec = vectorizer.transform([user_input])
        prediction = model.predict(user_vec)[0]

        if prediction == "REAL":
            st.success("✅ This news looks **REAL**")
        else:
            st.error("❌ This news looks **FAKE**")

# ----------------------
# Bulk Prediction (CSV Upload)
# ----------------------
st.subheader("🔹 Upload CSV for Bulk Prediction")
uploaded_file = st.file_uploader("Upload a CSV file with a column 'text'", type=["csv"])

if uploaded_file is not None:
    user_df = pd.read_csv(uploaded_file)

    if 'text' not in user_df.columns:
        st.error("CSV must have a column named 'text'")
    else:
        st.write("Preview of uploaded data:")
        st.dataframe(user_df.head())

        # Transform and predict
        user_vecs = vectorizer.transform(user_df['text'])
        preds = model.predict(user_vecs)

        user_df['Prediction'] = preds
        st.write("✅ Predictions Completed:")
        st.dataframe(user_df.head())

        # Download results
        csv = user_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Predictions", csv, "predictions.csv", "text/csv")
