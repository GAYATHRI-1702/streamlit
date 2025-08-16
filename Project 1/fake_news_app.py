# fake_news_app.py
# Ammu's Final Week Project 🚀
# Fake News Detection with Streamlit

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
st.title("📰 Fake News Detection App")
st.write("App started ✅")  

# ----------------------
# 1. Load Dataset
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

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Accuracy
y_pred = model.predict(X_test_tfidf)
acc = accuracy_score(y_test, y_pred)

# ----------------------
# 3. Streamlit UI
# ----------------------
st.title("📰 Fake News Detection App")
st.write("Enter a news headline or article, and I’ll tell you if it's **Real** or **Fake**!")

# Show dataset preview
if st.checkbox("Show Dataset Preview"):
    st.dataframe(df.head())

# Accuracy display
st.info(f"Model trained with Accuracy: **{acc*100:.2f}%**")

# User input
user_input = st.text_area("🖊️ Enter News Text Here:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some news text.")
    else:
        # Transform input
        user_vec = vectorizer.transform([user_input])
        prediction = model.predict(user_vec)[0]

        # Show result
        if prediction == "REAL":
            st.success("✅ This news looks **REAL**")
        else:
            st.error("❌ This news looks **FAKE**")
