# ============================================
# Movie Sentiment Analysis
# NLTK + TF-IDF (with bigrams) + Logistic Regression
# Google Colab Complete Code
# ============================================

!pip install -q nltk scikit-learn gradio pandas


import pandas as pd
import re
import nltk
import gradio as gr

from google.colab import files

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# Download NLTK resources

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')


# Upload dataset

print("Upload IMDB Dataset.csv")

uploaded = files.upload()


# Load dataset

df = pd.read_csv("IMDB Dataset.csv")

print(df.head())
print(df.shape)



# ============================================
# Text Cleaning
# ============================================


stop_words = set(stopwords.words('english'))

# Keep sentiment important words
important_words = {
    "not",
    "no",
    "never",
    "nor",
    "none",
    "cannot"
}


lemmatizer = WordNetLemmatizer()



def clean_text(text):

    text = str(text).lower()


    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)


    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)


    # Remove special characters
    text = re.sub(r'[^a-z\s]', '', text)


    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()


    # Tokenize
    words = text.split()


    # Remove stopwords but keep negation words
    words = [
        word for word in words
        if word not in stop_words or word in important_words
    ]


    # Lemmatization
    words = [
        lemmatizer.lemmatize(word)
        for word in words
    ]


    return " ".join(words)



print("Cleaning data...")

df["clean_review"] = df["review"].apply(clean_text)


print("Cleaning completed")



# ============================================
# Label Encoding
# ============================================


df["sentiment"] = df["sentiment"].map(
    {
        "positive":1,
        "negative":0
    }
)



# ============================================
# TF-IDF Feature Extraction
# ============================================


tfidf = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1,2)
)


X = tfidf.fit_transform(
    df["clean_review"]
)


y = df["sentiment"]



# ============================================
# Train Test Split
# ============================================


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)



# ============================================
# Model Training
# ============================================


model = LogisticRegression(
    max_iter=1000
)


model.fit(
    X_train,
    y_train
)


print("Training completed")



# ============================================
# Evaluation
# ============================================


prediction = model.predict(X_test)


print(
    "Accuracy:",
    accuracy_score(y_test,prediction)
)


print(
    classification_report(
        y_test,
        prediction
    )
)



# ============================================
# Prediction Function
# ============================================


def predict_sentiment(review):


    cleaned = clean_text(review)


    vector = tfidf.transform(
        [cleaned]
    )


    result = model.predict(vector)[0]


    confidence = max(
        model.predict_proba(vector)[0]
    ) * 100



    if result == 1:

        return (
            "😊 Positive Review\n"
            f"Confidence: {confidence:.2f}%"
        )

    else:

        return (
            "😞 Negative Review\n"
            f"Confidence: {confidence:.2f}%"
        )



# ============================================
# Gradio Web App
# ============================================


app = gr.Interface(

    fn=predict_sentiment,

    inputs=gr.Textbox(
        lines=6,
        placeholder="Enter movie review..."
    ),

    outputs="text",

    title="Movie Review Sentiment Analysis",

    description=
    "NLP Sentiment Analysis using NLTK + TF-IDF + Logistic Regression"

)



app.launch(
    share=True
)
