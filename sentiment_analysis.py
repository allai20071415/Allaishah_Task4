import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix, ConfusionMatrixDisplay
from wordcloud import WordCloud

BASE=Path(__file__).resolve().parents[1]
DATA=BASE/"data"
OUT=BASE/"outputs"
OUT.mkdir(exist_ok=True)

nltk.download("stopwords",quiet=True)
nltk.download("wordnet",quiet=True)

def find_csv():
    files=list(DATA.glob("*.csv")) if DATA.exists() else []
    if not files:
        return None
    for f in files:
        d=pd.read_csv(f,nrows=3)
        cols={str(c).strip().lower() for c in d.columns}
        if "sentiment" in cols and ({"translated_review","review","text"} & cols):
            return f
    raise ValueError("No compatible CSV found. Expected Sentiment and Translated_Review, Review, or Text.")


def create_demo_dataset():
    """Create a small built-in dataset when the Kaggle CSV is not present."""
    examples = {
        "Positive": [
            "I love this app it is excellent and very useful",
            "Amazing application works perfectly",
            "Great app easy to use and very helpful",
            "The app is fantastic and fast",
            "Very good application I recommend it",
            "Excellent service and wonderful experience",
            "This app makes my life easier",
            "I am very happy with this application",
            "Brilliant app with a simple interface",
            "The latest update is fantastic",
            "Really useful app and very reliable",
            "I enjoy using this application every day",
            "Super helpful and works smoothly",
            "Wonderful application with great features",
            "The app is awesome and convenient",
            "Fast clean and easy to use",
            "I highly recommend this excellent app",
            "This is one of the best apps I have used",
            "Very satisfied with the app",
            "The experience is great and simple",
            "Helpful features and excellent design",
            "The application works beautifully",
            "Good quality and very useful tools",
            "I am impressed with this app",
            "Fantastic performance after the update",
            "Easy navigation and excellent features",
            "The app saves me a lot of time",
            "Very happy with the new version",
            "Perfect application for my needs",
            "Great user experience and useful features",
        ],
        "Negative": [
            "I hate this app it is terrible",
            "The application crashes every time",
            "Very bad app and completely useless",
            "The latest update broke everything",
            "Terrible experience and very slow",
            "I am disappointed with this application",
            "The app keeps freezing on my phone",
            "Worst application I have ever used",
            "It does not work properly",
            "The application is frustrating and buggy",
            "Very poor performance after the update",
            "I cannot open the app anymore",
            "The app crashes whenever I try to login",
            "Bad design and difficult to use",
            "This application wastes my time",
            "I am unhappy with the new version",
            "The service is unreliable and broken",
            "Too many errors and crashes",
            "The app is slow and annoying",
            "I regret installing this application",
            "Nothing works correctly",
            "Very disappointing user experience",
            "The application stopped working",
            "The update made the app much worse",
            "Terrible performance and poor support",
            "The app freezes constantly",
            "I cannot use the main feature",
            "Very frustrating application",
            "This is a useless and broken app",
            "Poor quality and full of bugs",
        ],
        "Neutral": [
            "The app is okay and works as expected",
            "It is an average application",
            "The application has some useful features",
            "The app works but nothing special",
            "It is fine for basic use",
            "The application is neither good nor bad",
            "The app has both good and bad features",
            "It works sometimes depending on the device",
            "The design is simple and standard",
            "The application provides basic functionality",
            "It is an ordinary app for daily use",
            "The latest version is acceptable",
            "Some features are useful while others are not",
            "The app is okay but could be improved",
            "I have mixed feelings about this application",
            "It does what it says with average performance",
            "The application is fairly standard",
            "The features are reasonable",
            "The app is usable but not impressive",
            "It is neither fast nor slow",
            "The experience is normal",
            "The application meets basic requirements",
            "There are some advantages and disadvantages",
            "The update changed a few things",
            "It is suitable for simple tasks",
            "The interface is basic",
            "The app works on my device",
            "The application is acceptable",
            "Some parts are good and some need improvement",
            "Overall the app is average",
        ]
    }
    rows=[]
    for label,texts in examples.items():
        for i,text in enumerate(texts):
            rows.append({"text": text + f" for daily use {i}", "sentiment": label})
    # Ambiguous reviews make error analysis more realistic.
    ambiguous=[
        ("Good features but the app crashes sometimes", "Negative"),
        ("I like it but the new update is slow", "Negative"),
        ("The design is nice but nothing works", "Negative"),
        ("It works well but I dislike the update", "Negative"),
        ("The app is okay but sometimes frustrating", "Neutral"),
        ("Useful idea although performance is poor", "Neutral"),
        ("Nice interface but it does not always work", "Neutral"),
        ("The app is useful but I am not happy", "Neutral"),
        ("Excellent features but too many bugs", "Negative"),
        ("Bad performance but a very good interface", "Negative"),
    ]
    for text,label in ambiguous:
        for i in range(3):
            rows.append({"text":f"{text} example {i}","sentiment":label})
    return pd.DataFrame(rows)

def clean_text(text,stop_words,lemmatizer):
    text=str(text).lower()
    text=re.sub(r"http\S+|www\S+"," ",text)
    text=re.sub(r"@\w+|#\w+"," ",text)
    text=re.sub(r"[^a-z\s]"," ",text)
    words=text.split()
    words=[lemmatizer.lemmatize(w) for w in words if w not in stop_words and len(w)>2]
    return " ".join(words)

def main():
    csv=find_csv()
    if csv is None:
        print("No CSV found in data/. Using the built-in demonstration dataset.")
        print("Kaggle Play Store User Reviews is the reference dataset and is not included in this repository.\n")
        data=create_demo_dataset()
    else:
        print("Using dataset:",csv)
        df=pd.read_csv(csv)
        norm={str(c).strip().lower():c for c in df.columns}
        text_col=norm.get("translated_review") or norm.get("review") or norm.get("text")
        sent_col=norm.get("sentiment")
        if not text_col or not sent_col:
            raise ValueError(f"Could not detect columns. Found: {list(df.columns)}")

        data=df[[text_col,sent_col]].copy()
        data.columns=["text","sentiment"]

    data["text"]=data["text"].astype(str).str.strip()
    data["sentiment"]=data["sentiment"].astype(str).str.strip().str.title()
    data=data.dropna(subset=["text","sentiment"])
    data=data[data.sentiment.isin(["Positive","Neutral","Negative"])]
    data=data.drop_duplicates(subset=["text"]).reset_index(drop=True)

    print("Dataset:",csv)
    print("Shape:",data.shape)
    print("\nClass distribution:")
    print(data.sentiment.value_counts())

    order=["Positive","Neutral","Negative"]
    plt.figure(figsize=(7,5))
    counts = data["sentiment"].value_counts().reindex(order, fill_value=0)
    plt.bar(counts.index, counts.values)
    plt.title("Play Store Review Sentiment Distribution")
    plt.tight_layout()
    plt.savefig(OUT/"sentiment_distribution.png",dpi=200)
    plt.close()

    try:
        sw=set(stopwords.words("english"))
    except LookupError:
        sw={"the","a","an","and","or","but","is","are","was","were","to","of","in","on","for","with","this","that","it","i","my","me","be","as","at","by","from","very","have","has","had","not","so","too"}
        print("NLTK stopwords not available; using built-in English stopwords.")
    try:
        lem=WordNetLemmatizer()
        lem.lemmatize("test")
    except LookupError:
        class SimpleLemmatizer:
            def lemmatize(self,word):
                return word
        lem=SimpleLemmatizer()
        print("NLTK WordNet data not available; using basic token normalization.")
    data["clean_text"]=data.text.apply(lambda x:clean_text(x,sw,lem))
    data=data[data.clean_text.str.len()>0].reset_index(drop=True)

    Xtr,Xte,ytr,yte=train_test_split(data.clean_text,data.sentiment,test_size=.2,random_state=42,stratify=data.sentiment)
    vec=TfidfVectorizer(max_features=30000,ngram_range=(1,2),min_df=2,sublinear_tf=True)
    A=vec.fit_transform(Xtr)
    B=vec.transform(Xte)

    models={"Naive Bayes":MultinomialNB(),"Logistic Regression":LogisticRegression(max_iter=1000,random_state=42)}
    rows=[]

    for name,model in models.items():
        model.fit(A,ytr)
        pred=model.predict(B)
        p,r,f,_=precision_recall_fscore_support(yte,pred,average="weighted",zero_division=0)
        acc=accuracy_score(yte,pred)
        rows.append([name,acc,p,r,f])
        print("\n"+"="*60)
        print(name)
        print("="*60)
        print(classification_report(yte,pred,zero_division=0))

        labels=["Negative","Neutral","Positive"]
        cm=confusion_matrix(yte,pred,labels=labels)
        ConfusionMatrixDisplay(cm,display_labels=labels).plot(cmap="Blues",values_format="d")
        plt.title("Confusion Matrix - "+name)
        plt.tight_layout()
        plt.savefig(OUT/(name.lower().replace(" ","_")+"_confusion_matrix.png"),dpi=200)
        plt.close()

    results=pd.DataFrame(rows,columns=["Model","Accuracy","Precision","Recall","F1-Score"])
    print("\nModel comparison:")
    print(results.to_string(index=False))
    results.to_csv(OUT/"model_comparison.csv",index=False)

    results.set_index("Model").plot(kind="bar",figsize=(9,5),ylim=(0,1))
    plt.ylabel("Score")
    plt.title("Sentiment Model Performance")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUT/"model_comparison.png",dpi=200)
    plt.close()

    for s in order:
        text=" ".join(data.loc[data.sentiment==s,"clean_text"])
        if text.strip():
            wc=WordCloud(width=900,height=500,background_color="white",max_words=120,random_state=42).generate(text)
            plt.figure(figsize=(10,5))
            plt.imshow(wc,interpolation="bilinear")
            plt.axis("off")
            plt.title(s+" WordCloud")
            plt.tight_layout()
            plt.savefig(OUT/(s.lower()+"_wordcloud.png"),dpi=200)
            plt.close()

    best=models[results.sort_values("F1-Score",ascending=False).iloc[0].Model]
    best.fit(A,ytr)
    pred=best.predict(B)
    errors=pd.DataFrame({"Review":data.loc[yte.index,"text"].values,"Actual":yte.values,"Predicted":pred})
    errors=errors[errors.Actual!=errors.Predicted].head(5)
    errors.to_csv(OUT/"error_analysis.csv",index=False)
    print("\nFive misclassified reviews:")
    print(errors.to_string(index=False))
    results.sort_values("F1-Score",ascending=False).to_csv(OUT/"model_comparison.csv",index=False)

if __name__=="__main__":
    main()
