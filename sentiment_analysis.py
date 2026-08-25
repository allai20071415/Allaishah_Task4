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
    files=list(DATA.glob("*.csv"))
    if not files:
        raise FileNotFoundError("No CSV found in data/. Download the Kaggle Play Store User Reviews CSV and place it there.")
    for f in files:
        d=pd.read_csv(f,nrows=3)
        cols={str(c).strip().lower() for c in d.columns}
        if "sentiment" in cols and ({"translated_review","review","text"} & cols):
            return f
    raise ValueError("No compatible CSV found. Expected Sentiment and Translated_Review, Review, or Text.")

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
    sns.countplot(data=data,x="sentiment",order=order)
    plt.title("Play Store Review Sentiment Distribution")
    plt.tight_layout()
    plt.savefig(OUT/"sentiment_distribution.png",dpi=200)
    plt.close()

    sw=set(stopwords.words("english"))
    lem=WordNetLemmatizer()
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
