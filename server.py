from textblob import TextBlob
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle,re,nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

def predict_result(text,lemmatizer,tfidf,classifier):
    text=text.lower()
    text=re.sub('#','',text)
    text=re.sub('[^a-zA-Z ]','',text)
    words=nltk.word_tokenize(text)
    words=[lemmatizer.lemmatize(word) for word in words if word not in set(stopwords.words('english'))]
    sentence=' '.join(words)
    temp=[sentence]
    temp1=tfidf.transform(temp)
    result=classifier.predict(temp1)
    return result[0]

def get_label(result):
	if result==0:
		return "Not Disaster"
	elif result==1:
		return "Disaster"

app = Flask(__name__)
with open("disClassifier.pkl",'rb') as f:
	classifier=pickle.load(f)
with open("tfidf.pkl",'rb') as f:
	tfidf=pickle.load(f)
lemmatizer=WordNetLemmatizer()

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict',methods=['POST'])
def predict():
	text=request.form['tweet']
	result=predict_result(text,lemmatizer,tfidf,classifier)
	label=get_label(result)



	return render_template('index.html',prediction_text=label)
if __name__ == "__main__":
    app.run(debug=True)