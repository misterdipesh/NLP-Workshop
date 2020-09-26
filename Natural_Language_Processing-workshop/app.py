from flask import Flask, render_template, request, redirect
import pickle

app = Flask(__name__)

def load():
    with open('ham_spam.pkl', 'rb') as file:
        vectorizer, clf = pickle.load(file)
    return vectorizer, clf

def prediction_label(label):
	if label == 1:
		return 'Ham'
	else:
		return 'Spam'


@app.route('/', methods = ['GET', 'POST'])
def home():
	return render_template('home.html')

@app.route('/predict', methods = ['GET', 'POST'])
def prediction():
	if request.method == 'POST':
		text = request.form['message']
		data = [text]
		vectorizer, classifier = load()
		vec = vectorizer.transform(data)
		predic = classifier.predict(vec)
		pred_label = prediction_label(predic[0])
		return render_template('prediction.html', predicted_value = pred_label)
	else:
		return render_template('predict.html')


if __name__ == '__main__':
	app.run(debug = True)