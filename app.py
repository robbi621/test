from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import cv2
import numpy as np
import pandas as pd

app = Flask(__name__)

model = load_model('model_heroku.h5')


@app.route('/')
def index():
	return render_template("index.html", data="hey")


@app.route("/prediction", methods=["POST"])
def prediction():

	img = request.files['img']

	img.save("img.jpg")

	image = cv2.imread("img.jpg")

	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	image = cv2.resize(image, (100,100))

	image = np.reshape(image, (1,100,100,3))

	pred = model.predict(image)

	pred = np.argmax(pred)

	return render_template("prediction.html", data=pred)


if __name__ == "__main__":
	app.run(debug=True)