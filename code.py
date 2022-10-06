from joblib import load
from flask import Flask, render_template, request
from flask import url_for
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import os
from skimage.feature import hog
from skimage.transform import rescale, resize
app = Flask(__name__)

# Preprocessor


class Preprocessor():
    def __init__(self, path: str):
        self.from_path = path

    def transform(self):
        image = io.imread(self.from_path, as_gray=True)
        scaled = rescale(image, 1/3)
        resized_img = resize(scaled, (228, 228))
        tr_image = hog(resized_img)
        return tr_image


# Loading Model

model = load('NN2')
diseases = ['Bacterial Leaf Blight', 'Brown Spot', 'Leaf Smut']
@app.route('/remedy')
def remedy():
    return render_template('remedy.html')


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        file = request.files.get('image')
        file.save(os.path.join('static/uploads', file.filename))
        p = Preprocessor(os.path.join('static/uploads', file.filename))
        image = np.array([p.transform()])
        prediction = model.predict(image)
        probability = round(model.predict_proba(image)[0,prediction[0]]*100, 2)
        disease = diseases[prediction[0]]
        return render_template('home.html', prediction=disease, probability=probability, image=file.filename)
    return render_template('home.html', prerdiction=None)


if __name__ == '__main__':
    app.run(debug=True)
