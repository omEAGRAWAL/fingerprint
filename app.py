from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
model = load_model('fingerprint_matching_model.h5')

def preprocess(img, target_size=(90, 90)):
    img = image.load_img(img, target_size=target_size, color_mode='grayscale')
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        img1 = request.files['image1']
        img2 = request.files['image2']
        
        if not img1 or not img2:
            return "Please upload both images."

        img1_path = os.path.join('static', 'img1.png')
        img2_path = os.path.join('static', 'img2.png')

        img1.save(img1_path)
        img2.save(img2_path)

        pre1 = preprocess(img1_path)
        pre2 = preprocess(img2_path)
        
        similarity = model.predict([pre1, pre2])[0][0]
        similarity = round(float(similarity), 4)

        return render_template('index.html', score=similarity)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
