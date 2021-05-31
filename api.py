import os
from flask import Flask 
from flask import request
from flask import render_template 
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pytesseract
#change this path if you install pytesseract in another folder:
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

app = Flask(__name__) 
UPLOAD_FOLDER = "E:\\SY-ME-C\\SDP\\Textract\\static\\saved_images"

#Model 
def model_predict(image_path):
    im = np.array(Image.open(image_path))

    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    plt.figure(figsize=(20,20))
    plt.title('GRAYSCALE IMAGE')
    plt.imshow(im, cmap='gray'); plt.xticks([]); plt.yticks([])


    _, im = cv2.threshold(im, 240, 255, 1) 
    plt.figure(figsize=(20,20))
    plt.title('IMMAGINE BINARIA')
    plt.imshow(im, cmap='gray'); plt.xticks([]); plt.yticks([])


    def preprocess_final(im):
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        _, im = cv2.threshold(im, 240, 255, 1)
        return im

    img=np.array(Image.open(image_path))
    im=preprocess_final(img)
    text = pytesseract.image_to_string(im, lang='eng')
    return text


#Flask
@app.route("/", methods=["GET", "POST"])
def upload_predict():
    if request.method == "POST":
        image_file = request.files["image"]
        if image_file:
            image_location = os.path.join(
                UPLOAD_FOLDER,
                image_file.filename
            )
            image_file.save(image_location)

            prediction = model_predict(image_location)
            return render_template("index.html",scrollToAnchor='extract', prediction = prediction) 
    return render_template("index.html") 

if __name__ == "__main__": 
    app.run(port=12000, debug=True) 
