from flask import Flask, render_template, request
import MalariaPrediction
import os
from werkzeug import secure_filename

app = Flask(__name__)

@app.route('/')
def main_page():
    return render_template('index.html')

@app.route('/detector')
def detector():
    return render_template('detector.html')
@app.route('/uploadfile',methods=['POST'])
def uploadFile():
  if request.method == 'POST':
    file = request.files['pic']
    filename = file.filename
    file.save(os.path.join("UPLOADS/",secure_filename(filename)))

    pred = MalariaPrediction.processImage("UPLOADS/",filename)
    if pred > 0:
        result = "Your cell is Parasitized"
    else:
        result = "Your cell is Uninfected"
    #return result
    return render_template('results.html', result=result, pred = pred, filename="UPLOADS\\C33P1thinF_IMG_20150619_114756a_cell_180.png")
