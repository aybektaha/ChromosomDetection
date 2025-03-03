# app.py
from flask import Flask, render_template, request, send_file, send_from_directory
from PIL import Image
import os
from werkzeug.utils import secure_filename
from utils import process_chromosome_image

app = Flask(__name__)

# Configure upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'output'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'txt'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER

# Create necessary directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'image' not in request.files or 'text' not in request.files:
        return 'No file uploaded', 400
    
    image_file = request.files['image']
    text_file = request.files['text']
    
    if image_file.filename == '' or text_file.filename == '':
        return 'No selected file', 400
    
    if (image_file and allowed_file(image_file.filename) and 
        text_file and allowed_file(text_file.filename)):
        
        # Save uploaded files
        image_filename = secure_filename(image_file.filename)
        text_filename = secure_filename(text_file.filename)
        
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
        text_path = os.path.join(app.config['UPLOAD_FOLDER'], text_filename)
        
        image_file.save(image_path)
        text_file.save(text_path)
        model_path = "model_checkpoints/mobilenetv2_best.pth"


        chromosome_list = [str(i) for i in range(1, 23)]
        chromosome_list.extend(["x", "y"])
        karyo_table = [{"id": i, "images": []} for i in chromosome_list]
        # Process the files 
        karyotype_data, main_result = process_chromosome_image(image_path=image_path, txt_path=text_path, model_path=model_path)
        for chr in karyo_table:
            chr["images"] = karyotype_data.get(chr["id"], [])
            for i in range(len(chr["images"])):
                chr["images"][i] = chr["images"][i].split("output/")[1]
        
        main_result = main_result.split("output/")[1]

        return render_template('results.html', 
                             main_result=main_result, 
                             karyotype_data=karyo_table)
    
    return 'Invalid file type', 400

@app.route('/results/<filename>')
def results(filename):
    print(filename)
    # return send_file(filename)
    return send_from_directory(app.config["RESULTS_FOLDER"], filename)

if __name__ == '__main__':
    app.run(debug=True, port=80, host="0.0.0.0")