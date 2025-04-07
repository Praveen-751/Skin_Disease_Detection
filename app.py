from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename
import cv2
import os
from ML_test import load_model, load_label_encoder, predict_image_path
from IP_Identification import classify_input_image_from_web,load_templates

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Load SVM model and label encoder globally
svm_model = load_model('svm_model.xml')
label_encoder = load_label_encoder('label_encoder.pkl')

# Optionally load or create templates
templates = load_templates('templates.npy') if os.path.exists('templates.npy') else {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        file = request.files['image']
        method = request.form['method']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            print("File saved to:", file_path)  # Debug output
            if method == 'SVM':
                # Assuming you have an appropriate function for SVM prediction
                prediction = predict_image_path(file_path)  # Correct function usage
            elif method == 'Template Matching':
                image = cv2.imread(file_path, cv2.IMREAD_COLOR)
                image = cv2.resize(image, (128, 128))
                prediction = classify_input_image_from_web(image, templates)

            return render_template('result.html', filename=filename, prediction=prediction, image_url=file_path)

    return render_template('upload.html')



@app.route('/<filename>')
def send_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
