from flask import Flask, render_template, request, jsonify

from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
class_names = ['Potato__Early_blight', 'Potato_Late_blight', 'Potato__healthy','Insect','Virus']


# Define the upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Load your pre-trained Keras model
model = load_model(r'model1 (1).h5')

# Function to check if a file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to preprocess the uploaded image
def preprocess_image(file_path):
    img = image.load_img(file_path, target_size=(256, 256))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = img / 255.0  # Normalize pixel values (if needed)
    return img
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has a file part
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})

        uploaded_file = request.files['file']

        # If user does not select file, browser also submits an empty part
        if uploaded_file.filename == '':
            return jsonify({'error': 'No selected file'})

        if uploaded_file and allowed_file(uploaded_file.filename):
            try:
                filename = secure_filename(uploaded_file.filename)
                file_path = os.path.join(UPLOAD_FOLDER, filename)
                uploaded_file.save(file_path)

                img = preprocess_image(file_path)
                predictions = model.predict(img)

                # Assuming your model outputs class probabilities, you can extract the class with the highest probability
                predicted_class = int(np.argmax(predictions))
                predicted_class_name = class_names[predicted_class]


                return jsonify({'predicted_class': predicted_class_name})

            except Exception as e:
                return jsonify({'error': str(e)})
        else:
            return jsonify({'error': 'Invalid file extension'})

if __name__ == '__main__':
    app.run(debug=True)
