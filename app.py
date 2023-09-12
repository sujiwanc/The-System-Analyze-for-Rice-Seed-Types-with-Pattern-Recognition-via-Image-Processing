import os
from flask import Flask, render_template, request, jsonify, redirect, url_for
import tensorflow.compat.v1 as tf
import cv2
import numpy as np

app = Flask(__name__)

@app.route('/')
def welcome_page():
    return render_template('welcome_page.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/go_to_index')
def go_to_index():
    return redirect(url_for('index'))
    
@app.route('/contact')
def contact():
    return render_template('contact.html')

# Load the model and labels outside of the route for better performance
model_path = "retrained_graph.pb"
label_path = "retrained_labels.txt"
graph = None
label_lines = None

# สร้างรายการคำแปลภาษาไทย
thai_labels = {
    "RD61": "ข้าวเจ้ากข61",
    "Phitsanulok2": "ข้าวพิษณุโลก2",
    "RD80": "ข้าวเจ้ากข80",
    "RD41": "ข้าวเจ้ากข41"
}

def load_model():
    global graph, label_lines
    with tf.io.gfile.GFile(model_path, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    with open(label_path, 'r') as f:
        label_lines = [line.strip() for line in f.readlines()]

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')

load_model()

def classify_image(image_data):
    with tf.compat.v1.Session(graph=graph) as sess:
        # Feed the image_data as input to the graph and get predictions
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})

        # Sort to show labels of first prediction in order of confidence
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

        results = []
        for node_id in top_k:
            label = label_lines[node_id]
            score = predictions[0][node_id]
            
            # แปลคำภาษาอังกฤษเป็นภาษาไทย
            translated_label = thai_labels.get(label, label)  # หากไม่พบให้ใช้ค่าเดิม
            
            result = {'label': translated_label, 'score': float(score)}
            results.append(result)

        return results

@app.route('/', methods=['GET', 'POST'])
def classify():
    if request.method == 'POST':
        if 'image' in request.files:
            # Get the image file from the form data
            image_file = request.files['image']
            if image_file:
                # Read the image data
                image_data = image_file.read()

                # Perform image classification
                results = classify_image(image_data)

                # Convert results to a human-readable format with percentages
                predictions = [f"{result['label']} ({result['score']*100:.2f}%)"
                               for result in results]

                # Save the captured image to a temporary folder (optional)
                image_path = os.path.join("static", "uploads", "captured_image.jpg")
                with open(image_path, 'wb') as f:
                    f.write(image_data)

                return render_template('index.html', predictions=predictions, captured_image_path=image_path)

    return render_template('index.html', predictions=None, captured_image_path=None)

# ... (โค้ดอื่น ๆ ใน Flask) ...

if __name__ == '__main__':
    app.run(port=5001)
