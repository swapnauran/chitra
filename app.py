import cv2
import numpy as np
import onnxruntime
from flask import Flask, render_template, request, jsonify, send_file
from io import BytesIO

app = Flask(__name__)

# Model names and paths (replace with your actual paths if different)
model_choices = {
    "modelx2": "C:/Users/MSI/Desktop/newupscale 2/models/models_modelx2.ort",
    "modelx2_25_JXL": "C:/Users/MSI/Desktop/newupscale 2/models/models_modelx2 25 JXL.ort",
    "modelx4": "C:/Users/MSI/Desktop/newupscale 2/models/models_modelx4.ort",
    "minecraft_modelx4": "C:/Users/MSI/Desktop/newupscale 2/models/models_minecraft_modelx4.ort",
}

def pre_process(img: np.array) -> np.array:
    # Transpose image from HWC to CHW format and expand dimension for 4D input
    img = np.transpose(img[:, :, 0:3], (2, 0, 1))
    img = np.expand_dims(img, axis=0).astype(np.float32)
    return img

def post_process(img: np.array) -> np.array:
    # 1, C, H, W -> C, H, W
    img = np.squeeze(img)
    # C, H, W -> H, W, C
    img = np.transpose(img, (1, 2, 0))[:, :, ::-1].astype(np.uint8)
    
    return img

def inference(model_path: str, img_array: np.array) -> np.array:
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    try:
        # Create the ONNX Runtime session
        ort_session = onnxruntime.InferenceSession(model_path, options)

        # Prepare input data
        ort_inputs = {ort_session.get_inputs()[0].name: img_array}

        # Run inference
        ort_outs = ort_session.run(None, ort_inputs)

        # Return the output
        return ort_outs[0]
    except Exception as e:
        print(f"Error during inference: {e}")
        return None  # Indicate an error occurred

def convert_pil_to_cv2(image):
    """Converts a PIL image to OpenCV format."""
    open_cv_image = np.array(image)
    open_cv_image = open_cv_image[:, :, ::-1].copy()  # Convert to BGR
    return open_cv_image

def upscale(img, model):
    """Upscales an image using the specified model."""
    # Retrieve the model path from the dictionary
    model_path = model_choices.get(model)

    # Handle missing model paths gracefully
    if not model_path:
        return None, "Invalid model selection."

    # Convert PIL image to OpenCV format
    img = convert_pil_to_cv2(img)

    # Handle grayscale images
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Handle images with an alpha channel (assuming no alpha channel is present here)
    # ... (code to handle the alpha channel if necessary)

    # Upscale the image
    try:
        upscaled_image = post_process(inference(model_path, pre_process(img)))
    except Exception as e:
        print(f"Error during upscaling: {e}")
        return None, "Upscaling failed."

    return upscaled_image, None  # Return the upscaled image and any error message

@app.route('/')
def index():
    return render_template('ImageUpscaleEnhance.html')

@app.route('/upscale', methods=['POST'])
def upscale_image():
    if request.method == 'POST':
        if 'image' not in request.files:
            return jsonify({'error': 'Missing image file.'})

        # Receive the image data from the request
        image_file = request.files['image']

        # Read the image using OpenCV
        img = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_UNCHANGED)

        # Get the model choice from the form
        selected_model = request.form.get('model')

        # Upscale the image using the specified model
        upscaled_img, error_message = upscale(img, selected_model)

        if upscaled_img is not None:
            # Convert the upscaled image to bytes
            _, img_bytes = cv2.imencode('.png', upscaled_img)

            # Create an in-memory buffer for the image data
            buffer = BytesIO(img_bytes)

            # Return the upscaled image
            return send_file(
                buffer,
                mimetype='image/png',
                as_attachment=True,
                download_name='upscaled_image.png'
            )
        else:
            # Return an error message
            return jsonify({'error': error_message})

if __name__ == '__main__':
    app.run(debug=True)
