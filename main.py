from flask import Flask, render_template, request, send_file, redirect, url_for, session
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2

app = Flask(__name__)
app.secret_key = 'secret_key'

UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
HIST_FOLDER = 'static/histograms'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(HIST_FOLDER, exist_ok=True)

def load_image(path_or_file):
    img = Image.open(path_or_file)
    if getattr(img, "is_animated", False):
        img.seek(0)
    return img.convert('RGB')

def save_image(img_array, path):
    img = Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))
    img.save(path)

def save_histogram(img_array, path):
    plt.figure(figsize=(6, 3))
    if img_array.ndim == 3:
        colors = ('r', 'g', 'b')
        for i, color in enumerate(colors):
            plt.hist(img_array[..., i].ravel(), bins=256, color=color, alpha=0.5)
    else:
        plt.hist(img_array.ravel(), bins=256, color='gray')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def to_grayscale(img_array, method='luminosity'):
    if method == 'average':
        return np.mean(img_array[..., :3], axis=-1)
    elif method == 'lightness':
        return (np.max(img_array[..., :3], axis=-1) + np.min(img_array[..., :3], axis=-1)) / 2
    else:
        return np.dot(img_array[..., :3], [0.299, 0.587, 0.114])

# def adjust_brightness_deprecated(img_array, value):
#     return np.clip(img_array + value, 0, 255)
def adjust_brightness(img_array, gamma_slider):

    min_slider, max_slider = -100, 100
    min_gamma, max_gamma = 0.2, 5.0


    normalized = (gamma_slider - min_slider) / (max_slider - min_slider)

    gamma_value = min_gamma + normalized * (max_gamma - min_gamma)

    img_float = img_array.astype(np.float32) / 255.0

    if gamma_value < 1e-6:
        gamma_value = 1e-6

    inv_gamma = 1.0 / gamma_value
    img_corrected = np.power(img_float, inv_gamma)


    img_corrected = (img_corrected * 255.0).clip(0, 255).astype(np.uint8)

    return img_corrected


def adjust_contrast(img_array, alpha):
    return np.clip(128 + alpha * (img_array - 128), 0, 255)

def negative(img_array):
    return 255 - img_array

def binarize(img_array, thresh=128,method='luminosity'):
    gray = to_grayscale(img_array,method)
    return np.where(gray > thresh, 255, 0).astype(np.uint8)

def apply_filter(img_array, kernel):
    from scipy.signal import convolve2d
    if img_array.ndim == 3:
        return np.stack([
            convolve2d(img_array[..., i], kernel, mode='same', boundary='symm')
            for i in range(3)
        ], axis=-1)
    else:
        return convolve2d(img_array, kernel, mode='same', boundary='symm')

def generate_kernel(op, size):
    if op == 'average':
        return np.ones((size, size), dtype=np.float32) / (size * size)
    elif op == 'gaussian':
        def gaussian_kernel(k, sigma=1):
            ax = np.linspace(-(k - 1) / 2., (k - 1) / 2., k)
            xx, yy = np.meshgrid(ax, ax)
            kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
            return kernel / np.sum(kernel)
        return gaussian_kernel(size)
    elif op == 'sharpen':
        kernel = -1 * np.ones((size, size), dtype=np.float32)
        center = size // 2
        kernel[center, center] = (size * size)
        return kernel
    return None

def apply_opencv_filter(img_array, op, val=0, grayscale_method='luminosity', matrix_size=3):
    img = np.clip(img_array, 0, 255).astype(np.uint8)
    if op == 'grayscale':
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    elif op == 'brightness':
        return cv2.convertScaleAbs(img, alpha=1, beta=val)
    elif op == 'contrast':
        return cv2.convertScaleAbs(img, alpha=val, beta=0)
    elif op == 'negative':
        return 255 - img
    elif op == 'binarize':
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, val, 255, cv2.THRESH_BINARY)
        return binary
    elif op == 'average':
        return cv2.blur(img, (matrix_size, matrix_size))
    elif op == 'gaussian':
        return cv2.GaussianBlur(img, (matrix_size, matrix_size), 0)
    elif op == 'sharpen':
        kernel = generate_kernel('sharpen', matrix_size)
        return cv2.filter2D(img, -1, kernel)
    return img

@app.route('/')
def index():
    original_filename = session.get('original_file')
    processed_filename = session.get('processed_file')
    opencv_filename = session.get('opencv_file')
    selected_operation = session.get('operation', 'grayscale')
    slider_value = session.get('value', 0)
    grayscale_method = session.get('grayscale_method', 'luminosity')
    matrix_size = session.get('matrix_size', 3)
    base_filename = os.path.splitext(original_filename)[0] if original_filename else ''
    return render_template('index.html',
        original_file=f"original_{original_filename}.png" if original_filename else None,
        result_file=processed_filename,
        opencv_file=opencv_filename,
        selected_operation=selected_operation,
        slider_value=slider_value,
        grayscale_method=grayscale_method,
        matrix_size=matrix_size,
        original_hist=f"histograms/hist_orig_{base_filename}.png" if original_filename else None,
        processed_hist=f"histograms/hist_proc_{base_filename}.png" if processed_filename else None,
        opencv_hist=f"histograms/hist_opencv_{base_filename}.png" if opencv_filename else None
    )

@app.route('/set_operation', methods=['POST'])
def set_operation():
    data = request.form
    session['operation'] = data.get('operation', 'grayscale')
    session['value'] = float(data.get('value', 0))
    session['grayscale_method'] = data.get('grayscale_method', 'luminosity')
    session['matrix_size'] = int(data.get('matrix_size', 3))
    return redirect(url_for('apply'))

@app.route('/apply', methods=['GET', 'POST'])
def apply():
    original_filename = session.get('original_file')
    if not original_filename:
        return redirect(url_for('index'))

    op = session.get('operation', 'grayscale')
    val = float(session.get('value', 0))
    grayscale_method = session.get('grayscale_method', 'luminosity')
    matrix_size = session.get('matrix_size', 3)

    base_filename = os.path.splitext(original_filename)[0]
    img_path = os.path.join(UPLOAD_FOLDER, f"original_{original_filename}.png")
    img = load_image(img_path)
    img_array = np.array(img).astype(np.float32)


    if op == 'grayscale':
        result = to_grayscale(img_array, method=grayscale_method)
    elif op == 'brightness':
        result = adjust_brightness(img_array, val)
    elif op == 'contrast':
        result = adjust_contrast(img_array, val)
    elif op == 'negative':
        result = negative(img_array)
    elif op == 'binarize':
        result = binarize(img_array, thresh=val,method=grayscale_method)
    elif op in ['average', 'gaussian', 'sharpen']:
        kernel = generate_kernel(op, matrix_size)
        result = apply_filter(img_array, kernel)
    else:
        result = img_array

    if result.ndim == 2:
        result = np.stack([result]*3, axis=-1)

    processed_filename = f"processed_{base_filename}.png"
    result_path = os.path.join(RESULT_FOLDER, processed_filename)
    save_image(result, result_path)
    save_histogram(result, os.path.join(HIST_FOLDER, f"hist_proc_{base_filename}.png"))
    session['processed_file'] = processed_filename


    opencv_result = apply_opencv_filter(img_array, op, val, grayscale_method, matrix_size)
    if opencv_result.ndim == 2:
        opencv_result = np.stack([opencv_result]*3, axis=-1)
    opencv_filename = f"opencv_{base_filename}.png"
    opencv_path = os.path.join(RESULT_FOLDER, opencv_filename)
    save_image(opencv_result, opencv_path)
    save_histogram(opencv_result, os.path.join(HIST_FOLDER, f"hist_opencv_{base_filename}.png"))
    session['opencv_file'] = opencv_filename

    return redirect(url_for('index'))

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.get('image')
    if file:
        filename = file.filename
        session['original_file'] = filename
        img = load_image(file)
        img_array = np.array(img).astype(np.float32)
        base_filename = os.path.splitext(filename)[0]
        converted_path = os.path.join(UPLOAD_FOLDER, f"original_{filename}.png")
        save_image(img_array, converted_path)
        save_histogram(img_array, os.path.join(HIST_FOLDER, f"hist_orig_{base_filename}.png"))
    return redirect(url_for('index'))

@app.route('/uploads/<filename>')
def uploaded_image(filename):
    return send_file(os.path.join(UPLOAD_FOLDER, filename))

@app.route('/results/<filename>')
def result_image(filename):
    return send_file(os.path.join(RESULT_FOLDER, filename))

@app.route('/download/<filename>')
def download_image(filename):
    return send_file(os.path.join(RESULT_FOLDER, filename), as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
