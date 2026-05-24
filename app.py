"""
Flask Web Application for DNA-CBIR System
"""
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
from dna_encoding import DNAEncoder, DNATranslator, SimilarityMatcher
from cbir_models import InstanceBasedCBIR, ClassBasedCBIR, compute_retrieval_metrics
import json
from pathlib import Path
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
DATABASE_FOLDER = 'database'
MODELS_FOLDER = 'models'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DATABASE_FOLDER'] = DATABASE_FOLDER
app.config['MODELS_FOLDER'] = MODELS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create directories
for folder in [UPLOAD_FOLDER, DATABASE_FOLDER, MODELS_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Initialize DNA-CBIR components
encoder = DNAEncoder(num_msb_levels=3, num_bins=10)
translator = DNATranslator(amplification_weight=85)
similarity_matcher = SimilarityMatcher()

# Initialize CBIR models (will be configured on first use)
instance_cbir = None
class_cbir = None

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_image(filepath):
    """Load and preprocess image"""
    image = cv2.imread(filepath)
    if image is None:
        raise ValueError(f"Could not load image: {filepath}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def resize_image(image, max_size=512):
    """Resize image maintaining aspect ratio"""
    h, w = image.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        image = cv2.resize(image, (new_w, new_h))
    return image

def image_to_base64(image_path):
    """Convert image to base64 string"""
    with open(image_path, 'rb') as f:
        img_data = f.read()
    return base64.b64encode(img_data).decode('utf-8')

def numpy_to_base64(image_array):
    """Convert numpy array to base64"""
    img = Image.fromarray(image_array.astype('uint8'))
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Upload query image"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        return jsonify({
            'success': True,
            'filename': filename,
            'filepath': filepath
        })
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/api/search', methods=['POST'])
def search_images():
    """Perform image retrieval"""
    data = request.json
    query_file = data.get('query_file')
    model_type = data.get('model_type', 'instance')  # 'instance' or 'class'
    k = int(data.get('k', 10))
    use_multi_scale = data.get('use_multi_scale', False)
    similarity_metric = data.get('similarity_metric', 'weighted')
    
    if not query_file:
        return jsonify({'error': 'No query file specified'}), 400
    
    query_path = os.path.join(app.config['UPLOAD_FOLDER'], query_file)
    if not os.path.exists(query_path):
        return jsonify({'error': 'Query file not found'}), 404
    
    try:
        # Load query image
        query_image = load_image(query_path)
        query_image = resize_image(query_image)
        
        # Get database images
        database_images = []
        database_paths = []
        for filename in os.listdir(app.config['DATABASE_FOLDER']):
            if allowed_file(filename):
                img_path = os.path.join(app.config['DATABASE_FOLDER'], filename)
                img = load_image(img_path)
                img = resize_image(img)
                database_images.append(img)
                database_paths.append(filename)
        
        if len(database_images) == 0:
            return jsonify({'error': 'No images in database'}), 400
        
        # Perform retrieval based on model type
        if model_type == 'instance':
            # Instance-based CBIR (Model I)
            global instance_cbir
            instance_cbir = InstanceBasedCBIR(encoder, similarity_matcher, 
                                             use_multi_scale=use_multi_scale)
            instance_cbir.build_feature_space(database_images, database_paths)
            
            results = instance_cbir.retrieve_similar_images(
                query_image, k=k, metric=similarity_metric
            )
            
        else:
            # Class-based CBIR (Model II)
            return jsonify({'error': 'Class-based retrieval requires pre-trained model'}), 400
        
        # Prepare response
        retrieved_images = []
        for idx, score in results:
            img_filename = database_paths[idx]
            img_path = os.path.join(app.config['DATABASE_FOLDER'], img_filename)
            
            retrieved_images.append({
                'filename': img_filename,
                'score': float(score),
                'image': image_to_base64(img_path)
            })
        
        return jsonify({
            'success': True,
            'query_image': image_to_base64(query_path),
            'results': retrieved_images,
            'total_retrieved': len(retrieved_images)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze', methods=['POST'])
def analyze_image():
    """Analyze image and extract DNA features"""
    data = request.json
    filename = data.get('filename')
    
    if not filename:
        return jsonify({'error': 'No filename specified'}), 400
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404
    
    try:
        # Load image
        image = load_image(filepath)
        image = resize_image(image)
        
        # Extract DNA features
        dna_planes = encoder.encode_image_to_dna_planes(image)
        features = encoder.extract_features(image)
        multi_scale_features = encoder.extract_multi_scale_features(image)
        
        # Get DNA statistics
        dna_stats = {
            'num_channels': len(dna_planes),
            'dna_length_per_channel': len(dna_planes[0]) if dna_planes else 0,
            'feature_vector_length': len(features),
            'multi_scale_feature_length': len(multi_scale_features)
        }
        
        # Check GC content
        all_nucleotides = ''.join(dna_planes)
        gc_count = all_nucleotides.count('G') + all_nucleotides.count('C')
        at_count = all_nucleotides.count('A') + all_nucleotides.count('T')
        gc_percentage = (gc_count / (gc_count + at_count) * 100) if (gc_count + at_count) > 0 else 0
        
        dna_stats['gc_content_percentage'] = round(gc_percentage, 2)
        dna_stats['gc_constraint_satisfied'] = 40 <= gc_percentage <= 60
        
        return jsonify({
            'success': True,
            'dna_stats': dna_stats,
            'sample_dna_sequence': dna_planes[0][:100] if dna_planes else '',
            'image_shape': image.shape
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/database/upload', methods=['POST'])
def upload_database_images():
    """Upload multiple images to database"""
    if 'files[]' not in request.files:
        return jsonify({'error': 'No files provided'}), 400
    
    files = request.files.getlist('files[]')
    uploaded = []
    errors = []
    
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['DATABASE_FOLDER'], filename)
            file.save(filepath)
            uploaded.append(filename)
        else:
            errors.append(f"Invalid file: {file.filename}")
    
    return jsonify({
        'success': True,
        'uploaded': uploaded,
        'errors': errors,
        'total_uploaded': len(uploaded)
    })

@app.route('/api/database/list', methods=['GET'])
def list_database_images():
    """List all images in database"""
    images = []
    for filename in os.listdir(app.config['DATABASE_FOLDER']):
        if allowed_file(filename):
            images.append({
                'filename': filename,
                'path': os.path.join(app.config['DATABASE_FOLDER'], filename)
            })
    
    return jsonify({
        'success': True,
        'images': images,
        'total': len(images)
    })

@app.route('/api/compare-metrics', methods=['POST'])
def compare_similarity_metrics():
    """Compare different similarity metrics (IDEA 3)"""
    data = request.json
    query_file = data.get('query_file')
    k = int(data.get('k', 10))
    
    if not query_file:
        return jsonify({'error': 'No query file specified'}), 400
    
    query_path = os.path.join(app.config['UPLOAD_FOLDER'], query_file)
    
    try:
        # Load query image
        query_image = load_image(query_path)
        query_image = resize_image(query_image)
        
        # Load database
        database_images = []
        database_paths = []
        for filename in os.listdir(app.config['DATABASE_FOLDER']):
            if allowed_file(filename):
                img_path = os.path.join(app.config['DATABASE_FOLDER'], filename)
                img = load_image(img_path)
                img = resize_image(img)
                database_images.append(img)
                database_paths.append(filename)
        
        # Compare different metrics
        metrics = ['euclidean', 'cosine', 'chi_square', 'weighted']
        results = {}
        
        for metric in metrics:
            cbir = InstanceBasedCBIR(encoder, similarity_matcher)
            cbir.build_feature_space(database_images, database_paths)
            retrieved = cbir.retrieve_similar_images(query_image, k=k, metric=metric)
            
            results[metric] = [
                {
                    'filename': database_paths[idx],
                    'score': float(score)
                }
                for idx, score in retrieved
            ]
        
        return jsonify({
            'success': True,
            'metric_comparisons': results
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/database/<filename>')
def database_file(filename):
    """Serve database files"""
    return send_from_directory(app.config['DATABASE_FOLDER'], filename)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)