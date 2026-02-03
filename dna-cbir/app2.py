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

# Auto-load latest Model II if a saved model exists
try:
    if os.path.exists(app.config['MODELS_FOLDER']):
        model_files = sorted([
            f for f in os.listdir(app.config['MODELS_FOLDER'])
            if f.startswith('model2_') and f.endswith('.h5')
        ])
        if model_files:
            latest = model_files[-1]
            model_path = os.path.join(app.config['MODELS_FOLDER'], latest)
            architecture = latest[len('model2_'):-len('.h5')]
            try:
                class_cbir = ClassBasedCBIR(translator, encoder, architecture=architecture, num_classes=10)
                class_cbir.load_model(model_path)
                print(f"Auto-loaded Model II: {model_path}")
            except Exception as e:
                print(f"Failed to auto-load Model II {model_path}: {e}")
except Exception as e:
    print(f"Model autoload check failed: {e}")

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

def resize_for_model(image, model, default_size=224):
    """Resize image to match model input for DNA planes.

    The CNN expects inputs shaped (H, 3W, C) where 3W comes from concatenating
    three DNA planes horizontally. Given a model input shape (None, H, 3W, C),
    compute the original image width W = (3W) // 3 and resize the image to
    (W, H). If the model is not available or shape is unspecified, fall back
    to `resize_image` using `default_size`.
    """
    try:
        if model is not None and hasattr(model, 'input_shape') and model.input_shape:
            input_shape = model.input_shape
            # input_shape is (None, H, 3W, C)
            if input_shape[1] and input_shape[2]:
                expected_h = int(input_shape[1])
                expected_w = int(input_shape[2]) // 3
                return cv2.resize(image, (expected_w, expected_h))
    except Exception:
        pass
    return resize_image(image, max_size=default_size)

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
            
        elif model_type == 'class':
            # Class-based CBIR (Model II)
            global class_cbir
            
            if class_cbir is None:
                return jsonify({'error': 'Model II not trained. Please train the model first.'}), 400
            
            # Prepare query DNA plane
            query_resized = resize_image(query_image, max_size=224)
            query_dna = translator.prepare_dna_planes_for_cnn(query_resized, encoder)
            
            # Predict class
            predicted_class, confidence = class_cbir.predict_class(query_dna)
            
            # Filter database by predicted class
            class_images = []
            class_paths = []
            class_indices = []
            
            for idx, (img, path) in enumerate(zip(database_images, database_paths)):
                # Check if image belongs to predicted class (by filename prefix)
                if path.startswith(str(predicted_class)):
                    img_resized = resize_image(img, max_size=224)
                    dna_plane = translator.prepare_dna_planes_for_cnn(img_resized, encoder)
                    class_images.append(dna_plane)
                    class_paths.append(path)
                    class_indices.append(idx)
            
            if len(class_images) == 0:
                return jsonify({
                    'success': True,
                    'query_image': image_to_base64(query_path),
                    'predicted_class': str(predicted_class),
                    'confidence': float(confidence),
                    'results': [],
                    'total_retrieved': 0,
                    'message': f'No images found in predicted class: {predicted_class}'
                })
            
            # Retrieve within class
            results_within_class = class_cbir.retrieve_similar_images(
                query_dna, 
                class_images, 
                k=min(k, len(class_images))
            )
            
            # Map back to original indices
            results = [(class_indices[idx], score) for idx, score in results_within_class]
        
        else:
            return jsonify({'error': 'Invalid model type'}), 400
        
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
        
        response_data = {
            'success': True,
            'query_image': image_to_base64(query_path),
            'results': retrieved_images,
            'total_retrieved': len(retrieved_images),
            'model_type': model_type
        }
        
        # Add class info if using Model II
        if model_type == 'class' and class_cbir is not None:
            response_data['predicted_class'] = str(predicted_class)
            response_data['confidence'] = float(confidence)
        
        return jsonify(response_data)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
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

@app.route('/api/model2/prepare-dataset', methods=['POST'])
def prepare_model2_dataset():
    """Prepare dataset for Model II training"""
    try:
        from dataset_manager import DatasetManager
        
        data = request.json
        source_dir = data.get('source_dir', app.config['DATABASE_FOLDER'])
        target_size = tuple(data.get('target_size', [224, 224]))
        split_ratio = data.get('split_ratio', 0.8)
        
        manager = DatasetManager()
        
        # Organize dataset
        organized_path = manager.organize_dataset(source_dir)
        
        # Prepare for CBIR
        cbir_path = manager.prepare_for_cbir(
            organized_path,
            target_size=target_size,
            split_ratio=split_ratio
        )
        
        return jsonify({
            'success': True,
            'cbir_path': str(cbir_path),
            'train_dir': str(cbir_path / 'train'),
            'test_dir': str(cbir_path / 'test')
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/model2/train', methods=['POST'])
def train_model2():
    """Train Model II (Class-based CBIR)"""
    try:
        data = request.json
        train_dir = data.get('train_dir')
        architecture = data.get('architecture', 'resnet50')
        epochs = int(data.get('epochs', 50))
        batch_size = int(data.get('batch_size', 32))
        learning_rate = float(data.get('learning_rate', 0.001))
        
        if not train_dir or not os.path.exists(train_dir):
            return jsonify({'error': 'Invalid training directory'}), 400
        
        # Load training data
        from dataset_manager import DatasetManager
        manager = DatasetManager()
        class_mapping = manager.get_class_mapping(train_dir)
        num_classes = len(class_mapping)
        
        # Load images and labels
        train_images = []
        train_labels = []
        
        for class_name, class_idx in class_mapping.items():
            class_dir = os.path.join(train_dir, class_name)
            for filename in os.listdir(class_dir):
                if allowed_file(filename):
                    img_path = os.path.join(class_dir, filename)
                    img = load_image(img_path)
                    img = resize_image(img, max_size=224)
                    train_images.append(img)
                    train_labels.append(class_name)
        
        train_images = np.array(train_images)
        train_labels = np.array(train_labels)
        
        # Prepare DNA planes
        dna_images = []
        for img in train_images:
            dna_plane = translator.prepare_dna_planes_for_cnn(img, encoder)
            dna_images.append(dna_plane)
        dna_images = np.array(dna_images)
        
        # Initialize and train Model II
        global class_cbir
        from cbir_models import ClassBasedCBIR
        class_cbir = ClassBasedCBIR(
            translator, 
            encoder, 
            architecture=architecture,
            num_classes=num_classes
        )
        
        # Train
        history = class_cbir.train_model(
            dna_images, 
            train_labels,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate
        )
        
        # Save model
        model_path = os.path.join(app.config['MODELS_FOLDER'], f'model2_{architecture}.h5')
        class_cbir.save_model(model_path)
        
        return jsonify({
            'success': True,
            'model_path': model_path,
            'num_classes': num_classes,
            'history': {
                'loss': [float(x) for x in history.get('loss', [])],
                'accuracy': [float(x) for x in history.get('accuracy', [])]
            }
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/model2/load', methods=['POST'])
def load_model2():
    """Load pre-trained Model II"""
    try:
        data = request.json
        model_path = data.get('model_path')
        architecture = data.get('architecture', 'resnet50')
        
        if not model_path or not os.path.exists(model_path):
            return jsonify({'error': 'Model file not found'}), 404
        
        # Initialize Model II
        global class_cbir
        from cbir_models import ClassBasedCBIR
        class_cbir = ClassBasedCBIR(
            translator, 
            encoder, 
            architecture=architecture,
            num_classes=10  # Will be updated from model
        )
        
        # Load model
        class_cbir.load_model(model_path)
        
        return jsonify({
            'success': True,
            'model_loaded': True,
            'architecture': architecture
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/model2/predict', methods=['POST'])
def predict_model2():
    """Predict class using Model II"""
    global class_cbir
    
    if class_cbir is None:
        return jsonify({'error': 'Model II not trained or loaded'}), 400
    
    try:
        data = request.json
        filename = data.get('filename')
        
        if not filename:
            return jsonify({'error': 'No filename provided'}), 400
        
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 404
        
        # Load and prepare image
        image = load_image(filepath)
        # Resize to the model's expected input size (fallback to 224)
        image = resize_for_model(image, class_cbir.model, default_size=224)
        
        # Prepare DNA plane
        dna_plane = translator.prepare_dna_planes_for_cnn(image, encoder)
        
        # Predict
        predicted_class, confidence = class_cbir.predict_class(dna_plane)
        
        return jsonify({
            'success': True,
            'predicted_class': str(predicted_class),
            'confidence': float(confidence)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/model2/search', methods=['POST'])
def search_model2():
    """Search using Model II (class-based retrieval)"""
    global class_cbir
    
    if class_cbir is None:
        return jsonify({'error': 'Model II not trained or loaded'}), 400
    
    try:
        data = request.json
        query_file = data.get('query_file')
        k = int(data.get('k', 10))
        
        if not query_file:
            return jsonify({'error': 'No query file specified'}), 400
        
        query_path = os.path.join(app.config['UPLOAD_FOLDER'], query_file)
        if not os.path.exists(query_path):
            return jsonify({'error': 'Query file not found'}), 404
        
        # Load query image
        query_image = load_image(query_path)
        # Resize to the model's expected input size (fallback to 224)
        query_image = resize_for_model(query_image, class_cbir.model, default_size=224)
        
        # Prepare DNA plane
        query_dna = translator.prepare_dna_planes_for_cnn(query_image, encoder)
        
        # Predict class
        predicted_class, confidence = class_cbir.predict_class(query_dna)
        
        # Get images from predicted class
        database_images = []
        database_paths = []
        
        for filename in os.listdir(app.config['DATABASE_FOLDER']):
            if allowed_file(filename) and filename.startswith(str(predicted_class)):
                img_path = os.path.join(app.config['DATABASE_FOLDER'], filename)
                img = load_image(img_path)
                # Resize to the model's expected input size (fallback to 224)
                img = resize_for_model(img, class_cbir.model, default_size=224)
                
                # Prepare DNA plane
                dna_plane = translator.prepare_dna_planes_for_cnn(img, encoder)
                database_images.append(dna_plane)
                database_paths.append(filename)
        
        # Retrieve similar images within class
        if len(database_images) > 0:
            results = class_cbir.retrieve_similar_images(
                query_dna, 
                database_images, 
                k=min(k, len(database_images))
            )
            
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
                'predicted_class': str(predicted_class),
                'confidence': float(confidence),
                'query_image': image_to_base64(query_path),
                'results': retrieved_images,
                'total_retrieved': len(retrieved_images)
            })
        else:
            return jsonify({
                'success': True,
                'predicted_class': str(predicted_class),
                'confidence': float(confidence),
                'query_image': image_to_base64(query_path),
                'results': [],
                'total_retrieved': 0,
                'message': f'No images found in predicted class: {predicted_class}'
            })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/model2/status', methods=['GET'])
def model2_status():
    """Get Model II status"""
    global class_cbir
    
    available_models = [
        f for f in os.listdir(app.config['MODELS_FOLDER'])
        if f.startswith('model2_') and f.endswith('.h5')
    ] if os.path.exists(app.config['MODELS_FOLDER']) else []

    trained_flag = (class_cbir is not None) or (len(available_models) > 0)

    return jsonify({
        'success': True,
        'trained': trained_flag,
        'available_models': available_models
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
    app.run(debug=True, host='0.0.0.0', port=5000)