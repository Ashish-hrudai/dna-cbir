"""
CBIR Models Module
Implements Model I (Instance-based) and Model II (Class-based) CBIR systems
"""
import numpy as np
import cv2
from typing import List, Tuple, Dict
import pickle
import os
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications import ResNet50, VGG16, VGG19, InceptionV3
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
import tensorflow as tf


class InstanceBasedCBIR:
    """
    Model I: Instance-based CBIR using codon features (Algorithm 4)
    """
    
    def __init__(self, encoder, similarity_matcher, use_multi_scale: bool = False):
        """
        Initialize Instance-based CBIR
        
        Args:
            encoder: DNAEncoder instance
            similarity_matcher: SimilarityMatcher instance
            use_multi_scale: Whether to use multi-scale features (Idea 2)
        """
        self.encoder = encoder
        self.similarity_matcher = similarity_matcher
        self.use_multi_scale = use_multi_scale
        self.feature_space = {}
        self.image_paths = []
        
    def build_feature_space(self, images: List[np.ndarray], 
                           image_paths: List[str]) -> None:
        """
        Build feature space from dataset images
        
        Args:
            images: List of images
            image_paths: List of corresponding image paths
        """
        self.image_paths = image_paths
        
        for idx, image in enumerate(images):
            if self.use_multi_scale:
                features = self.encoder.extract_multi_scale_features(image)
            else:
                features = self.encoder.extract_features(image)
            
            self.feature_space[idx] = features
    
    def retrieve_similar_images(self, query_image: np.ndarray, 
                               k: int = 10,
                               metric: str = 'weighted') -> List[Tuple[int, float]]:
        """
        Retrieve top-k similar images for query
        
        Args:
            query_image: Query image
            k: Number of images to retrieve
            metric: Similarity metric ('euclidean', 'cosine', 'chi_square', 'weighted')
            
        Returns:
            List of (image_index, similarity_score) tuples
        """
        # Extract query features
        if self.use_multi_scale:
            query_features = self.encoder.extract_multi_scale_features(query_image)
        else:
            query_features = self.encoder.extract_features(query_image)
        
        # Compute similarities
        similarities = []
        for idx, features in self.feature_space.items():
            if metric == 'euclidean':
                score = self.similarity_matcher.euclidean_distance(query_features, features)
            elif metric == 'cosine':
                score = self.similarity_matcher.cosine_similarity(query_features, features)
            elif metric == 'chi_square':
                score = self.similarity_matcher.chi_square_distance(query_features, features)
            elif metric == 'weighted':
                score = self.similarity_matcher.compute_weighted_similarity(query_features, features)
            else:
                score = self.similarity_matcher.euclidean_distance(query_features, features)
            
            similarities.append((idx, score))
        
        # Sort by similarity (lower score = more similar)
        similarities.sort(key=lambda x: x[1])
        
        # Return top-k
        return similarities[:k]
    
    def save_feature_space(self, filepath: str) -> None:
        """Save feature space to disk"""
        data = {
            'feature_space': self.feature_space,
            'image_paths': self.image_paths
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    def load_feature_space(self, filepath: str) -> None:
        """Load feature space from disk"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.feature_space = data['feature_space']
        self.image_paths = data['image_paths']


class ClassBasedCBIR:
    """
    Model II: Class-based CBIR using CNN and amplified DNA planes (Algorithm 5)
    """
    
    def __init__(self, translator, encoder, 
                 architecture: str = 'resnet50',
                 num_classes: int = 10):
        """
        Initialize Class-based CBIR
        
        Args:
            translator: DNATranslator instance
            encoder: DNAEncoder instance
            architecture: CNN architecture ('resnet50', 'vgg16', 'vgg19', 'inception_v3')
            num_classes: Number of classes in dataset
        """
        self.translator = translator
        self.encoder = encoder
        self.architecture = architecture
        self.num_classes = num_classes
        self.model = None
        self.label_encoder = LabelEncoder()
        
    def prepare_dna_data(self, images: List[np.ndarray]) -> np.ndarray:
        """
        Prepare DNA planes for CNN training
        
        Args:
            images: List of input images
            
        Returns:
            Array of amplified DNA planes
        """
        dna_data = []
        for image in images:
            amplified_planes = self.translator.prepare_dna_planes_for_cnn(image, self.encoder)
            dna_data.append(amplified_planes)
        
        return np.array(dna_data)
    
    def build_model(self, input_shape: Tuple[int, int, int],
                   use_pretrained: bool = True) -> Model:
        """
        Build CNN model for classification
        
        Args:
            input_shape: Input shape (H, W, C)
            use_pretrained: Whether to use ImageNet pretrained weights
            
        Returns:
            Compiled Keras model
        """
        weights = 'imagenet' if use_pretrained else None
        
        # Select base architecture
        if self.architecture == 'resnet50':
            base_model = ResNet50(weights=weights, include_top=False, 
                                 input_shape=input_shape)
        elif self.architecture == 'vgg16':
            base_model = VGG16(weights=weights, include_top=False,
                              input_shape=input_shape)
        elif self.architecture == 'vgg19':
            base_model = VGG19(weights=weights, include_top=False,
                              input_shape=input_shape)
        elif self.architecture == 'inception_v3':
            base_model = InceptionV3(weights=weights, include_top=False,
                                    input_shape=input_shape)
        else:
            raise ValueError(f"Unknown architecture: {self.architecture}")
        
        # Freeze base model layers initially
        for layer in base_model.layers:
            layer.trainable = False
        
        # Add custom classification head
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)
        predictions = Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(inputs=base_model.input, outputs=predictions)
        
        return model
    
    def train_model(self, train_images: np.ndarray, train_labels: np.ndarray,
                   val_images: np.ndarray = None, val_labels: np.ndarray = None,
                   epochs: int = 50, batch_size: int = 32,
                   learning_rate: float = 0.001) -> Dict:
        """
        Train the CNN model
        
        Args:
            train_images: Training images (DNA planes)
            train_labels: Training labels
            val_images: Validation images
            val_labels: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            
        Returns:
            Training history
        """
        # Encode labels
        train_labels_encoded = self.label_encoder.fit_transform(train_labels)
        
        # Build model if not already built
        if self.model is None:
            input_shape = train_images.shape[1:]
            self.model = self.build_model(input_shape)
        
        # Compile model
        optimizer = RMSprop(learning_rate=learning_rate, momentum=0.9)
        self.model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Data augmentation
        datagen = ImageDataGenerator(
            rotation_range=360,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            vertical_flip=True,
            zoom_range=0.2
        )
        
        # Prepare validation data
        validation_data = None
        if val_images is not None and val_labels is not None:
            val_labels_encoded = self.label_encoder.transform(val_labels)
            validation_data = (val_images, val_labels_encoded)
        
        # Train
        history = self.model.fit(
            datagen.flow(train_images, train_labels_encoded, batch_size=batch_size),
            epochs=epochs,
            validation_data=validation_data,
            verbose=1
        )
        
        return history.history
    
    def predict_class(self, query_image: np.ndarray) -> Tuple[int, float]:
        """
        Predict class for query image
        
        Args:
            query_image: Query image (DNA plane)
            
        Returns:
            (predicted_class, confidence)
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Validate input shape matches model; attempt to auto-resize if needed
        expected_shape = tuple(self.model.input_shape[1:])
        if query_image.shape != expected_shape:
            try:
                expected_h, expected_w = expected_shape[0], expected_shape[1]
                # cv2.resize expects (width, height)
                resized = cv2.resize(query_image, (expected_w, expected_h), interpolation=cv2.INTER_LINEAR)
                query_image = resized
                # Note: resizing DNA planes is a pragmatic fallback; for best results
                # resize the original RGB image before DNA encoding and retrain the model.
            except Exception as e:
                raise ValueError(f"Model expects input shape {expected_shape}, found {query_image.shape}. "
                                 f"Automatic resizing failed: {e}")

        # Add batch dimension
        query_batch = np.expand_dims(query_image, axis=0)
        
        # Predict
        predictions = self.model.predict(query_batch, verbose=0)
        class_idx = np.argmax(predictions[0])
        confidence = predictions[0][class_idx]
        
        # Decode label
        predicted_class = self.label_encoder.inverse_transform([class_idx])[0]
        
        return predicted_class, confidence
    
    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract deep features from penultimate layer
        
        Args:
            image: Input image (DNA plane)
            
        Returns:
            Feature vector
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Create feature extraction model
        feature_model = Model(
            inputs=self.model.input,
            outputs=self.model.layers[-3].output  # Before final dense layer
        )
        
        # Validate input shape matches model; attempt to auto-resize if needed
        expected_shape = tuple(self.model.input_shape[1:])
        if image.shape != expected_shape:
            try:
                expected_h, expected_w = expected_shape[0], expected_shape[1]
                image = cv2.resize(image, (expected_w, expected_h), interpolation=cv2.INTER_LINEAR)
            except Exception as e:
                raise ValueError(f"Model expects input shape {expected_shape}, found {image.shape}. "
                                 f"Automatic resizing failed: {e}")

        # Extract features
        query_batch = np.expand_dims(image, axis=0)
        features = feature_model.predict(query_batch, verbose=0)
        
        return features.flatten()
    
    def retrieve_similar_images(self, query_image: np.ndarray,
                               class_images: List[np.ndarray],
                               k: int = 10) -> List[Tuple[int, float]]:
        """
        Retrieve similar images within predicted class
        
        Args:
            query_image: Query image (DNA plane)
            class_images: Images from predicted class (DNA planes)
            k: Number of images to retrieve
            
        Returns:
            List of (image_index, similarity_score) tuples
        """
        # Extract query features
        query_features = self.extract_features(query_image)
        
        # Compute similarities
        similarities = []
        for idx, image in enumerate(class_images):
            features = self.extract_features(image)
            score = np.linalg.norm(query_features - features)
            similarities.append((idx, score))
        
        # Sort and return top-k
        similarities.sort(key=lambda x: x[1])
        return similarities[:k]
    
    def save_model(self, filepath: str) -> None:
        """Save model and label encoder"""
        self.model.save(filepath)
        with open(filepath + '_labels.pkl', 'wb') as f:
            pickle.dump(self.label_encoder, f)
    
    def load_model(self, filepath: str) -> None:
        """Load model and label encoder"""
        self.model = tf.keras.models.load_model(filepath)
        with open(filepath + '_labels.pkl', 'rb') as f:
            self.label_encoder = pickle.load(f)


def compute_retrieval_metrics(retrieved_indices: List[int],
                              relevant_indices: List[int],
                              k: int) -> Dict[str, float]:
    """
    Compute precision, recall, and F-score for retrieval
    
    Args:
        retrieved_indices: Indices of retrieved images
        relevant_indices: Indices of relevant images
        k: Number of retrieved images
        
    Returns:
        Dictionary with metrics
    """
    retrieved_set = set(retrieved_indices[:k])
    relevant_set = set(relevant_indices)
    
    true_positives = len(retrieved_set.intersection(relevant_set))
    
    precision = true_positives / k if k > 0 else 0
    recall = true_positives / len(relevant_set) if len(relevant_set) > 0 else 0
    f_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision * 100,
        'recall': recall * 100,
        'f_score': f_score * 100
    }