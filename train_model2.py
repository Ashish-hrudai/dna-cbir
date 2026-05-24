"""
Standalone Model II (Class-based CBIR) Training Script
Train deep learning model for class-based image retrieval
"""
import os
import sys
import argparse
import numpy as np
import cv2
from pathlib import Path

from dna_encoding import DNAEncoder, DNATranslator
from cbir_models import ClassBasedCBIR
from dataset_manager import DatasetManager


def load_dataset_from_directory(dataset_dir, target_size=(224, 224)):
    """Load images and labels from directory structure"""
    dataset_dir = Path(dataset_dir)
    
    images = []
    labels = []
    class_names = []
    
    # Get all class directories
    class_dirs = sorted([d for d in dataset_dir.iterdir() if d.is_dir()])
    
    for class_dir in class_dirs:
        class_name = class_dir.name
        class_names.append(class_name)
        
        # Load images from this class
        for img_file in class_dir.iterdir():
            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                img = cv2.imread(str(img_file))
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, target_size)
                    images.append(img)
                    labels.append(class_name)
    
    return np.array(images), np.array(labels), class_names


def train_model2(
    train_dir,
    architecture='resnet50',
    epochs=50,
    batch_size=32,
    learning_rate=0.001,
    output_dir='models'
):
    """
    Train Model II for class-based CBIR
    
    Args:
        train_dir: Directory containing training images (class-based folders)
        architecture: CNN architecture to use
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
        output_dir: Directory to save trained model
    """
    print("\n" + "="*70)
    print("MODEL II (CLASS-BASED CBIR) TRAINING")
    print("="*70)
    
    # Initialize components
    print("\n1. Initializing DNA encoding components...")
    encoder = DNAEncoder(num_msb_levels=3, num_bins=10)
    translator = DNATranslator(amplification_weight=85)
    
    # Load training data
    print("\n2. Loading training data...")
    train_images, train_labels, class_names = load_dataset_from_directory(train_dir)
    
    print(f"   ✓ Loaded {len(train_images)} images")
    print(f"   ✓ Number of classes: {len(class_names)}")
    print(f"   ✓ Classes: {', '.join(class_names)}")
    
    # Prepare DNA planes
    print("\n3. Preparing DNA planes for training images...")
    dna_images = []
    for idx, img in enumerate(train_images):
        if (idx + 1) % 10 == 0 or idx == 0:
            print(f"   Processing image {idx + 1}/{len(train_images)}...", end='\r')
        
        dna_plane = translator.prepare_dna_planes_for_cnn(img, encoder)
        dna_images.append(dna_plane)
    
    dna_images = np.array(dna_images)
    print(f"\n   ✓ Prepared {len(dna_images)} DNA planes")
    print(f"   ✓ DNA plane shape: {dna_images[0].shape}")
    
    # Initialize Model II
    print(f"\n4. Initializing Model II with {architecture}...")
    model = ClassBasedCBIR(
        translator,
        encoder,
        architecture=architecture,
        num_classes=len(class_names)
    )
    
    # Train model
    print(f"\n5. Training model...")
    print(f"   Architecture: {architecture}")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {learning_rate}")
    print()
    
    history = model.train_model(
        dna_images,
        train_labels,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate
    )
    
    # Save model
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f'model2_{architecture}.h5')
    
    print(f"\n6. Saving model to {model_path}...")
    model.save_model(model_path)
    
    # Print results
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"\nModel saved to: {model_path}")
    print(f"Architecture: {architecture}")
    print(f"Number of classes: {len(class_names)}")
    
    if 'accuracy' in history:
        final_acc = history['accuracy'][-1]
        print(f"Final training accuracy: {final_acc * 100:.2f}%")
    
    if 'loss' in history:
        final_loss = history['loss'][-1]
        print(f"Final training loss: {final_loss:.4f}")
    
    print("\n" + "="*70)
    print("You can now use Model II for class-based retrieval!")
    print("="*70)
    
    return model_path, history


def main():
    parser = argparse.ArgumentParser(description='Train Model II for DNA-CBIR')
    parser.add_argument('--train-dir', type=str, default='datasets/demo_dataset/train',
                       help='Directory containing training images')
    parser.add_argument('--architecture', type=str, default='resnet50',
                       choices=['resnet50', 'vgg16', 'vgg19', 'inception_v3'],
                       help='CNN architecture to use')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--output-dir', type=str, default='models',
                       help='Directory to save trained model')
    parser.add_argument('--prepare-dataset', action='store_true',
                       help='Prepare dataset before training')
    parser.add_argument('--dataset-source', type=str,
                       help='Source directory for dataset preparation')
    
    args = parser.parse_args()
    
    # Prepare dataset if requested
    if args.prepare_dataset:
        if not args.dataset_source:
            print("Error: --dataset-source required when using --prepare-dataset")
            sys.exit(1)
        
        print("Preparing dataset...")
        manager = DatasetManager()
        cbir_path = manager.prepare_for_cbir(
            args.dataset_source,
            target_size=(224, 224),
            split_ratio=0.8
        )
        args.train_dir = str(cbir_path / 'train')
        print(f"Dataset prepared at: {cbir_path}")
    
    # Check if training directory exists
    if not os.path.exists(args.train_dir):
        print(f"Error: Training directory not found: {args.train_dir}")
        print("\nTo prepare a dataset, use:")
        print("  python train_model2.py --prepare-dataset --dataset-source /path/to/images")
        print("\nOr use the demo dataset:")
        print("  python setup_datasets.py --demo")
        print("  python train_model2.py --train-dir datasets/demo_dataset_cbir/train")
        sys.exit(1)
    
    # Train model
    try:
        model_path, history = train_model2(
            train_dir=args.train_dir,
            architecture=args.architecture,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            output_dir=args.output_dir
        )
        
        print(f"\n✓ Success! Model saved to: {model_path}")
        print("\nTo use the model:")
        print("  1. Start the web app: python app.py")
        print("  2. Go to 'Model II Training' tab")
        print("  3. Load the trained model")
        print("  4. Use 'Class-Based' option in Image Search")
        
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()