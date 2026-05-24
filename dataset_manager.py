"""
Dataset Management Module for DNA-CBIR
Handles dataset downloading, organization, and preprocessing
"""
import os
import urllib.request
import zipfile
import tarfile
from pathlib import Path
import shutil
import json
import cv2
import numpy as np

class DatasetManager:
    """Manages datasets for DNA-CBIR experiments"""
    
    DATASET_CONFIGS = {
        'corel-1k': {
            'name': 'Corel-1K',
            'url': 'http://www.ci.gxnu.edu.cn/cbir/Dataset/Corel-1k.zip',
            'classes': 10,
            'images_per_class': 100,
            'total_images': 1000,
            'description': '10 categories with 100 images each'
        },
        'corel-5k': {
            'name': 'Corel-5K',
            'classes': 50,
            'images_per_class': 100,
            'total_images': 5000,
            'description': '50 categories with 100 images each'
        },
        'corel-10k': {
            'name': 'Corel-10K',
            'classes': 100,
            'images_per_class': 100,
            'total_images': 10000,
            'description': '100 categories with 100 images each'
        },
        'caltech-101': {
            'name': 'Caltech-101',
            'url': 'https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip',
            'classes': 101,
            'total_images': 9144,
            'description': '101 object categories'
        },
        'flowers-17': {
            'name': 'Oxford Flowers-17',
            'url': 'https://www.robots.ox.ac.uk/~vgg/data/flowers/17/17flowers.tgz',
            'classes': 17,
            'total_images': 1360,
            'description': '17 flower categories, 80 images each'
        }
    }
    
    def __init__(self, base_dir='datasets'):
        """
        Initialize dataset manager
        
        Args:
            base_dir: Base directory for storing datasets
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
    def list_available_datasets(self):
        """List all available datasets"""
        print("\n" + "="*70)
        print("AVAILABLE DATASETS FOR DNA-CBIR")
        print("="*70)
        
        for key, config in self.DATASET_CONFIGS.items():
            print(f"\n{config['name']} ({key}):")
            print(f"  Classes: {config['classes']}")
            print(f"  Total Images: {config['total_images']}")
            print(f"  Description: {config['description']}")
            
            # Check if downloaded
            dataset_path = self.base_dir / key
            if dataset_path.exists():
                print(f"  Status: ✓ Downloaded")
            else:
                print(f"  Status: ✗ Not downloaded")
        
        print("\n" + "="*70)
    
    def download_dataset(self, dataset_key, force=False):
        """
        Download a dataset
        
        Args:
            dataset_key: Key of the dataset to download
            force: Force re-download if already exists
        """
        if dataset_key not in self.DATASET_CONFIGS:
            print(f"Error: Unknown dataset '{dataset_key}'")
            return False
        
        config = self.DATASET_CONFIGS[dataset_key]
        dataset_dir = self.base_dir / dataset_key
        
        # Check if already exists
        if dataset_dir.exists() and not force:
            print(f"{config['name']} already downloaded at {dataset_dir}")
            return True
        
        # Check if URL is available
        if 'url' not in config:
            print(f"Note: {config['name']} requires manual download")
            print(f"Please download from the official source and place in: {dataset_dir}")
            return False
        
        print(f"\nDownloading {config['name']}...")
        print(f"URL: {config['url']}")
        
        try:
            # Create temp directory
            temp_dir = self.base_dir / 'temp'
            temp_dir.mkdir(exist_ok=True)
            
            # Download file
            filename = config['url'].split('/')[-1]
            filepath = temp_dir / filename
            
            print(f"Downloading to {filepath}...")
            urllib.request.urlretrieve(config['url'], filepath)
            
            # Extract
            print("Extracting...")
            if filename.endswith('.zip'):
                with zipfile.ZipFile(filepath, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
            elif filename.endswith(('.tar.gz', '.tgz')):
                with tarfile.open(filepath, 'r:gz') as tar_ref:
                    tar_ref.extractall(temp_dir)
            
            # Move to final location
            dataset_dir.mkdir(exist_ok=True)
            
            # Find extracted folder
            extracted_items = list(temp_dir.iterdir())
            for item in extracted_items:
                if item != filepath and item.is_dir():
                    shutil.move(str(item), str(dataset_dir / item.name))
            
            # Cleanup
            shutil.rmtree(temp_dir)
            
            print(f"✓ Successfully downloaded {config['name']} to {dataset_dir}")
            return True
            
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            return False
    
    def create_sample_dataset(self, name='sample', num_classes=5, images_per_class=20):
        """
        Create a sample synthetic dataset for testing
        
        Args:
            name: Name of the dataset
            num_classes: Number of classes
            images_per_class: Images per class
        """
        print(f"\nCreating sample dataset '{name}'...")
        
        dataset_dir = self.base_dir / name
        dataset_dir.mkdir(exist_ok=True)
        
        class_names = ['red', 'green', 'blue', 'yellow', 'purple'][:num_classes]
        colors = [
            (200, 50, 50),    # Red
            (50, 200, 50),    # Green
            (50, 50, 200),    # Blue
            (200, 200, 50),   # Yellow
            (150, 50, 200)    # Purple
        ]
        
        total_images = 0
        
        for class_idx, class_name in enumerate(class_names):
            class_dir = dataset_dir / class_name
            class_dir.mkdir(exist_ok=True)
            
            color = colors[class_idx]
            
            for img_idx in range(images_per_class):
                # Create image with some variation
                img = np.zeros((128, 128, 3), dtype=np.uint8)
                
                # Add base color with noise
                noise = np.random.randint(-30, 30, (128, 128, 3))
                img[:, :] = color
                img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
                
                # Add some patterns
                if img_idx % 3 == 0:
                    # Horizontal stripes
                    img[::4, :] = 255
                elif img_idx % 3 == 1:
                    # Vertical stripes
                    img[:, ::4] = 255
                else:
                    # Random dots
                    for _ in range(50):
                        x, y = np.random.randint(0, 128, 2)
                        cv2.circle(img, (x, y), 5, (255, 255, 255), -1)
                
                # Save image
                filename = f"{class_name}_{img_idx:03d}.png"
                filepath = class_dir / filename
                cv2.imwrite(str(filepath), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                total_images += 1
        
        print(f"✓ Created {total_images} images in {num_classes} classes")
        print(f"  Location: {dataset_dir}")
        
        # Save metadata
        metadata = {
            'name': name,
            'classes': num_classes,
            'class_names': class_names,
            'images_per_class': images_per_class,
            'total_images': total_images
        }
        
        with open(dataset_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return dataset_dir
    
    def organize_dataset(self, dataset_path, output_path=None):
        """
        Organize dataset into standard structure (class-based folders)
        
        Args:
            dataset_path: Path to unorganized dataset
            output_path: Output path for organized dataset
        """
        dataset_path = Path(dataset_path)
        if output_path is None:
            output_path = dataset_path.parent / f"{dataset_path.name}_organized"
        
        output_path = Path(output_path)
        output_path.mkdir(exist_ok=True)
        
        print(f"\nOrganizing dataset from {dataset_path}...")
        
        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(dataset_path.rglob(f"*{ext}"))
            image_files.extend(dataset_path.rglob(f"*{ext.upper()}"))
        
        print(f"Found {len(image_files)} images")
        
        # Group by class (assuming class name is in filename or parent dir)
        class_groups = {}
        
        for img_path in image_files:
            # Try to determine class from parent directory
            if img_path.parent != dataset_path:
                class_name = img_path.parent.name
            else:
                # Try to extract from filename
                class_name = img_path.stem.split('_')[0]
            
            if class_name not in class_groups:
                class_groups[class_name] = []
            class_groups[class_name].append(img_path)
        
        # Copy to organized structure
        for class_name, images in class_groups.items():
            class_dir = output_path / class_name
            class_dir.mkdir(exist_ok=True)
            
            for img_path in images:
                shutil.copy2(img_path, class_dir / img_path.name)
        
        print(f"✓ Organized into {len(class_groups)} classes at {output_path}")
        return output_path
    
    def prepare_for_cbir(self, dataset_path, target_size=(256, 256), split_ratio=0.8):
        """
        Prepare dataset for CBIR experiments
        
        Args:
            dataset_path: Path to dataset
            target_size: Resize images to this size
            split_ratio: Train/test split ratio
        """
        dataset_path = Path(dataset_path)
        
        print(f"\nPreparing dataset for CBIR...")
        print(f"  Source: {dataset_path}")
        print(f"  Target size: {target_size}")
        print(f"  Split ratio: {split_ratio}")
        
        # Create output directories
        output_dir = dataset_path.parent / f"{dataset_path.name}_cbir"
        train_dir = output_dir / 'train'
        test_dir = output_dir / 'test'
        database_dir = output_dir / 'database'
        
        for dir_path in [train_dir, test_dir, database_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Get all class directories
        class_dirs = [d for d in dataset_path.iterdir() if d.is_dir()]
        
        total_train = 0
        total_test = 0
        
        for class_dir in class_dirs:
            class_name = class_dir.name
            
            # Create class subdirectories
            (train_dir / class_name).mkdir(exist_ok=True)
            (test_dir / class_name).mkdir(exist_ok=True)
            
            # Get all images
            image_files = []
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                image_files.extend(class_dir.glob(f"*{ext}"))
                image_files.extend(class_dir.glob(f"*{ext.upper()}"))
            
            # Shuffle and split
            np.random.shuffle(image_files)
            split_idx = int(len(image_files) * split_ratio)
            train_files = image_files[:split_idx]
            test_files = image_files[split_idx:]
            
            # Process train images
            for img_path in train_files:
                img = cv2.imread(str(img_path))
                if img is not None:
                    img = cv2.resize(img, target_size)
                    out_path = train_dir / class_name / img_path.name
                    cv2.imwrite(str(out_path), img)
                    
                    # Copy to database
                    db_out_path = database_dir / f"{class_name}_{img_path.name}"
                    cv2.imwrite(str(db_out_path), img)
                    total_train += 1
            
            # Process test images
            for img_path in test_files:
                img = cv2.imread(str(img_path))
                if img is not None:
                    img = cv2.resize(img, target_size)
                    out_path = test_dir / class_name / img_path.name
                    cv2.imwrite(str(out_path), img)
                    total_test += 1
        
        # Save metadata
        metadata = {
            'source': str(dataset_path),
            'target_size': target_size,
            'split_ratio': split_ratio,
            'num_classes': len(class_dirs),
            'total_train': total_train,
            'total_test': total_test
        }
        
        with open(output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✓ Prepared dataset:")
        print(f"  Train: {total_train} images")
        print(f"  Test: {total_test} images")
        print(f"  Database: {total_train} images")
        print(f"  Location: {output_dir}")
        
        return output_dir
    
    def load_dataset_metadata(self, dataset_path):
        """Load dataset metadata if exists"""
        metadata_path = Path(dataset_path) / 'metadata.json'
        
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                return json.load(f)
        
        return None
    
    def get_class_mapping(self, dataset_path):
        """Get class name to index mapping"""
        dataset_path = Path(dataset_path)
        class_dirs = sorted([d.name for d in dataset_path.iterdir() if d.is_dir()])
        return {name: idx for idx, name in enumerate(class_dirs)}


def main():
    """Demo dataset management"""
    manager = DatasetManager()
    
    # List available datasets
    manager.list_available_datasets()
    
    # Create sample dataset
    print("\n" + "="*70)
    sample_path = manager.create_sample_dataset(
        name='sample_dataset',
        num_classes=5,
        images_per_class=20
    )
    
    # Prepare for CBIR
    print("\n" + "="*70)
    cbir_path = manager.prepare_for_cbir(sample_path, target_size=(128, 128))
    
    # Show class mapping
    print("\n" + "="*70)
    print("CLASS MAPPING:")
    class_mapping = manager.get_class_mapping(cbir_path / 'train')
    for class_name, idx in class_mapping.items():
        print(f"  {idx}: {class_name}")
    
    print("\n✓ Dataset management demo completed!")
    print(f"\nYou can now use the dataset at: {cbir_path}")
    print("- Use 'database/' folder for CBIR experiments")
    print("- Use 'train/' and 'test/' for training/evaluation")


if __name__ == "__main__":
    main()