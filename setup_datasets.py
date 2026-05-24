"""
Quick Dataset Setup Script
Downloads and prepares datasets for DNA-CBIR experiments
"""
import argparse
from dataset_manager import DatasetManager
import os

def setup_quick_demo():
    """Setup a quick demo with sample dataset"""
    print("\n" + "="*70)
    print("QUICK DEMO SETUP")
    print("="*70)
    
    manager = DatasetManager()
    
    # Create sample dataset
    print("\n1. Creating sample dataset...")
    sample_path = manager.create_sample_dataset(
        name='demo_dataset',
        num_classes=5,
        images_per_class=30
    )
    
    # Prepare for CBIR
    print("\n2. Preparing for CBIR...")
    cbir_path = manager.prepare_for_cbir(
        sample_path,
        target_size=(128, 128),
        split_ratio=0.8
    )
    
    # Copy database images to main database folder
    import shutil
    from pathlib import Path
    
    main_db = Path('database')
    main_db.mkdir(exist_ok=True)
    
    db_source = cbir_path / 'database'
    for img in db_source.glob('*.png'):
        shutil.copy(img, main_db / img.name)
    
    print("\n" + "="*70)
    print("✓ DEMO SETUP COMPLETE!")
    print("="*70)
    print(f"\nDataset location: {cbir_path}")
    print(f"Database images: {len(list(main_db.glob('*.png')))} images copied to database/")
    print("\nYou can now:")
    print("  1. python app.py           # Start the web application")
    print("  2. Go to Database tab       # See the images")
    print("  3. Try Image Search         # Search similar images")


def setup_custom_dataset(path):
    """Setup a custom dataset from a directory"""
    print("\n" + "="*70)
    print("CUSTOM DATASET SETUP")
    print("="*70)
    
    manager = DatasetManager()
    
    # Organize dataset
    print("\n1. Organizing dataset...")
    organized_path = manager.organize_dataset(path)
    
    # Prepare for CBIR
    print("\n2. Preparing for CBIR...")
    cbir_path = manager.prepare_for_cbir(
        organized_path,
        target_size=(256, 256),
        split_ratio=0.8
    )
    
    # Copy to database
    import shutil
    from pathlib import Path
    
    main_db = Path('database')
    main_db.mkdir(exist_ok=True)
    
    db_source = cbir_path / 'database'
    count = 0
    for img in db_source.glob('*.*'):
        if img.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
            shutil.copy(img, main_db / img.name)
            count += 1
    
    print("\n" + "="*70)
    print("✓ CUSTOM DATASET SETUP COMPLETE!")
    print("="*70)
    print(f"\nDataset location: {cbir_path}")
    print(f"Database images: {count} images copied to database/")


def list_datasets():
    """List all available datasets"""
    manager = DatasetManager()
    manager.list_available_datasets()
    
    print("\n" + "="*70)
    print("MANUAL DOWNLOAD INSTRUCTIONS")
    print("="*70)
    print("\nFor datasets that require manual download:")
    print("\n1. Corel-1K (Recommended for quick testing):")
    print("   - Often available from: http://wang.ist.psu.edu/docs/related/")
    print("   - Or search: 'Corel-1K image dataset download'")
    
    print("\n2. Caltech-101:")
    print("   - Official: https://data.caltech.edu/records/mzrjq-6wc02")
    print("   - Download and extract to datasets/caltech-101/")
    
    print("\n3. Oxford Flowers-17:")
    print("   - Official: https://www.robots.ox.ac.uk/~vgg/data/flowers/17/")
    print("   - Download 17flowers.tgz")
    
    print("\n4. PASCAL VOC (for multi-label):")
    print("   - Official: http://host.robots.ox.ac.uk/pascal/VOC/")
    print("   - Download VOC2012")
    
    print("\n5. Other recommended datasets:")
    print("   - CIFAR-10: https://www.cs.toronto.edu/~kriz/cifar.html")
    print("   - ImageNet subset: https://image-net.org/")
    print("   - COCO: https://cocodataset.org/")


def main():
    parser = argparse.ArgumentParser(description='Setup datasets for DNA-CBIR')
    parser.add_argument('--demo', action='store_true', 
                       help='Setup quick demo with sample dataset')
    parser.add_argument('--custom', type=str,
                       help='Path to custom dataset directory')
    parser.add_argument('--list', action='store_true',
                       help='List available datasets')
    
    args = parser.parse_args()
    
    if args.demo:
        setup_quick_demo()
    elif args.custom:
        setup_custom_dataset(args.custom)
    elif args.list:
        list_datasets()
    else:
        print("\nDNA-CBIR Dataset Setup")
        print("="*50)
        print("\nUsage:")
        print("  python setup_datasets.py --demo          # Quick demo")
        print("  python setup_datasets.py --custom PATH   # Custom dataset")
        print("  python setup_datasets.py --list          # List datasets")
        print("\nExamples:")
        print("  python setup_datasets.py --demo")
        print("  python setup_datasets.py --custom /path/to/images")


if __name__ == "__main__":
    main()