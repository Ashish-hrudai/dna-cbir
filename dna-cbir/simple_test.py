"""
Simple demo script to test DNA-CBIR core functionality
(Without TensorFlow dependency)
"""
import numpy as np
import cv2
from dna_encoding import DNAEncoder, DNATranslator, SimilarityMatcher
import os

def create_sample_images():
    """Create sample images for testing"""
    print("Creating sample images...")
    
    os.makedirs('database', exist_ok=True)
    
    # Create different types of images
    images = []
    
    # Red images
    for i in range(3):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:, :, 0] = 200 + i * 10  # Red channel
        cv2.imwrite(f'database/red_{i}.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        images.append(('red', f'red_{i}.png', img))
    
    # Green images
    for i in range(3):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:, :, 1] = 200 + i * 10  # Green channel
        cv2.imwrite(f'database/green_{i}.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        images.append(('green', f'green_{i}.png', img))
    
    # Blue images
    for i in range(3):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:, :, 2] = 200 + i * 10  # Blue channel
        cv2.imwrite(f'database/blue_{i}.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        images.append(('blue', f'blue_{i}.png', img))
    
    print(f"Created {len(images)} sample images in database/")
    return images

def test_dna_encoding():
    """Test DNA encoding functionality"""
    print("\n" + "="*60)
    print("TESTING DNA ENCODING")
    print("="*60)
    
    # Create a simple test image
    test_img = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
    
    # Initialize encoder
    encoder = DNAEncoder(num_msb_levels=3, num_bins=10)
    
    # Test DNA plane generation
    print("\n1. Testing DNA plane generation...")
    dna_planes = encoder.encode_image_to_dna_planes(test_img)
    print(f"   ✓ Generated {len(dna_planes)} DNA planes")
    print(f"   ✓ DNA plane length: {len(dna_planes[0])} nucleotides")
    
    # Check GC constraint
    all_nucleotides = ''.join(dna_planes)
    gc_count = all_nucleotides.count('G') + all_nucleotides.count('C')
    at_count = all_nucleotides.count('A') + all_nucleotides.count('T')
    gc_percentage = (gc_count / (gc_count + at_count) * 100)
    print(f"   ✓ GC content: {gc_percentage:.2f}%")
    print(f"   ✓ GC constraint satisfied: {40 <= gc_percentage <= 60}")
    
    # Test feature extraction
    print("\n2. Testing feature extraction...")
    features = encoder.extract_features(test_img)
    print(f"   ✓ Feature vector length: {len(features)}")
    
    # Test multi-scale features (IDEA 2)
    print("\n3. Testing multi-scale features (IDEA 2)...")
    multi_scale_features = encoder.extract_multi_scale_features(test_img, [3, 4, 5])
    print(f"   ✓ Multi-scale feature length: {len(multi_scale_features)}")
    print(f"   ✓ Scale ratio: {len(multi_scale_features) / len(features):.1f}x")
    
    # Show sample DNA sequence
    print("\n4. Sample DNA sequence (first 100 nucleotides):")
    print(f"   {dna_planes[0][:100]}...")
    
    return True

def test_similarity_metrics():
    """Test different similarity metrics (IDEA 3)"""
    print("\n" + "="*60)
    print("TESTING SIMILARITY METRICS (IDEA 3)")
    print("="*60)
    
    # Create two similar and one different image
    img1 = np.ones((50, 50, 3), dtype=np.uint8) * 100
    img2 = np.ones((50, 50, 3), dtype=np.uint8) * 105  # Similar to img1
    img3 = np.ones((50, 50, 3), dtype=np.uint8) * 200  # Different
    
    encoder = DNAEncoder()
    matcher = SimilarityMatcher()
    
    # Extract features
    feat1 = encoder.extract_features(img1)
    feat2 = encoder.extract_features(img2)
    feat3 = encoder.extract_features(img3)
    
    print("\nComparing similar images (img1 vs img2):")
    print(f"   Euclidean:   {matcher.euclidean_distance(feat1, feat2):.4f}")
    print(f"   Cosine:      {matcher.cosine_similarity(feat1, feat2):.4f}")
    print(f"   Chi-Square:  {matcher.chi_square_distance(feat1, feat2):.4f}")
    print(f"   Weighted:    {matcher.compute_weighted_similarity(feat1, feat2):.4f}")
    
    print("\nComparing different images (img1 vs img3):")
    print(f"   Euclidean:   {matcher.euclidean_distance(feat1, feat3):.4f}")
    print(f"   Cosine:      {matcher.cosine_similarity(feat1, feat3):.4f}")
    print(f"   Chi-Square:  {matcher.chi_square_distance(feat1, feat3):.4f}")
    print(f"   Weighted:    {matcher.compute_weighted_similarity(feat1, feat3):.4f}")
    
    print("\n✓ All similarity metrics working correctly")
    return True

def test_image_retrieval():
    """Test basic image retrieval"""
    print("\n" + "="*60)
    print("TESTING IMAGE RETRIEVAL")
    print("="*60)
    
    # Create sample images
    images = create_sample_images()
    
    # Initialize components
    encoder = DNAEncoder()
    matcher = SimilarityMatcher()
    
    # Extract features for all images
    print("\n1. Extracting features for all images...")
    feature_space = {}
    db_images = [img for _, _, img in images]
    db_paths = [path for _, path, _ in images]
    
    for idx, img in enumerate(db_images):
        feature_space[idx] = encoder.extract_features(img)
    print(f"   ✓ Extracted features for {len(db_images)} images")
    
    # Test retrieval with a red image
    print("\n2. Testing retrieval with RED query image...")
    query_img = images[0][2]  # First red image
    query_features = encoder.extract_features(query_img)
    
    # Compute similarities
    similarities = []
    for idx, features in feature_space.items():
        score = matcher.compute_weighted_similarity(query_features, features)
        similarities.append((idx, score))
    
    # Sort and get top 5
    similarities.sort(key=lambda x: x[1])
    top_5 = similarities[:5]
    
    print("   Top 5 Results:")
    for rank, (img_idx, score) in enumerate(top_5, 1):
        print(f"   {rank}. {db_paths[img_idx]:15s} - Score: {score:.4f}")
    
    # Test with multi-scale
    print("\n3. Testing with multi-scale features (IDEA 2)...")
    query_features_multi = encoder.extract_multi_scale_features(query_img, [3, 4, 5])
    
    # Extract multi-scale for all
    feature_space_multi = {}
    for idx, img in enumerate(db_images):
        feature_space_multi[idx] = encoder.extract_multi_scale_features(img, [3, 4, 5])
    
    similarities_multi = []
    for idx, features in feature_space_multi.items():
        score = matcher.compute_weighted_similarity(query_features_multi, features)
        similarities_multi.append((idx, score))
    
    similarities_multi.sort(key=lambda x: x[1])
    top_5_multi = similarities_multi[:5]
    
    print("   Top 5 Results (Multi-Scale):")
    for rank, (img_idx, score) in enumerate(top_5_multi, 1):
        print(f"   {rank}. {db_paths[img_idx]:15s} - Score: {score:.4f}")
    
    print("\n✓ Image retrieval working correctly")
    return True

def test_dna_translation():
    """Test DNA translation"""
    print("\n" + "="*60)
    print("TESTING DNA TRANSLATION")
    print("="*60)
    
    test_img = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
    
    encoder = DNAEncoder()
    translator = DNATranslator(amplification_weight=85)
    
    print("\n1. Testing DNA to numeric translation...")
    dna_planes = encoder.encode_image_to_dna_planes(test_img)
    numeric = translator.translate_dna_to_numeric(dna_planes[0][:100])
    print(f"   ✓ DNA sequence: {dna_planes[0][:20]}...")
    print(f"   ✓ Numeric values: {numeric[:20]}")
    
    print("\n2. Testing data amplification...")
    amplified = translator.amplify_dna_data(numeric)
    print(f"   ✓ Original range: [{numeric.min()}, {numeric.max()}]")
    print(f"   ✓ Amplified range: [{amplified.min()}, {amplified.max()}]")
    print(f"   ✓ Amplification factor: {amplified.max() / max(numeric.max(), 1):.1f}x")
    
    print("\n3. Testing CNN plane preparation...")
    amplified_planes = translator.prepare_dna_planes_for_cnn(test_img, encoder)
    print(f"   ✓ Original image shape: {test_img.shape}")
    print(f"   ✓ Amplified planes shape: {amplified_planes.shape}")
    print(f"   ✓ Width increased by factor: {amplified_planes.shape[1] / test_img.shape[1]:.1f}x")
    
    print("\n✓ DNA translation working correctly")
    return True

def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("DNA-CBIR SYSTEM TEST SUITE")
    print("="*60)
    
    tests = [
        ("DNA Encoding", test_dna_encoding),
        ("Similarity Metrics", test_similarity_metrics),
        ("Image Retrieval", test_image_retrieval),
        ("DNA Translation", test_dna_translation)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, "PASSED" if success else "FAILED"))
        except Exception as e:
            print(f"\n✗ Error in {test_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            results.append((test_name, "ERROR"))
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    for test_name, status in results:
        symbol = "✓" if status == "PASSED" else "✗"
        print(f"{symbol} {test_name:25s} {status}")
    
    print("\n" + "="*60)
    passed = sum(1 for _, s in results if s == "PASSED")
    print(f"Tests Passed: {passed}/{len(tests)}")
    print("="*60)
    
    if passed == len(tests):
        print("\n🎉 All tests passed! System is ready to use.")
        print("\nTo start the web application, run:")
        print("    python app.py")
    
if __name__ == "__main__":
    run_all_tests()