"""
DNA Encoding Module for Image Feature Extraction
Implements the DNA-CBIR encoding scheme from the paper
"""

import numpy as np
from typing import Tuple, List, Dict
import cv2


class DNAEncoder:
    """
    DNA Encoding class that converts images to DNA planes using MSB encoding
    """
    
    # DNA Trees for encoding (as per Fig. 3 in paper)
    TREE_T = {
        '000': 'ATA',
        '001': 'AGT',
        '010': 'ACA',
        '011': 'ACG',
        '100': 'TTA',
        '101': 'TGT',
        '110': 'TCA',
        '111': 'TCG'
    }
    
    TREE_T_PRIME = {
        '000': 'GAG',
        '001': 'GAC',
        '010': 'GCG',
        '011': 'GCC',
        '100': 'CAG',
        '101': 'CAC',
        '110': 'CCG',
        '111': 'CCC'
    }
    
    # Codon groups (as per TABLE III)
    CODON_GROUPS = {
        'ATA': 0, 'GAG': 0,  # Group 0
        'AGT': 1, 'GAC': 1,  # Group 1
        'ACA': 2, 'GCG': 2,  # Group 2
        'ACG': 3, 'GCC': 3,  # Group 3
        'TTA': 4, 'CAG': 4,  # Group 4
        'TGT': 5, 'CAC': 5,  # Group 5
        'TCA': 6, 'CCG': 6,  # Group 6
        'TCG': 7, 'CCC': 7   # Group 7
    }
    
    def __init__(self, num_msb_levels: int = 3, num_bins: int = 10):
        """
        Initialize DNA Encoder
        
        Args:
            num_msb_levels: Number of MSB levels to extract (default: 3 from paper, extended for idea 2)
            num_bins: Number of bins for histogram (default: 10)
        """
        self.num_msb_levels = num_msb_levels
        self.num_bins = num_bins
        
    def extract_msb_bits(self, pixel_value: int, num_bits: int = 3) -> str:
        """
        Extract the most significant bits from a pixel value
        
        Args:
            pixel_value: Pixel intensity value (0-255)
            num_bits: Number of MSB to extract
            
        Returns:
            Binary string of MSB
        """
        binary = format(pixel_value, '08b')
        return binary[:num_bits]
    
    def encode_pixel_to_dna(self, pixel_value: int, is_odd: bool, num_bits: int = 3) -> str:
        """
        Encode a pixel value to DNA nucleotides
        
        Args:
            pixel_value: Pixel intensity value
            is_odd: Whether pixel is at odd position (for GC constraint)
            num_bits: Number of MSB to extract
            
        Returns:
            DNA nucleotide sequence
        """
        msb = self.extract_msb_bits(pixel_value, num_bits)
        
        if is_odd:
            return self.TREE_T[msb]
        else:
            return self.TREE_T_PRIME[msb]
    
    def encode_channel_to_dna(self, channel: np.ndarray) -> str:
        """
        Encode a single channel to DNA sequence (Algorithm 1)
        
        Args:
            channel: 2D numpy array representing one channel
            
        Returns:
            DNA sequence string
        """
        dna_sequence = ""
        flat_channel = channel.flatten()
        
        for idx, pixel in enumerate(flat_channel):
            is_odd = (idx % 2 == 0)  # 0-indexed, so even indices are "odd" positions
            dna_codon = self.encode_pixel_to_dna(int(pixel), is_odd)
            dna_sequence += dna_codon
            
        return dna_sequence
    
    def encode_image_to_dna_planes(self, image: np.ndarray) -> List[str]:
        """
        Encode entire image to DNA planes
        
        Args:
            image: Image array (H x W x C)
            
        Returns:
            List of DNA sequences, one per channel
        """
        if len(image.shape) == 2:
            # Grayscale image
            return [self.encode_channel_to_dna(image)]
        else:
            # Color image
            dna_planes = []
            for channel_idx in range(image.shape[2]):
                channel = image[:, :, channel_idx]
                dna_planes.append(self.encode_channel_to_dna(channel))
            return dna_planes
    
    def dna_to_codons(self, dna_sequence: str) -> List[str]:
        """
        Convert DNA sequence to codon sequence (Algorithm 2)
        
        Args:
            dna_sequence: DNA nucleotide sequence
            
        Returns:
            List of codons (triplets)
        """
        codons = []
        for i in range(0, len(dna_sequence) - 2, 3):
            codon = dna_sequence[i:i+3]
            codons.append(codon)
        return codons
    
    def extract_codon_groups(self, codons: List[str]) -> List[int]:
        """
        Map codons to their groups
        
        Args:
            codons: List of codon strings
            
        Returns:
            List of group IDs
        """
        return [self.CODON_GROUPS.get(codon, 0) for codon in codons]
    
    def split_codon_subsequences(self, codon_groups: List[int]) -> Tuple[List[int], List[int]]:
        """
        Split codon groups into T and T' subsequences
        
        Args:
            codon_groups: List of codon group IDs
            
        Returns:
            Tuple of (T_subsequence, T_prime_subsequence)
        """
        t_subseq = [codon_groups[i] for i in range(0, len(codon_groups), 2)]
        t_prime_subseq = [codon_groups[i] for i in range(1, len(codon_groups), 2)]
        
        return t_subseq, t_prime_subseq
    
    def compute_bin_histograms(self, subsequence: List[int]) -> np.ndarray:
        """
        Compute histogram for each bin in subsequence (Algorithm 3)
        
        Args:
            subsequence: List of codon groups
            
        Returns:
            Feature vector from binned histograms
        """
        if len(subsequence) == 0:
            return np.zeros(8 * self.num_bins)
        
        bin_size = max(1, len(subsequence) // self.num_bins)
        features = []
        
        for bin_idx in range(self.num_bins):
            start_idx = bin_idx * bin_size
            end_idx = min((bin_idx + 1) * bin_size, len(subsequence))
            
            if start_idx >= len(subsequence):
                features.extend([0] * 8)
                continue
            
            bin_data = subsequence[start_idx:end_idx]
            
            # Compute histogram for 8 codon groups
            hist = np.zeros(8)
            for group_id in bin_data:
                if 0 <= group_id < 8:
                    hist[group_id] += 1
            
            # Normalize
            if len(bin_data) > 0:
                hist = hist / len(bin_data)
            
            features.extend(hist)
        
        return np.array(features)
    
    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract complete codon-based DNA features (Algorithm 3)
        
        Args:
            image: Input image
            
        Returns:
            Feature vector
        """
        # Encode image to DNA planes
        dna_planes = self.encode_image_to_dna_planes(image)
        
        all_features = []
        
        for dna_sequence in dna_planes:
            # Convert to codons
            codons = self.dna_to_codons(dna_sequence)
            
            # Get codon groups
            codon_groups = self.extract_codon_groups(codons)
            
            # Split into subsequences
            t_subseq, t_prime_subseq = self.split_codon_subsequences(codon_groups)
            
            # Compute features for both subsequences
            t_features = self.compute_bin_histograms(t_subseq)
            t_prime_features = self.compute_bin_histograms(t_prime_subseq)
            
            # Concatenate
            channel_features = np.concatenate([t_features, t_prime_features])
            all_features.append(channel_features)
        
        # Concatenate all channels
        return np.concatenate(all_features)
    
    def extract_multi_scale_features(self, image: np.ndarray, 
                                    msb_levels: List[int] = [3, 4, 5]) -> np.ndarray:
        """
        IDEA 2 IMPLEMENTATION: Multi-scale MSB feature extraction
        Extract features at multiple MSB levels to capture both coarse and fine patterns
        
        Args:
            image: Input image
            msb_levels: List of MSB levels to extract (e.g., [3, 4, 5])
            
        Returns:
            Concatenated multi-scale feature vector
        """
        multi_scale_features = []
        
        for num_bits in msb_levels:
            # Temporarily set MSB level
            original_level = self.num_msb_levels
            self.num_msb_levels = num_bits
            
            # Extract features at this scale
            features = self.extract_features(image)
            multi_scale_features.append(features)
            
            # Restore original level
            self.num_msb_levels = original_level
        
        # Concatenate all scales
        return np.concatenate(multi_scale_features)


class DNATranslator:
    """
    DNA Data Translation for Model II (Class-based CBIR)
    Implements Watson-Crick complementary rules
    """
    
    # Rule 1 from Fig. 8
    NUCLEOTIDE_TO_BINARY = {
        'A': '00',
        'T': '11',
        'G': '01',
        'C': '10'
    }
    
    def __init__(self, amplification_weight: int = 85):
        """
        Initialize DNA Translator
        
        Args:
            amplification_weight: Weight for data amplification (default: 85 from paper)
        """
        self.amplification_weight = amplification_weight
    
    def translate_dna_to_numeric(self, dna_sequence: str) -> np.ndarray:
        """
        Translate DNA sequence to numeric values
        
        Args:
            dna_sequence: DNA nucleotide sequence
            
        Returns:
            Numeric array
        """
        numeric_values = []
        for nucleotide in dna_sequence:
            binary = self.NUCLEOTIDE_TO_BINARY.get(nucleotide, '00')
            numeric_values.append(int(binary, 2))
        
        return np.array(numeric_values)
    
    def amplify_dna_data(self, numeric_data: np.ndarray) -> np.ndarray:
        """
        Amplify DNA data to enhance discriminative power (Equation 1)
        
        Args:
            numeric_data: Numeric DNA plane
            
        Returns:
            Amplified numeric DNA plane
        """
        return self.amplification_weight * numeric_data
    
    def prepare_dna_planes_for_cnn(self, image: np.ndarray, 
                                   encoder: DNAEncoder) -> np.ndarray:
        """
        Prepare amplified DNA planes for CNN input
        
        Args:
            image: Input image
            encoder: DNAEncoder instance
            
        Returns:
            Amplified DNA planes reshaped for CNN (H x 3W x C)
        """
        # Encode to DNA planes
        dna_planes = encoder.encode_image_to_dna_planes(image)
        
        # Translate and amplify each plane
        amplified_planes = []
        for dna_seq in dna_planes:
            numeric = self.translate_dna_to_numeric(dna_seq)
            amplified = self.amplify_dna_data(numeric)
            amplified_planes.append(amplified)
        
        # Reshape to match image dimensions (H x 3W)
        h, w = image.shape[:2]
        num_channels = len(amplified_planes)
        
        reshaped_planes = np.zeros((h, w * 3, num_channels))
        for ch_idx, plane in enumerate(amplified_planes):
            # Reshape to (H, 3W)
            plane_2d = plane.reshape(h, w * 3)
            reshaped_planes[:, :, ch_idx] = plane_2d
        
        # Normalize to [0, 255] range
        reshaped_planes = np.clip(reshaped_planes, 0, 255).astype(np.uint8)
        
        return reshaped_planes


class SimilarityMatcher:
    """
    IDEA 3 IMPLEMENTATION: Multiple similarity metrics for improved retrieval
    """
    
    @staticmethod
    def euclidean_distance(feat1: np.ndarray, feat2: np.ndarray) -> float:
        """Standard Euclidean distance"""
        return np.linalg.norm(feat1 - feat2)
    
    @staticmethod
    def cosine_similarity(feat1: np.ndarray, feat2: np.ndarray) -> float:
        """
        Cosine similarity (more suitable for deep features)
        Returns distance (1 - similarity) for consistency
        """
        dot_product = np.dot(feat1, feat2)
        norm1 = np.linalg.norm(feat1)
        norm2 = np.linalg.norm(feat2)
        
        if norm1 == 0 or norm2 == 0:
            return 1.0
        
        similarity = dot_product / (norm1 * norm2)
        return 1.0 - similarity
    
    @staticmethod
    def chi_square_distance(feat1: np.ndarray, feat2: np.ndarray) -> float:
        """
        Chi-square distance (more suitable for histogram features)
        """
        # Add small epsilon to avoid division by zero
        epsilon = 1e-10
        chi_square = np.sum((feat1 - feat2) ** 2 / (feat1 + feat2 + epsilon))
        return chi_square
    
    @staticmethod
    def compute_weighted_similarity(feat1: np.ndarray, feat2: np.ndarray,
                                   weights: Dict[str, float] = None) -> float:
        """
        Compute weighted combination of multiple similarity metrics
        
        Args:
            feat1: First feature vector
            feat2: Second feature vector
            weights: Dictionary of metric weights
            
        Returns:
            Weighted similarity score
        """
        if weights is None:
            # Default weights optimized for DNA features
            weights = {
                'euclidean': 0.3,
                'cosine': 0.4,
                'chi_square': 0.3
            }
        
        euclidean = SimilarityMatcher.euclidean_distance(feat1, feat2)
        cosine = SimilarityMatcher.cosine_similarity(feat1, feat2)
        chi_square = SimilarityMatcher.chi_square_distance(feat1, feat2)
        
        # Normalize each metric to [0, 1] range
        euclidean_norm = euclidean / (1 + euclidean)
        cosine_norm = cosine
        chi_square_norm = chi_square / (1 + chi_square)
        
        # Weighted combination
        weighted_score = (weights['euclidean'] * euclidean_norm +
                         weights['cosine'] * cosine_norm +
                         weights['chi_square'] * chi_square_norm)
        
        return weighted_score