import torch
import librosa
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from models.mfcc_cnn import MFCC_CNN

class GenreClassifier:
    """
    Music genre classifier using MFCC features and CNN.
    This class handles loading the model and making predictions on audio files.
    """
    
    def __init__(self, model_path, device=None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load genre mapping
        data_path = Path(__file__).parent.parent.parent / "data" / "processed" / "mfcc_data.npz"
        data = np.load(data_path, allow_pickle=True)
        self.genres = data["mapping"].tolist()
        
        # MFCC parameters (loading from the training data to ensure consistency)
        mfcc_shape = data["mfcc"].shape  # (N, time, n_mfcc)
        self.expected_mfcc_vectors_per_segment = mfcc_shape[1]  #time dimension
        self.n_mfcc = mfcc_shape[2]  # feature dimension
        
        # Omust match feature_extractor.py
        self.n_fft = 2048
        self.hop_length = 512
        self.sample_rate = 22050
        self.num_segments = 5
        
       
        self.num_samples_per_segment = (self.expected_mfcc_vectors_per_segment - 1) * self.hop_length
        self.samples_per_track = self.num_samples_per_segment * self.num_segments
        self.track_duration = self.samples_per_track / self.sample_rate
        
        print(f"Loaded MFCC parameters from training data:")
        print(f"  Expected shape per segment: ({self.expected_mfcc_vectors_per_segment}, {self.n_mfcc})")
        print(f"  Track duration: {self.track_duration:.1f} seconds")
        
        self.model = MFCC_CNN(
            num_classes=len(self.genres), 
            input_shape=(1, self.expected_mfcc_vectors_per_segment, self.n_mfcc)
        )
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded on {self.device}")
        print(f"Genres: {self.genres}")
    
    def extract_mfcc(self, audio_path):
        """
        Extract MFCC features from an audio file.
        Handles any duration by padding/truncating to match training length.
        """
        signal, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        # pad or truncate to expected total length
        if len(signal) < self.samples_per_track:
            # pad short audio with zeros
            signal = np.pad(signal, (0, self.samples_per_track - len(signal)), mode='constant')
        else:
            # truncate long audio to 30 s
            signal = signal[:self.samples_per_track]
        
        mfccs = []
        
        for s in range(self.num_segments):
            start_sample = self.num_samples_per_segment * s
            finish_sample = start_sample + self.num_samples_per_segment
            
            # extract MFCC for this segment
            mfcc = librosa.feature.mfcc(
                y=signal[start_sample:finish_sample],
                sr=sr,
                n_mfcc=self.n_mfcc,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )
            mfcc = mfcc.T  
            
            #ensure shape
            if len(mfcc) < self.expected_mfcc_vectors_per_segment:
                # Pad short
                pad_width = self.expected_mfcc_vectors_per_segment - len(mfcc)
                mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode='constant')
            elif len(mfcc) > self.expected_mfcc_vectors_per_segment:
                # Truncate if long
                mfcc = mfcc[:self.expected_mfcc_vectors_per_segment, :]
            
            mfccs.append(mfcc)
        
        return np.array(mfccs)
    
    def predict(self, audio_path, return_probabilities=False):
        """
        Predict the genre of an audio file.
        Returns:
            If return_probabilities=False: genre name (str)
            If return_probabilities=True: dict with 'genre', 'confidence', 'probabilities'
        """
        mfccs = self.extract_mfcc(audio_path)
        
        mfccs = mfccs[:, None, :, :]
        
        X = torch.from_numpy(mfccs).float().to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(X)
            probabilities = torch.softmax(outputs, dim=1)
            
            # Average predictions across segments
            avg_probabilities = probabilities.mean(dim=0)
            predicted_idx = torch.argmax(avg_probabilities).item()
            confidence = avg_probabilities[predicted_idx].item()
        
        predicted_genre = self.genres[predicted_idx]
        
        if return_probabilities:
            # Return detailed results
            all_probs = {
                genre: float(avg_probabilities[i].cpu())
                for i, genre in enumerate(self.genres)
            }
            return {
                'genre': predicted_genre,
                'confidence': confidence,
                'probabilities': all_probs
            }
        else:
            return predicted_genre
    
    
def classify_song(audio_path, model_path=None):
   
    if model_path is None:
        model_path = Path(__file__).parent.parent.parent / "outputs" / "mfcc_cnn" / "mfcc_cnn_trained.pth"
    
    classifier = GenreClassifier(model_path)
    return classifier.predict(audio_path, return_probabilities=True)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Classify music genre from audio file')
    parser.add_argument('audio_file', type=str, help='Path to audio file')
    parser.add_argument('--model', type=str, default=None, help='Path to model file')
    parser.add_argument('--device', type=str, default=None, choices=['cuda', 'cpu'], 
                       help='Device to run inference on')
    
    args = parser.parse_args()
    
    model_path = args.model
    if model_path is None:
        model_path = Path(__file__).parent.parent.parent / "outputs" / "mfcc_cnn" / "mfcc_cnn_trained.pth"
    
    classifier = GenreClassifier(model_path, device=args.device)
    
    print(f"\nAnalyzing: {args.audio_file}")
    print("-" * 60)
    
    result = classifier.predict(args.audio_file, return_probabilities=True)
    
    print(f"\nPredicted Genre: {result['genre'].upper()}")
    print(f"Confidence: {result['confidence']*100:.2f}%")
    print("\nAll Probabilities:")
    
    # Sort by probability
    sorted_probs = sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True)
    for genre, prob in sorted_probs:
        bar = "█" * int(prob * 50)
        print(f"  {genre:12s} {prob*100:5.2f}% {bar}")