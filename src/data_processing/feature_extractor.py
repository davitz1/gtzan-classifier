import os 
import librosa
import json
import math
import numpy as np

SAMPLE_RATE=22050
DURATION=30
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

def save_mfcc(dataset_path, output_path, n_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):
    """Extracts MFCCs from GTZAN dataset and stores the following into a .npz file:
    -mfcc(N, time, n_mfcc)
    -labels(N,)
    -mapping: genre index -> genre name
    """
    mapping = []
    mfccs = []
    labels = []
    num_samples_per_segment = int(SAMPLES_PER_TRACK/num_segments)
    expected_num_mfcc_vectors_per_segment = math.ceil(num_samples_per_segment/hop_length)

    #looping through all genres
    for dirpath, dirnames, filenames in os.walk(dataset_path):
        if dirpath != str(dataset_path): 
            genre = os.path.basename(dirpath)
            mapping.append(genre)
            label = len(mapping)-1
            print(f"Processing: {genre}")

            #process files for a genre
            for f in filenames:
                file_path = os.path.join(dirpath, f)
                try:
                    signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
                except Exception as e:
                    print(f"Could not load {file_path}: {e}")
                    continue
                
                #process segments extracting mfccs
                for s in range(num_segments):
                    start_sample = num_samples_per_segment * s
                    finish_sample = start_sample + num_samples_per_segment
                    
                    mfcc = librosa.feature.mfcc(y=signal[start_sample:finish_sample],
                                                sr=sr,
                                                n_fft=n_fft,
                                                n_mfcc=n_mfcc,
                                                hop_length=hop_length).T
                    
                    #store mfccs and labels
                    if len(mfcc) == expected_num_mfcc_vectors_per_segment:
                        mfccs.append(mfcc)
                        labels.append(label)
    
    #save the data
    np.savez(output_path, mfcc=np.array(mfccs), labels=np.array(labels), mapping=np.array(mapping))
    print(f"Data saved to {output_path}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_root = os.path.dirname(os.path.dirname(current_dir))
    
    DATASET_PATH = os.path.join(workspace_root, "data", "gtzan", "genres_original")
    OUTPUT_PATH = os.path.join(workspace_root, "data", "processed", "mfcc_data.npz")
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    save_mfcc(DATASET_PATH, OUTPUT_PATH, num_segments=10)
                
