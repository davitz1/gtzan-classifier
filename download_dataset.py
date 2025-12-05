"""
Download GTZAN dataset from Kaggle.
Requires: pip install kaggle
"""
import subprocess
import sys
from pathlib import Path
import shutil

def download_gtzan_kaggle():
    try:
        import kaggle
    except ImportError:
        print("Installing kaggle...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle"])
    
    # Check for credentials
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    
    # Setup paths
    project_root = Path(__file__).parent
    download_path = project_root / "data" / "gtzan"
    download_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nDownloading to: {download_path}")
    print("This will take a few minutes (1.2 GB)...\n")
    
    # Download and unzip
    dataset = "andradaolteanu/gtzan-dataset-music-genre-classification"
    subprocess.run([
        "kaggle", "datasets", "download", 
        "-d", dataset,
        "-p", str(download_path),
        "--unzip"
    ], check=True)
    
    print("\nDownload complete!")
    
    target_genres = download_path / "genres_original"
    
    # Check if already in correct location
    if not target_genres.exists():
        # Look for it in Data folder
        downloaded_genres = download_path / "Data" / "genres_original"
        
        if downloaded_genres.exists():
            print(f"Moving genres_original to correct location...")
            shutil.move(str(downloaded_genres), str(target_genres))
            
            # Remove the now-empty (or remaining) Data folder
            data_folder = download_path / "Data"
            if data_folder.exists():
                shutil.rmtree(data_folder)
                print(f"Cleaned up Data folder")
    
    # Verify
    if target_genres.exists():
        genres = sorted([d.name for d in target_genres.iterdir() if d.is_dir()])
        print(f"\nFound {len(genres)} genres: {genres}")
        print(f"\nFinal structure: data/gtzan/genres_original/")
        for genre in genres:
            wav_count = len(list((target_genres / genre).glob("*.wav")))
            print(f"  {genre}: {wav_count} files")
    else:
        print(f"\nWarning: genres_original not found")
        print(f"   Check contents of: {download_path}")
    
    return True

if __name__ == "__main__":
    try:
        success = download_gtzan_kaggle()
        
        if success:
            print("\n" + "="*60)
            print("NEXT STEPS")
            print("="*60)
            print("1. Extract features: python src/data_processing/feature_extractor.py")
            print("2. Train model: python src/training/train_mfcc.py")
            print("3. Run 'uvicorn' src.app.main:app --reload")
            print("4. Open your browser and go to: http://localhost:8000")
    
    except subprocess.CalledProcessError:
        print("\nDownload failed!")
        print("Manual download: https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification")
    except Exception as e:
        print(f"\nError: {e}")
