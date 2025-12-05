import requests
import sys
from pathlib import Path

def test_predict(audio_path):
    url = "http://localhost:8000/predict"
    
    if not Path(audio_path).exists():
        print(f"Error: File {audio_path} not found")
        return

    print(f"Testing with file: {audio_path}")
    
    with open(audio_path, "rb") as f:
        files = {"file": f}
        try:
            response = requests.post(url, files=files)
            
            if response.status_code == 200:
                print("Success!")
                print(response.json())
            else:
                print(f"Failed with status code {response.status_code}")
                print(response.text)
        except Exception as e:
            print(f"Connection error: {e}")
            print("Is the server running?")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
    else:
        print("Usage: python test_api.py <path_to_audio_file>")
        sys.exit(1)
        
    test_predict(audio_path)
