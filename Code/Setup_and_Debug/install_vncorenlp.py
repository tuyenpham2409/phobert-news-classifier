import os
import urllib.request
import ssl

# Ignore SSL certificate errors
ssl._create_default_https_context = ssl._create_unverified_context

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Point to Web_Application/vncorenlp
VNCORENLP_DIR = os.path.join(BASE_DIR, "..", "Web_Application", "vncorenlp")
MODELS_DIR = os.path.join(VNCORENLP_DIR, "models", "wordsegmenter")

os.makedirs(MODELS_DIR, exist_ok=True)

files = {
    "VnCoreNLP-1.1.1.jar": "https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/VnCoreNLP-1.1.1.jar",
    "models/wordsegmenter/vi-vocab": "https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/vi-vocab",
    "models/wordsegmenter/wordsegmenter.rdr": "https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/wordsegmenter.rdr"
}

print(f"Downloading VnCoreNLP to {VNCORENLP_DIR}...")

for file_path, url in files.items():
    dest_path = os.path.join(VNCORENLP_DIR, file_path)
    print(f"Downloading {file_path}...")
    try:
        urllib.request.urlretrieve(url, dest_path)
        print(f"Downloaded: {dest_path}")
    except Exception as e:
        print(f"Error downloading {file_path}: {e}")

print("\nSetup complete!")
