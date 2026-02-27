import sys
import os

# Add root path to sys.path so 'src' can be found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from extract_features import main

# Simulate command line arguments
sys.argv = [
    "extract_features.py",
    "--data-dir", "/home/ssanna/nas/RAM/Android",
    "--categories", "benign_dumps", "dumps_dataset",
    "--sonify", "wav8",
    "--analysis-type", "dynamic",
    "--feature-type", "manual",
    "--workers", "8",

    "--start", "500",
    "--end", "1000",

    "--output-dir", "/home/ssanna/Desktop/malware_ram/Android/audio/malware-sonification-signal-processing/",
]

if __name__ == "__main__":
    main()
