import sys
import os

# Add root path to sys.path so 'src' can be found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from extract_features import main

# Simulate command line arguments
sys.argv = [
    "extract_features.py",
    "--data-dir", "PATH",
    "--categories", "benign_ram_dumps", "malware_ram_dumps",
    "--sonify", "wav8",
    "--analysis-type", "dynamic",
    "--feature-type", "manual",
    "--workers", "8",

    "--start", "0",
    "--end", "500",

    "--output-dir", "PATH",
]

if __name__ == "__main__":
    main()
