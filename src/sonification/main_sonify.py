import sys
import os

# Add root path to sys.path so 'src' can be found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.sonification.sonification import main

# Simulate command line arguments
sys.argv = [
    "sonification.py",
    "--data-dir", "/home/ssanna/nas/RAM/Android",
    "--categories", "benign_dumps", "dumps_dataset",
    "--sonify", "wav8",
    "--dataset-type", "datasetRAM",
    #"--memory-regions", "all",
    "--workers", "8",
    "--start", "500",
    "--end", "1000",
]

if __name__ == "__main__":
    main()
