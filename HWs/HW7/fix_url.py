import os
import sys
import re
from unittest.mock import patch

# Monkey patch for huggingface_hub to fix backslash issues in Windows paths
import huggingface_hub

# Store original implementation
original_hf_hub_download = huggingface_hub.hf_hub_download

def patched_hf_hub_download(repo_id, filename, **kwargs):
    # Replace any backslashes in the filename with forward slashes
    fixed_filename = filename.replace("\\", "/")
    # Call the original function with the fixed filename
    return original_hf_hub_download(repo_id, fixed_filename, **kwargs)

# Apply the patch to the huggingface_hub module
huggingface_hub.file_download.hf_hub_download = patched_hf_hub_download
huggingface_hub.hf_hub_download = patched_hf_hub_download

# Import and run the original script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hw7

# Run the main function
if __name__ == "__main__":
    hw7.main(visualize=True) 