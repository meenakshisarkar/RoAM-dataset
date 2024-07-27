from huggingface_hub import snapshot_download
from os import system
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))

snapshot_download(repo_id="meenakshi-roam/testing", repo_type="dataset", local_dir=dir_path)

