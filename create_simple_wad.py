import os
import urllib.request
import sys

def download_basic_wad():
    """
    Download the basic.wad file directly using urllib
    This is a simpler method that should work on most systems
    """
    wad_url = "https://github.com/mwydmuch/ViZDoom/raw/master/scenarios/basic.wad"
    
    print(f"Downloading basic.wad from {wad_url}...")
    try:
        urllib.request.urlretrieve(wad_url, "basic.wad")
        if os.path.exists("basic.wad"):
            print("Successfully downloaded basic.wad!")
            return True
        else:
            print("Download completed but file not found!")
            return False
    except Exception as e:
        print(f"Error downloading basic.wad: {e}")
        return False

if __name__ == "__main__":
    if os.path.exists("basic.wad"):
        print("basic.wad already exists in the current directory.")
    else:
        success = download_basic_wad()
        
        if not success:
            print("\nFailed to download basic.wad!")
            print("Please download it manually from:")
            print("https://github.com/mwydmuch/ViZDoom/raw/master/scenarios/basic.wad")
            print("Save it as 'basic.wad' in the same directory as your Python scripts.")
            sys.exit(1)