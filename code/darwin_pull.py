from darwin.client import Client
import darwin

API_KEY = "	mNcEh_J.XHgOKlY-XvbmfR6QuRiEfi3H3gL1fTTh"
client = Client.from_api_key(API_KEY)

try:
    dataset = client.get_remote_dataset("rockfishvision/people-tracking-videos")
except darwin.exceptions.NotFound:
    print(f"Dataset people-tracking-videos not found")

release_name = "tracking-1-17-2023"
try:
    release = dataset.get_release(release_name)
except darwin.exceptions.NotFound:
    print("release not found")

print("Pulling release now")
dataset.pull(release=release)