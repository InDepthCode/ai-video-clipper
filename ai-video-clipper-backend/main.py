from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
import modal
from modal import Image, App, Volume, Secret, enter, method # THIS LINE IS CRUCIAL AND WAS MISSING/COMMENTED
from pydantic import BaseModel
import requests
import os


class ProcessVideoRequest(BaseModel):
    s3_key: str


image = (
    Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.12") # Now 'Image' is defined
    .apt_install(["ffmpeg", "libgl1-mesa-glx", "wget", "libcudnn8", "libcudnn8-dev"])
    .pip_install_from_requirements("requirements.txt")
    .run_commands([
        "mkdir -p /usr/share/fonts/truetype/custom",
        "wget -O /usr/share/fonts/truetype/custom/Anton.ttf https://github.com/google/fonts/raw/main/ofl/anton/Anton-Regular.ttf",
        "fc-cache -f -v"
    ])
    .add_local_dir("asd", "/asd", copy=True)
)


app = App("ai-video-clipper", image=image)

volume = Volume.from_name("ai-video-clipper-cache", create_if_missing=True) # Now 'Volume' is defined
mount_path = "/root/.cache/torch"

auth_scheme = HTTPBearer()

@app.cls(
    gpu="L40S",
    timeout=900,
    retries=0,
    scaledown_window=20,
    secrets=[Secret.from_name("ai-video-clipper-secret")], # Now 'Secret' is defined
    volumes={mount_path: volume}
)
class AiVideoClipper:
    @modal.enter() # Now 'enter' is defined
    def load_model(self):
        print("Loading models")
        pass

    @modal.fastapi_endpoint(method="POST") # Now 'method' is defined
    def process_video(self, request: ProcessVideoRequest, token: HTTPAuthorizationCredentials = Depends(auth_scheme)):
        print(f"Processing video for s3_key: {request.s3_key}")
        
        return {"message": f"Video {request.s3_key} processed successfully!"}


@app.local_entrypoint()
def main():
    base_url = AiVideoClipper.process_video.web_url
    if not base_url:
        print("Web endpoint URL not found. Ensure the app is deployed or running locally.")
        return

    local_api_key = os.environ.get("LOCAL_TEST_API_KEY", "your-fallback-test-api-key")
    if local_api_key == "your-fallback-test-api_key":
        print("Warning: LOCAL_TEST_API_KEY environment variable not set. Using fallback API key.")

    payload = {
        "s3_key": "test1/rg1.mp4"
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {local_api_key}"
    }

    print(f"Sending request to: {base_url}")
    print(f"Payload: {payload}")

    try:
        response = requests.post(base_url, json=payload, headers=headers)
        response.raise_for_status()
        print("Response:", response.json())
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error: {e}")
        print(f"Response content: {e.response.text}")
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")