from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
import modal
from modal import Image, App, Volume, Secret, enter, method
from pydantic import BaseModel
import requests
import os
import uuid
import pathlib
import boto3


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

volume = Volume.from_name(
    "ai-video-clipper-cache", create_if_missing=True
    )


mount_path = "/root/.cache/torch"

auth_scheme = HTTPBearer()

@app.cls(
    gpu="L40S",
    timeout=900,
    retries=0,
    scaledown_window=20,
    secrets=[Secret.from_name("ai-video-clipper-secret")],
    volumes={mount_path: volume}
)
class AiVideoClipper:
    @modal.enter() # Now 'enter' is defined
    def load_model(self):
        print("Loading models")
        
        pass
    
    def transcribe_video(self, base_dir:str, video_path:str) -> str:
        pass


    @modal.fastapi_endpoint(method="POST")
    def process_video(self, request: ProcessVideoRequest, token: HTTPAuthorizationCredentials = Depends(auth_scheme)):
        s3_key = request.s3_key

        if token.credentials != os.environ["AUTH_TOKEN"]:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                                detail="Incorrect bearer token", headers={"WWW-Authenticate": "Bearer"})

        
        run_id = str(uuid.uuid4())
        base_dir = pathlib.Path("/tmp") / run_id
        base_dir.mkdir(parents=True, exist_ok=True)

        # Download the video from S3
        video_path = base_dir / "input.mp4"
        s3_client = boto3.client("s3")
        s3_client.download_file("ai-video-clippers", s3_key, str(video_path))
        print(os.listdir(base_dir))

        
        return {"message": f"Video {request.s3_key} processed successfully!"}


@app.local_entrypoint()
def main():

    ai_video_clipper = AiVideoClipper()
    base_url = AiVideoClipper.process_video.web_url
    if not base_url:
        print("Web endpoint URL not found. Ensure the app is deployed or running locally.")
        return

   
    payload = {
        "s3_key": "test1/rg1_30min.mp4"
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer 123123"
    }

    response = requests.post(base_url, json=payload,
                             headers=headers)
    response.raise_for_status()
    result = response.json()
    print(result)