import glob
import json
import pickle
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
import whisperx
import time
import subprocess
import google.generativeai as genai
import shutil
# This Modal app processes videos by downloading them from S3, transcribing them with WhisperX, and returning the transcript.

class ProcessVideoRequest(BaseModel):
    # Defines the expected structure for incoming video processing requests.
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

def create_vertical_video(tracks,scores,pyframes_path,pyavi_path,audio_path,output_path,frame_rate=30):
    target_width = 1080
    target_height = 1920

    flist = glob.glob(os.path.join(pyframes_path,"*.jpg"))
    flist.sort()

    if not flist:
        print(f"No frames found in {pyframes_path}")
        return


def process_clip(base_dir:pathlib.Path, video_path:pathlib.Path, s3_key:str, start_time:float, end_time:float, index:int, transcript_segments:list[dict]):
    clip_name = f"clip_{index}.mp4"
    s3_key_dir = os.path.dirname(s3_key)
    output_s3_key = f"{s3_key_dir}/clips/{clip_name}"
    print(f"Output S3 key: {output_s3_key}")

    clip_dir = base_dir / clip_name
    clip_dir.mkdir(parents=True, exist_ok=True)

    # segment path : Original clip from start to end
    clip_segment_paht = clip_dir / f"{clip_name}_segment.mp4"

    vertical_mp4_path = clip_dir /"pyavi"/ "video_out_vertical.mp4"

    subtitle_output_path = clip_dir / "pyavi" / "video_with_subtitles.mp4"

    (clip_dir / "pywork").mkdir(exist_ok=True)
    pyframes_path = clip_dir/"pyframes"
    pyavi_path = clip_dir/"pyavi"
    audio_path = clip_dir/"pyavi"/"audio.wav"


    pyframes_path.mkdir(exist_ok=True)
    pyavi_path.mkdir(exist_ok=True)


    # Now cutting the clip from the video
    duration = end_time - start_time
    cut_command = (f"ffmpeg -i {video_path} -ss {start_time} -t {duration} "
                f"({clip_segment_paht})") # This is a hack to get the clip segment path
    subprocess.run(cut_command, shell=True, check=True, capture_output=True)

    # Now extracting the audio from the clip
    audio_command = (f"ffmpeg -i {clip_segment_paht} -vn -acodec pcm_s16le -ar 16000 -ac 1 {audio_path}")
    subprocess.run(audio_command, shell=True, check=True, capture_output=True)

    shutil.copy(clip_segment_paht, base_dir /f"{clip_name}.mp4")

    columbia_command = (f"python Columbia_test.py --video_name {clip_name} "
                        
                        f"--VideoFolder {str(base_dir)}"
                        f"--pretrained_model weight/finetuning_TalkSet.model"
                        )

    columbia_start_time = time.time()
    subprocess.run(columbia_command, cwd="/asd", shell=True)
    columbia_end_time = time.time()
    print(f"Columbia processing time: {columbia_end_time - columbia_start_time} seconds")

    track_path = clip_dir / "pywork" / "track.pckl"
    scores_path = clip_dir / "pywork" / "scores.pckl"
    if not track_path.exists() or not scores_path.exists():
        print(f"Columbia failed to generate track or scores for {clip_name}")
        return

    with open(track_path, "rb") as f:
        track = pickle.load(f)

    with open(scores_path, "rb") as f:
        scores = pickle.load(f)
    
    create_vertical_video_time = time.time()
    create_vertical_video(track,scores,pyframes_path,pyavi_path,audio_path,vertical_mp4_path)
    create_vertical_video_end_time = time.time()
    print(f"Vertical video creation time: {create_vertical_video_end_time - create_vertical_video_time} seconds")

    





    
@app.cls(
    gpu="L40S",
    timeout=900,
    retries=0,
    scaledown_window=20,
    secrets=[Secret.from_name("ai-video-clipper-secret")],
    volumes={mount_path: volume}
)
class AiVideoClipper:
    # This class handles the AI-powered video clipping and transcription.
    @modal.enter() # Now 'enter' is defined
    def load_model(self):
        # Loads the WhisperX transcription and alignment models into memory.
        print("Loading models")
        self.whisperx_model = whisperx.load_model("large-v2", device="cuda", compute_type="float16")

        self.alignment_model, self.metadata = whisperx.load_align_model(

            language_code="en", 
            device="cuda"
        )


        print("Transcription models loaded.... ")



        print("Loading gemini client")
        self.gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        print("Gemini client loaded")

    
        
    
    def transcribe_video(self, base_dir:str, video_path:str) -> str:
        # Transcribes the audio from a given video file and aligns the words.
        audio_path = base_dir / "audio.wav"
        extract_cmd = f"ffmpeg -i {video_path} -vn -acodec pcm_s16le -ar 16000 -ac 1 {audio_path}"
        subprocess.run(extract_cmd, shell=True, check=True ,capture_output=True)
        print("Starting transcription....")
        start_time = time.time()

        audio = whisperx.load_audio(str(audio_path))
        result = self.whisperx_model.transcribe(audio, batch_size=16)

        result = whisperx.align(
        result["segments"], 
        self.alignment_model, 
        self.metadata, 
        audio, 
        device="cuda",
        return_char_alignments=False
        )

        duration = time.time() - start_time
        print("Transcription and alignment completed in {:.2f} seconds".format(duration))

        segments = []

        if "word_segments" in result:
            for word_segment in result["word_segments"]:
                segments.append({
                    "start": word_segment["start"],
                    "end": word_segment["end"],
                    "word": word_segment["word"]
                })

        return json.dumps(segments, indent=4)

    
    def identify_moments(self, transcript_segments:list[dict]):
        response = self.gemini_client.models.generate_content(model="Gemini-2.5-Flash-Preview-05-20", content = f"""
    **Objective:** Extract compelling, self-contained story or Q&A segments from a podcast video transcript.

    **Input:** A JSON-formatted transcript, where each entry is a sentence with its 'start' and 'end' timestamps.

    **Task:**
    Identify and extract clips that represent:
    1.  Complete stories.
    2.  A question and its direct answer.

    **Clip Guidelines:**
    * **Contextualization:** For Q&A, you may include a few preceding sentences (up to 3) if they significantly enhance understanding of the question.
    * **Boundary Adherence:** Each clip MUST begin and end precisely at existing sentence boundaries.
    * **Timestamp Integrity:** Use ONLY the provided 'start' and 'end' timestamps from the transcript. DO NOT modify them.
    * **Length:** Prioritize clips between 40-60 seconds in duration.
    * **Completeness:** Maximize inclusion of relevant contextual content within the clip.

    **Exclusions:**
    * Greetings, thank-yous, or goodbyes.
    * Non-question/answer conversational filler (e.g., "uhm," "you know," brief affirmations unless critical to a story).

    **Output Format:**
    A JSON list of objects, each representing an extracted clip. Each clip object MUST have:
    * `"start"`: The start timestamp in seconds (float).
    * `"end"`: The end timestamp in seconds (float).

    Example: `[{"start": 12.34, "end": 67.89}, {"start": 105.00, "end": 150.25}]`

    **Empty Output:**
    If no valid clips can be extracted based on the above criteria, return an empty JSON list: `[]`.

    The transcript is as follows:
   
    """ + str(transcript_segments))
        print(f"Identified moments: {response.text}")
        return response.text

    

    @modal.fastapi_endpoint(method="POST")
    def process_video(self, request: ProcessVideoRequest, token: HTTPAuthorizationCredentials = Depends(auth_scheme)):
        # FastAPI endpoint to process video: downloads from S3, transcribes, and returns the transcript.
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
        
        # Transcribe the video
        transcript_segment_json = self.transcribe_video(base_dir, video_path)
        transcript_segments = json.loads(transcript_segment_json)

        # Identify moments
        print("Identifying moments")
        identified_moments_raw = self.identify_moments(transcript_segments)
        cleaned_json_string = identified_moments_raw.strip()
        if cleaned_json_string.startswith("```json"):
            cleaned_json_string = cleaned_json_string[len("```json"):].strip()
        if cleaned_json_string.endswith("```"):
            cleaned_json_string = cleaned_json_string[:-len("```")].strip()
        

        clip_moments = json.loads(cleaned_json_string)
        if not clip_moments or not isinstance(clip_moments, list):
            print("Invalid JSON format for clip moments")
            clip_moments = []


        print(clip_moments)

        #3. Process the clips
        for index,moment in enumerate(clip_moments[:3]):
            if "start" in moment and "end" in moment:
                print("Processing clip" + str(index) + "from" + str(moment["start"]) + "to" + str(moment["end"]) )
                process_clip(base_dir, video_path,s3_key, moment["start"], moment["end"], index, transcript_segments)

        

        if(base_dir.exists()):
            print("Cleaning up temporary directory" + str(base_dir))
            shutil.rmtree(base_dir)
        return {"message": "Video processed successfully"}

  




        


@app.local_entrypoint()
def main():
    # Local entrypoint for testing the video processing functionality.

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