import os
import uuid
import shutil
import asyncio
import logging
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from gradio_client import Client, handle_file
from dotenv import load_dotenv
from supabase import create_client, Client as SupabaseClient
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_KEY")

if not HF_TOKEN:
    logger.error("HF_TOKEN not found in environment variables")
    raise ValueError("HF_TOKEN is required")

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    logger.error("SUPABASE_URL or SUPABASE_SERVICE_KEY not found in environment variables")
    raise ValueError("Supabase credentials are required")

app = FastAPI(title="AI Video Generator", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global clients
client = None
audio_client = None
supabase: SupabaseClient = None

@app.on_event("startup")
async def startup_event():
    global client, audio_client, supabase
    try:
        logger.info("Initializing Gradio client...")
        client = Client("Lightricks/ltx-video-distilled", hf_token=HF_TOKEN)
        logger.info("Gradio client initialized successfully")

        logger.info("Initializing Audio Gradio client...")
        audio_client = Client("chenxie95/MeanAudio", hf_token=HF_TOKEN)
        logger.info("Audio Gradio client initialized successfully")
        
        logger.info("Initializing Supabase client...")
        supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
        logger.info("Supabase client initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize clients: {e}")
        raise

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "client_ready": client is not None,
        "audio_client_ready": audio_client is not None,
        "supabase_ready": supabase is not None
    }

def parse_prompt(prompt: str):
    """Parse prompt to extract magic prompt and caption"""
    if "!@#" not in prompt:
        return prompt.strip(), None
    
    parts = prompt.split("!@#", 1)
    magic_prompt = parts[0].strip()
    caption = parts[1].strip() if len(parts) > 1 else None
    
    # Replace ^ with empty string
    magic_prompt = magic_prompt if magic_prompt != "^" else ""
    caption = caption if caption != "^" else None
    
    return magic_prompt, caption

@app.post("/generate/")
async def generate_video(
    file: UploadFile = File(...),
    prompt: str = Form(...),
    sender_uid: str = Form(...),
    receiver_uids: str = Form(...)
):
    """Generate video from image and prompt, add audio, then merge them"""
    temp_image_path = None
    temp_video_path = None
    temp_audio_path = None
    temp_merged_path = None

    try:
        # Parse the prompt to extract magic prompt and caption
        magic_prompt, caption = parse_prompt(prompt)
        
        logger.info(f"Parsed prompt - Magic: '{magic_prompt}', Caption: '{caption}'")
        
        # Determine if we should skip API processing
        skip_api = (magic_prompt == "" or magic_prompt is None)
        
        # Improved image validation
        content_type = file.content_type or ""
        filename = file.filename or ""

        # Check content type OR file extension
        valid_content_types = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp']
        valid_extensions = ['.jpg', '.jpeg', '.png', '.webp']

        is_valid_content_type = any(content_type.startswith(ct) for ct in ['image/jpeg', 'image/jpg', 'image/png', 'image/webp', 'image/'])
        is_valid_extension = any(filename.lower().endswith(ext) for ext in valid_extensions)

        if not (is_valid_content_type or is_valid_extension):
            logger.warning(f"Invalid file - Content-Type: {content_type}, Filename: {filename}")
            raise HTTPException(status_code=400, detail="File must be an image (jpg, png, webp)")

        logger.info(f"Starting processing for user {sender_uid}")
        logger.info(f"Original Prompt: {prompt}")
        logger.info(f"Skip API: {skip_api}")
        logger.info(f"Receivers: {receiver_uids}")
        logger.info(f"File info - Content-Type: {content_type}, Filename: {filename}")

        # Create temp directory if it doesn't exist
        temp_dir = Path("/tmp")
        temp_dir.mkdir(exist_ok=True)

        # Save uploaded image temporarily
        image_id = str(uuid.uuid4())
        file_extension = Path(filename).suffix or '.jpg'
        temp_image_path = temp_dir / f"{image_id}{file_extension}"

        # Save file
        with open(temp_image_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        # Apply EXIF orientation to ensure photos reach API upright
        from PIL import Image, ImageOps
        try:
            with Image.open(temp_image_path) as img:
                # Apply EXIF orientation to correct rotation automatically
                corrected_img = ImageOps.exif_transpose(img)
                if corrected_img is None:
                    # If no EXIF data, use original image
                    corrected_img = img
                corrected_img.save(temp_image_path)
                logger.info(f"Image orientation corrected using EXIF data")
        except Exception as e:
            logger.warning(f"Failed to correct image orientation: {e}, proceeding with original image")

        logger.info(f"Image saved to {temp_image_path}")

        # Validate file size (optional)
        file_size = temp_image_path.stat().st_size
        if file_size > 10 * 1024 * 1024:  # 10MB limit
            raise HTTPException(status_code=400, detail="File too large (max 10MB)")

        # Check if Supabase is available
        if supabase is None:
            raise HTTPException(status_code=503, detail="Storage service not available")

        video_url = None
        
        if skip_api:
            # Skip API processing, upload image directly to Supabase
            logger.info("Skipping API processing, uploading image directly to Supabase")
            video_url = await _upload_image_to_supabase(str(temp_image_path), sender_uid)
            logger.info(f"Image uploaded to Supabase: {video_url}")
        else:
            # Process with API
            # Check if clients are available
            if client is None:
                raise HTTPException(status_code=503, detail="AI video service not available")
            
            if audio_client is None:
                raise HTTPException(status_code=503, detail="AI audio service not available")

            # Start both video and audio generation concurrently
            logger.info("Starting video and audio generation concurrently...")
            
            # Create tasks for both generations
            video_task = asyncio.create_task(
                asyncio.wait_for(
                    asyncio.to_thread(_predict_video, str(temp_image_path), magic_prompt),
                    timeout=300.0  # 5 minutes timeout
                )
            )

            audio_task = asyncio.create_task(
                asyncio.wait_for(
                    asyncio.to_thread(_predict_audio, magic_prompt),
                    timeout=300.0  # 5 minutes timeout
                )
            )

            # Wait for both tasks to complete
            video_result, audio_result = await asyncio.gather(video_task, audio_task)

            if not video_result or len(video_result) < 2:
                raise HTTPException(status_code=500, detail="Invalid response from video AI model")

            if not audio_result:
                raise HTTPException(status_code=500, detail="Invalid response from audio AI model")

            local_video_path = video_result[0].get("video") if isinstance(video_result[0], dict) else video_result[0]
            seed_used = video_result[1] if len(video_result) > 1 else "unknown"
            local_audio_path = audio_result

            logger.info(f"Video generated locally: {local_video_path}")
            logger.info(f"Audio generated locally: {local_audio_path}")

            # Merge video and audio
            merged_video_path = await _merge_video_audio(local_video_path, local_audio_path)
            logger.info(f"Video and audio merged: {merged_video_path}")

            # Upload merged video to Supabase storage
            video_url = await _upload_video_to_supabase(merged_video_path, sender_uid)
            logger.info(f"Merged video uploaded to Supabase: {video_url}")

        # Save chat messages to Firebase for each receiver
        receiver_list = [uid.strip() for uid in receiver_uids.split(",") if uid.strip()]
        await _save_chat_messages_to_firebase(sender_uid, receiver_list, video_url, magic_prompt or "", caption, skip_api)

        return JSONResponse({
            "success": True,
            "video_url": video_url,
            "sender_uid": sender_uid,
            "receiver_uids": receiver_list,
            "caption": caption,
            "skipped_api": skip_api
        })

    except asyncio.TimeoutError:
        logger.error("Video/Audio generation timed out after 5 minutes")
        raise HTTPException(
            status_code=408, 
            detail="Video/Audio generation timed out. Please try with a simpler prompt or smaller image."
        )

    except HTTPException:
        # Re-raise HTTP exceptions
        raise

    except Exception as e:
        logger.error(f"Error generating video: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate video: {str(e)}"
        )

    finally:
        # Cleanup temporary files
        for temp_path in [temp_image_path, temp_video_path, temp_audio_path, temp_merged_path]:
            if temp_path and Path(temp_path).exists():
                try:
                    Path(temp_path).unlink()
                    logger.info(f"Cleaned up temp file: {temp_path}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup temp file {temp_path}: {e}")

async def _upload_image_to_supabase(local_image_path: str, sender_uid: str) -> str:
    """Upload image to Supabase storage and return public URL"""
    try:
        image_path = Path(local_image_path)
        if not image_path.exists():
            raise Exception(f"Image file not found: {local_image_path}")

        # Generate unique filename for Supabase storage
        image_id = str(uuid.uuid4())
        file_extension = image_path.suffix or '.jpg'
        storage_path = f"images/{sender_uid}/{image_id}{file_extension}"

        # Read image file
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()

        logger.info(f"Uploading image to Supabase: {storage_path}")

        # Upload to Supabase storage
        try:
            result = supabase.storage.from_("videos").upload(
                path=storage_path,
                file=image_data,
                file_options={
                    "content-type": f"image/{file_extension.replace('.', '')}",
                    "cache-control": "3600"
                }
            )
            logger.info(f"Upload result: {result}")

        except Exception as upload_error:
            logger.error(f"Upload failed: {upload_error}")
            raise Exception(f"Supabase upload failed: {upload_error}")

        # Get public URL
        try:
            url_result = supabase.storage.from_("videos").get_public_url(storage_path)
            logger.info(f"Generated public URL: {url_result}")

            if not url_result:
                raise Exception("Failed to get public URL")

            return url_result

        except Exception as url_error:
            logger.error(f"Failed to get public URL: {url_error}")
            raise Exception(f"Failed to get public URL: {url_error}")

    except Exception as e:
        logger.error(f"Failed to upload image to Supabase: {e}")
        raise Exception(f"Storage upload failed: {str(e)}")

async def _merge_video_audio(video_path: str, audio_path: str) -> str:
    """Merge video and audio files using ffmpeg"""
    try:
        import subprocess
        
        # Generate output path
        temp_dir = Path("/tmp")
        output_id = str(uuid.uuid4())
        merged_path = temp_dir / f"{output_id}_merged.mp4"
        
        logger.info(f"Merging video {video_path} with audio {audio_path}")
        
        # Use ffmpeg to merge video and audio
        cmd = [
            'ffmpeg', '-y',  # -y to overwrite output file
            '-i', video_path,  # input video
            '-i', audio_path,  # input audio
            '-c:v', 'copy',    # copy video codec (no re-encoding)
            '-c:a', 'aac',     # encode audio to AAC
            '-strict', 'experimental',
            '-shortest',       # finish when shortest stream ends
            str(merged_path)
        ]
        
        # Run ffmpeg command
        result = await asyncio.to_thread(
            subprocess.run, cmd, 
            capture_output=True, 
            text=True, 
            timeout=120  # 2 minute timeout for merging
        )
        
        if result.returncode != 0:
            logger.error(f"FFmpeg failed: {result.stderr}")
            raise Exception(f"Video-audio merging failed: {result.stderr}")
        
        if not merged_path.exists():
            raise Exception("Merged video file was not created")
        
        logger.info(f"Successfully merged video and audio: {merged_path}")
        return str(merged_path)
        
    except Exception as e:
        logger.error(f"Failed to merge video and audio: {e}")
        raise Exception(f"Video-audio merging failed: {str(e)}")

async def _upload_video_to_supabase(local_video_path: str, sender_uid: str) -> str:
    """Upload video to Supabase storage and return public URL"""
    try:
        video_path = Path(local_video_path)
        if not video_path.exists():
            raise Exception(f"Video file not found: {local_video_path}")

        # Generate unique filename for Supabase storage
        video_id = str(uuid.uuid4())
        storage_path = f"videos/{sender_uid}/{video_id}.mp4"

        # Read video file
        with open(video_path, "rb") as video_file:
            video_data = video_file.read()

        logger.info(f"Uploading video to Supabase: {storage_path}")

        # Upload to Supabase storage
        try:
            result = supabase.storage.from_("videos").upload(
                path=storage_path,
                file=video_data,
                file_options={
                    "content-type": "video/mp4",
                    "cache-control": "3600"
                }
            )
            logger.info(f"Upload result: {result}")

        except Exception as upload_error:
            logger.error(f"Upload failed: {upload_error}")
            raise Exception(f"Supabase upload failed: {upload_error}")

        # Get public URL
        try:
            url_result = supabase.storage.from_("videos").get_public_url(storage_path)
            logger.info(f"Generated public URL: {url_result}")

            if not url_result:
                raise Exception("Failed to get public URL")

            return url_result

        except Exception as url_error:
            logger.error(f"Failed to get public URL: {url_error}")
            raise Exception(f"Failed to get public URL: {url_error}")

    except Exception as e:
        logger.error(f"Failed to upload video to Supabase: {e}")
        raise Exception(f"Storage upload failed: {str(e)}")

async def _save_chat_messages_to_firebase(sender_uid: str, receiver_list: list, video_url: str, prompt: str, caption: str, is_image_only: bool):
    """Save chat messages with video URL to Firebase for each receiver"""
    try:
        import firebase_admin
        from firebase_admin import credentials, firestore
        from datetime import datetime
        import pytz

        # Initialize Firebase Admin if not already done
        if not firebase_admin._apps:
            try:
                # Use the specified service account file path
                cred = credentials.Certificate("/etc/secrets/services")
                firebase_admin.initialize_app(cred)
            except Exception as e:
                logger.error(f"Failed to initialize Firebase with service account: {e}")
                raise Exception("Firebase initialization failed")

        db = firestore.client()

        # Current timestamp with timezone
        ist = pytz.timezone('Asia/Kolkata')
        timestamp = datetime.now(ist)

        logger.info(f"Saving messages to Firebase for {len(receiver_list)} receivers")

        for receiver_id in receiver_list:
            if not receiver_id:  # Skip empty receiver IDs
                continue

            try:
                logger.info(f"Processing message for receiver: {receiver_id}")

                # Check if receiver_id ends with "(group)"
                if receiver_id.endswith("(group)"):
                    # Handle group receiver - save to existing group document (remove "(group)" suffix)
                    group_id = receiver_id.replace("(group)", "")  # Remove the "(group)" suffix
                    
                    # Create message document for group
                    group_message_data = {
                        "senderId": sender_uid,
                        "text": prompt,
                        "videoUrl": video_url,
                        "messageType": "image" if is_image_only else "video",
                        "timestamp": timestamp,
                        "isRead": False,
                        "createdAt": timestamp,
                        "updatedAt": timestamp,
                        "hasVideo": not is_image_only,
                        "mediaType": "image" if is_image_only else "video",
                        "videoStatus": "uploaded"
                    }
                    
                    # Add caption field if caption exists
                    if caption:
                        group_message_data["caption"] = caption

                    # Save to groups/{group_id}/messages/ (without "(group)" in the path)
                    doc_ref = db.collection("groups").document(group_id).collection("messages").add(group_message_data)
                    message_id = doc_ref[1].id
                    logger.info(f"Message saved to groups/{group_id}/messages/ with ID: {message_id}")

                else:
                    # Handle regular individual receiver (existing logic)
                    message_data = {
                        "senderId": sender_uid,
                        "receiverId": receiver_id,
                        "text": prompt,
                        "videoUrl": video_url,
                        "messageType": "image" if is_image_only else "video",
                        "timestamp": timestamp,
                        "isRead": False,
                        "createdAt": timestamp,
                        "updatedAt": timestamp,
                        "hasVideo": not is_image_only,
                        "mediaType": "image" if is_image_only else "video",
                        "videoStatus": "uploaded"
                    }
                    
                    # Add caption field if caption exists
                    if caption:
                        message_data["caption"] = caption

                    # Save message to chats/{receiver_id}/messages/ collection
                    doc_ref = db.collection("chats").document(receiver_id).collection("messages").add(message_data)
                    message_id = doc_ref[1].id
                    logger.info(f"Message saved to chats/{receiver_id}/messages/ with ID: {message_id}")

                    # Also save to sender's chat collection for their own reference
                    doc_ref_sender = db.collection("chats").document(sender_uid).collection("messages").add(message_data)
                    sender_message_id = doc_ref_sender[1].id
                    logger.info(f"Message saved to chats/{sender_uid}/messages/ with ID: {sender_message_id}")

                    # Create or update chat document (keeping original chat logic for main chat list)
                    chat_participants = sorted([sender_uid, receiver_id])
                    chat_id = f"{chat_participants[0]}_{chat_participants[1]}"

                    chat_data = {
                        "participants": [sender_uid, receiver_id],
                        "participantIds": chat_participants,
                        "lastMessage": prompt,
                        "lastMessageType": "image" if is_image_only else "video",
                        "lastMessageTimestamp": timestamp,
                        "lastSenderId": sender_uid,
                        "lastVideoUrl": video_url,
                        "lastMediaType": "image" if is_image_only else "video",
                        "hasUnreadVideo": not is_image_only,
                        "updatedAt": timestamp,
                        "unreadCount": {
                            receiver_id: firestore.Increment(1)
                        }
                    }
                    
                    # Add caption to chat data if it exists
                    if caption:
                        chat_data["lastCaption"] = caption

                    # Create chat if it doesn't exist, or update if it does
                    chat_ref = db.collection("chats").document(chat_id)

                    # Check if chat exists
                    chat_doc = chat_ref.get()
                    if chat_doc.exists:
                        # Update existing chat
                        update_data = {
                            "lastMessage": prompt,
                            "lastMessageType": "image" if is_image_only else "video",
                            "lastMessageTimestamp": timestamp,
                            "lastSenderId": sender_uid,
                            "lastVideoUrl": video_url,
                            "lastMediaType": "image" if is_image_only else "video",
                            "hasUnreadVideo": not is_image_only,
                            "updatedAt": timestamp,
                            f"unreadCount.{receiver_id}": firestore.Increment(1)
                        }
                        
                        # Add caption to update if it exists
                        if caption:
                            update_data["lastCaption"] = caption
                        
                        chat_ref.update(update_data)
                        logger.info(f"Updated existing chat: {chat_id}")
                    else:
                        # Create new chat
                        chat_data["createdAt"] = timestamp
                        chat_data["unreadCount"] = {
                            sender_uid: 0,
                            receiver_id: 1
                        }
                        chat_ref.set(chat_data)
                        logger.info(f"Created new chat: {chat_id}")

            except Exception as e:
                logger.error(f"Failed to save message for receiver {receiver_id}: {e}")
                continue  # Continue with other receivers even if one fails

        logger.info("Successfully saved all messages to Firebase")

    except Exception as e:
        logger.error(f"Failed to save chat messages to Firebase: {e}", exc_info=True)
        # Don't raise exception here - video generation was successful
        # Just log the error and continue

def _predict_video(image_path: str, prompt: str):
    """Synchronous function to call the Gradio client"""
    try:
        return client.predict(
            prompt=prompt,
            negative_prompt="worst quality, inconsistent motion, blurry, artifacts",
            input_image_filepath=handle_file(image_path),
            input_video_filepath=None,
            height_ui=960,
            width_ui=544,
            mode="image-to-video",
            duration_ui=2,
            ui_frames_to_use=9,
            seed_ui=42,
            randomize_seed=True,
            ui_guidance_scale=5,
            improve_texture_flag=True,
            api_name="/image_to_video"
        )
    except Exception as e:
        logger.error(f"Gradio client prediction failed: {e}")
        raise

def _predict_audio(prompt: str):
    """Synchronous function to call the Audio Gradio client"""
    try:
        logger.info(f"Generating audio with prompt: {prompt}")
        
        result = audio_client.predict(
            prompt=prompt,
            duration=5,
            cfg_strength=4.5,
            num_steps=1,
            variant="meanaudio_s_full",
            seed=42,
            api_name="/predict"
        )
        
        logger.info(f"Audio generation result: {result}")
        return result[0]  # Return the first element (filepath) from the tuple
        
    except Exception as e:
        logger.error(f"Audio Gradio client prediction failed: {e}")
        raise

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Global exception handler caught: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )

if __name__ == "__main__":
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        timeout_keep_alive=300,  # 5 minutes keep alive
        timeout_graceful_shutdown=30
    )
