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
supabase: SupabaseClient = None

@app.on_event("startup")
async def startup_event():
    global client, supabase
    try:
        logger.info("Initializing Gradio client...")
        client = Client("mcp-tools/FLUX.1-Kontext-Dev", hf_token=HF_TOKEN)
        logger.info("Gradio client initialized successfully")
        
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
        "supabase_ready": supabase is not None
    }

@app.post("/generate/")
async def generate_video(
    file: UploadFile = File(...),
    prompt: str = Form(...),
    sender_uid: str = Form(...),
    receiver_uids: str = Form(...)
):
    """Generate video from image and prompt"""
    temp_image_path = None
    temp_video_path = None
    
    try:
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
        
        if len(prompt.strip()) == 0:
            raise HTTPException(status_code=400, detail="Prompt cannot be empty")

        logger.info(f"Starting video generation for user {sender_uid}")
        logger.info(f"Prompt: {prompt}")
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

        # Rotate image 90 degrees clockwise to fix phone orientation
        from PIL import Image
        try:
            with Image.open(temp_image_path) as img:
                # Rotate 90 degrees clockwise (270 degrees counter-clockwise)
                rotated_img = img.rotate(-90, expand=True)
                rotated_img.save(temp_image_path)
                logger.info(f"Image rotated 90 degrees clockwise")
        except Exception as e:
            logger.warning(f"Failed to rotate image: {e}, proceeding with original image")

        logger.info(f"Image saved to {temp_image_path}")

        # Validate file size (optional)
        file_size = temp_image_path.stat().st_size
        if file_size > 10 * 1024 * 1024:  # 10MB limit
            raise HTTPException(status_code=400, detail="File too large (max 10MB)")

        # Check if clients are available
        if client is None:
            raise HTTPException(status_code=503, detail="AI service not available")
        
        if supabase is None:
            raise HTTPException(status_code=503, detail="Storage service not available")

        # Call HF model with timeout
        logger.info("Calling Hugging Face model...")
        
        # Run the prediction with asyncio timeout
        result = await asyncio.wait_for(
            asyncio.to_thread(_predict_video, str(temp_image_path), prompt),
            timeout=300.0  # 5 minutes timeout
        )

        if not result or len(result) < 2:
            raise HTTPException(status_code=500, detail="Invalid response from AI model")

        local_video_path = result[0].get("path") if isinstance(result[0], dict) else result[0]
        seed_used = result[1] if len(result) > 1 else "unknown"

        logger.info(f"Video generated locally: {local_video_path}")

        # Upload video to Supabase storage
        video_url = await _upload_video_to_supabase(local_video_path, sender_uid)
        
        logger.info(f"Video uploaded to Supabase: {video_url}")

        # Save chat messages to Firebase for each receiver
        receiver_list = [uid.strip() for uid in receiver_uids.split(",") if uid.strip()]
        await _save_chat_messages_to_firebase(sender_uid, receiver_list, video_url, prompt)

        return JSONResponse({
            "success": True,
            "video_url": video_url,
            "seed": seed_used,
            "sender_uid": sender_uid,
            "receiver_uids": receiver_list
        })

    except asyncio.TimeoutError:
        logger.error("Video generation timed out after 5 minutes")
        raise HTTPException(
            status_code=408, 
            detail="Video generation timed out. Please try with a simpler prompt or smaller image."
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
        for temp_path in [temp_image_path, temp_video_path]:
            if temp_path and Path(temp_path).exists():
                try:
                    Path(temp_path).unlink()
                    logger.info(f"Cleaned up temp file: {temp_path}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup temp file {temp_path}: {e}")

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

async def _save_chat_messages_to_firebase(sender_uid: str, receiver_list: list, video_url: str, prompt: str):
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
        
        logger.info(f"Saving video messages to Firebase for {len(receiver_list)} receivers")
        
        for receiver_id in receiver_list:
            if not receiver_id:  # Skip empty receiver IDs
                continue
                
            try:
                logger.info(f"Processing message for receiver: {receiver_id}")
                
                # Create message document with all required fields for video message
                message_data = {
                    "senderId": sender_uid,
                    "receiverId": receiver_id,
                    "text": prompt,  # The prompt as message text
                    "videoUrl": video_url,  # Supabase video URL - THIS IS KEY FOR VIDEO DISPLAY
                    "messageType": "video",  # Message type - IMPORTANT for app to know it's a video
                    "timestamp": timestamp,
                    "isRead": False,
                    "createdAt": timestamp,
                    "updatedAt": timestamp,
                    # Additional fields for better video handling
                    "hasVideo": True,  # Flag to easily identify video messages
                    "mediaType": "video",  # Explicit media type
                    "videoStatus": "uploaded"  # Status of video (uploaded, processing, failed)
                }
                
                # Save message to chats/{receiver_id}/messages/ collection
                doc_ref = db.collection("chats").document(receiver_id).collection("messages").add(message_data)
                message_id = doc_ref[1].id
                logger.info(f"Video message saved to chats/{receiver_id}/messages/ with ID: {message_id}")
                
                # Also save to sender's chat collection for their own reference
                doc_ref_sender = db.collection("chats").document(sender_uid).collection("messages").add(message_data)
                sender_message_id = doc_ref_sender[1].id
                logger.info(f"Video message saved to chats/{sender_uid}/messages/ with ID: {sender_message_id}")
                
                # Create or update chat document (keeping original chat logic for main chat list)
                # Use consistent chat ID format (smaller UID first)
                chat_participants = sorted([sender_uid, receiver_id])
                chat_id = f"{chat_participants[0]}_{chat_participants[1]}"
                
                # Updated chat data with video-specific fields
                chat_data = {
                    "participants": [sender_uid, receiver_id],
                    "participantIds": chat_participants,  # For easier querying
                    "lastMessage": prompt,  # Show prompt as last message preview
                    "lastMessageType": "video",  # IMPORTANT: Tells app last message was video
                    "lastMessageTimestamp": timestamp,
                    "lastSenderId": sender_uid,
                    "lastVideoUrl": video_url,  # Store last video URL for quick access
                    "lastMediaType": "video",  # Explicit media type for last message
                    "hasUnreadVideo": True,  # Flag for unread video content
                    "updatedAt": timestamp,
                    "unreadCount": {
                        receiver_id: firestore.Increment(1)  # Increment unread count for receiver
                    }
                }
                
                # Create chat if it doesn't exist, or update if it does
                chat_ref = db.collection("chats").document(chat_id)
                
                # Check if chat exists
                chat_doc = chat_ref.get()
                if chat_doc.exists:
                    # Update existing chat with video-specific fields
                    update_data = {
                        "lastMessage": prompt,
                        "lastMessageType": "video",  # Key for app to show video icon/preview
                        "lastMessageTimestamp": timestamp,
                        "lastSenderId": sender_uid,
                        "lastVideoUrl": video_url,  # URL for video preview/thumbnail
                        "lastMediaType": "video",
                        "hasUnreadVideo": True,  # Important for showing video notification
                        "updatedAt": timestamp,
                        f"unreadCount.{receiver_id}": firestore.Increment(1)
                    }
                    chat_ref.update(update_data)
                    logger.info(f"Updated existing chat with video: {chat_id}")
                else:
                    # Create new chat with video data
                    chat_data["createdAt"] = timestamp
                    chat_data["unreadCount"] = {
                        sender_uid: 0,
                        receiver_id: 1
                    }
                    chat_ref.set(chat_data)
                    logger.info(f"Created new chat with video: {chat_id}")
                
            except Exception as e:
                logger.error(f"Failed to save video message for receiver {receiver_id}: {e}")
                continue  # Continue with other receivers even if one fails
        
        logger.info("Successfully saved all video messages with URLs to Firebase")
        
    except Exception as e:
        logger.error(f"Failed to save chat messages to Firebase: {e}", exc_info=True)
        # Don't raise exception here - video generation was successful
        # Just log the error and continue

def _predict_video(image_path: str, prompt: str):
    """Synchronous function to call the Gradio client"""
    try:
        return client.predict(
            input_image=handle_file(image_path),
            prompt=prompt,
            seed=0,
            randomize_seed=True,
            guidance_scale=2.5,
            steps=20,
            api_name="/infer"
        )
    except Exception as e:
        logger.error(f"Gradio client prediction failed: {e}")
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
