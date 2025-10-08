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

# Queue system for GPU requests
request_queue = asyncio.Queue()
queue_processing = False

@app.on_event("startup")
async def startup_event():
    global client, supabase
    try:
        logger.info("Initializing Gradio client...")
        client = Client("multimodalart/Qwen-Image-Edit-Fast", hf_token=HF_TOKEN)
        logger.info("Gradio client initialized successfully")
        
        logger.info("Initializing Supabase client...")
        supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
        logger.info("Supabase client initialized successfully")
        
        # Start queue processor
        asyncio.create_task(process_queue())
        logger.info("Queue processor started")
    except Exception as e:
        logger.error(f"Failed to initialize clients: {e}")
        raise

async def process_queue():
    """Background task to process queued requests one at a time"""
    global queue_processing
    queue_processing = True
    
    logger.info("Queue processor is running...")
    
    while True:
        try:
            # Get next request from queue
            queue_item = await request_queue.get()
            
            logger.info(f"Processing queued request. Queue size: {request_queue.qsize()}")
            
            # Extract task data
            task_func = queue_item["task"]
            result_future = queue_item["future"]
            
            try:
                # Execute the task
                result = await task_func()
                # Set the result
                result_future.set_result(result)
                logger.info(f"Request completed successfully. Remaining queue: {request_queue.qsize()}")
            except Exception as e:
                # Set the exception
                logger.error(f"Request failed in queue: {type(e).__name__}: {str(e)}")
                result_future.set_exception(e)
            finally:
                # Mark task as done
                request_queue.task_done()
                
        except Exception as e:
            logger.error(f"Error in queue processor: {e}", exc_info=True)
            await asyncio.sleep(1)

async def add_to_queue(task_func):
    """Add a task to the queue and wait for its result"""
    # Create a future to hold the result
    result_future = asyncio.Future()
    
    # Add to queue
    queue_item = {
        "task": task_func,
        "future": result_future
    }
    
    await request_queue.put(queue_item)
    queue_position = request_queue.qsize()
    
    logger.info(f"Request added to queue. Position: {queue_position}")
    
    # Wait for the result
    result = await result_future
    return result

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "client_ready": client is not None,
        "supabase_ready": supabase is not None,
        "queue_size": request_queue.qsize(),
        "queue_processing": queue_processing
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
async def generate_image(
    file: UploadFile = File(...),
    prompt: str = Form(...),
    sender_uid: str = Form(...),
    receiver_uids: str = Form(...)
):
    """Generate image from image and prompt using Qwen-Image-Edit-Fast"""
    
    # Create task function that will be queued
    async def process_task():
        temp_image_path = None
        temp_video_path = None
        
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

            image_url = None
            
            if skip_api:
                # Skip API processing, upload image directly to Supabase
                logger.info("Skipping API processing, uploading image directly to Supabase")
                image_url = await _upload_media_to_supabase(str(temp_image_path), sender_uid, "image")
                logger.info(f"Image uploaded to Supabase: {image_url}")
            else:
                # Process with API
                # Check if client is available
                if client is None:
                    raise HTTPException(status_code=503, detail="AI service not available")

                logger.info("Calling Hugging Face Qwen model...")
                
                # Run the prediction with asyncio timeout
                result = await asyncio.wait_for(
                    asyncio.to_thread(_predict_image, str(temp_image_path), magic_prompt),
                    timeout=300.0  # 5 minutes timeout
                )

                if not result or len(result) < 2:
                    raise HTTPException(status_code=500, detail="Invalid response from AI model")

                local_video_path = result[0].get("path") if isinstance(result[0], dict) else result[0]
                seed_used = result[1] if len(result) > 1 else "unknown"

                logger.info(f"Image generated locally: {local_video_path}")

                # Upload image to Supabase storage
                image_url = await _upload_media_to_supabase(local_video_path, sender_uid, "image")
                
                logger.info(f"Image uploaded to Supabase: {image_url}")

            # Save chat messages to Firebase for each receiver
            receiver_list = [uid.strip() for uid in receiver_uids.split(",") if uid.strip()]
            await _save_chat_messages_to_firebase(sender_uid, receiver_list, image_url, magic_prompt or "", caption, skip_api)

            return {
                "success": True,
                "image_url": image_url,
                "sender_uid": sender_uid,
                "receiver_uids": receiver_list,
                "caption": caption,
                "skipped_api": skip_api
            }

        except asyncio.TimeoutError:
            logger.error("Image generation timed out after 5 minutes")
            raise HTTPException(
                status_code=408, 
                detail="Image generation timed out. Please try with a simpler prompt or smaller image."
            )
        
        except HTTPException:
            # Re-raise HTTP exceptions
            raise
        
        except Exception as e:
            logger.error(f"Error generating video: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate image: {str(e)}"
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
    
    # Add task to queue and wait for result
    try:
        result = await add_to_queue(process_task)
        return JSONResponse(result)
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Queue task failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def _upload_media_to_supabase(local_media_path: str, sender_uid: str, media_type: str = "video") -> str:
    """Upload media (video or image) to Supabase storage and return public URL"""
    try:
        media_path = Path(local_media_path)
        if not media_path.exists():
            raise Exception(f"Media file not found: {local_media_path}")

        # Generate unique filename for Supabase storage
        media_id = str(uuid.uuid4())
        
        if media_type == "image":
            # Determine file extension from the actual file
            file_extension = media_path.suffix.lower()
            if not file_extension or file_extension not in ['.jpg', '.jpeg', '.png', '.webp']:
                file_extension = '.png'  # Default to PNG
            
            # Store directly in videos folder with image extension
            storage_path = f"videos/{sender_uid}/{media_id}{file_extension}"
            
            # Determine content type based on extension
            content_type_map = {
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg', 
                '.png': 'image/png',
                '.webp': 'image/webp'
            }
            content_type = content_type_map.get(file_extension, 'image/png')
        else:
            # Video file
            storage_path = f"videos/{sender_uid}/{media_id}.mp4"
            content_type = "video/mp4"

        # Read media file
        with open(media_path, "rb") as media_file:
            media_data = media_file.read()

        logger.info(f"Uploading {media_type} to Supabase: {storage_path}")

        # Upload to Supabase storage (using same videos bucket)
        try:
            result = supabase.storage.from_("videos").upload(
                path=storage_path,
                file=media_data,
                file_options={
                    "content-type": content_type,
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
        logger.error(f"Failed to upload {media_type} to Supabase: {e}")
        raise Exception(f"Storage upload failed: {str(e)}")

async def _save_chat_messages_to_firebase(sender_uid: str, receiver_list: list, image_url: str, prompt: str, caption: str, is_image_only: bool):
    """Save chat messages with image URL to Firebase for each receiver"""
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
        
        logger.info(f"Saving image messages to Firebase for {len(receiver_list)} receivers")
        
        for receiver_id in receiver_list:
            if not receiver_id:  # Skip empty receiver IDs
                continue
                
            try:
                logger.info(f"Processing message for receiver: {receiver_id}")
                
                # Create message document with all required fields for image message
                message_data = {
                    "senderId": sender_uid,
                    "receiverId": receiver_id,
                    "text": prompt,
                    "videoUrl": image_url,  # Using videoUrl for consistency
                    "messageType": "image",
                    "timestamp": timestamp,
                    "isRead": False,
                    "createdAt": timestamp,
                    "updatedAt": timestamp,
                    "hasVideo": False,
                    "mediaType": "image",
                    "videoStatus": "uploaded"
                }
                
                # Add caption field if caption exists
                if caption:
                    message_data["caption"] = caption
                
                # Save message to chats/{receiver_id}/messages/ collection
                doc_ref = db.collection("chats").document(receiver_id).collection("messages").add(message_data)
                message_id = doc_ref[1].id
                logger.info(f"Image message saved to chats/{receiver_id}/messages/ with ID: {message_id}")
                
                # Also save to sender's chat collection for their own reference
                doc_ref_sender = db.collection("chats").document(sender_uid).collection("messages").add(message_data)
                sender_message_id = doc_ref_sender[1].id
                logger.info(f"Image message saved to chats/{sender_uid}/messages/ with ID: {sender_message_id}")
                
                # Create or update chat document (keeping original chat logic for main chat list)
                # Use consistent chat ID format (smaller UID first)
                chat_participants = sorted([sender_uid, receiver_id])
                chat_id = f"{chat_participants[0]}_{chat_participants[1]}"
                
                # Updated chat data with image-specific fields
                chat_data = {
                    "participants": [sender_uid, receiver_id],
                    "participantIds": chat_participants,
                    "lastMessage": prompt,
                    "lastMessageType": "image",
                    "lastMessageTimestamp": timestamp,
                    "lastSenderId": sender_uid,
                    "lastVideoUrl": image_url,  # Using lastVideoUrl for consistency
                    "lastMediaType": "image",
                    "hasUnreadVideo": False,
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
                    # Update existing chat with image-specific fields
                    update_data = {
                        "lastMessage": prompt,
                        "lastMessageType": "image",
                        "lastMessageTimestamp": timestamp,
                        "lastSenderId": sender_uid,
                        "lastVideoUrl": image_url,  # Using lastVideoUrl for consistency
                        "lastMediaType": "image",
                        "hasUnreadVideo": False,
                        "updatedAt": timestamp,
                        f"unreadCount.{receiver_id}": firestore.Increment(1)
                    }
                    
                    # Add caption to update if it exists
                    if caption:
                        update_data["lastCaption"] = caption
                    
                    chat_ref.update(update_data)
                    logger.info(f"Updated existing chat with image: {chat_id}")
                else:
                    # Create new chat with image data
                    chat_data["createdAt"] = timestamp
                    chat_data["unreadCount"] = {
                        sender_uid: 0,
                        receiver_id: 1
                    }
                    chat_ref.set(chat_data)
                    logger.info(f"Created new chat with image: {chat_id}")
                
            except Exception as e:
                logger.error(f"Failed to save image message for receiver {receiver_id}: {e}")
                continue  # Continue with other receivers even if one fails
        
        logger.info("Successfully saved all image messages with URLs to Firebase")
        
    except Exception as e:
        logger.error(f"Failed to save chat messages to Firebase: {e}", exc_info=True)
        # Don't raise exception here - image generation was successful
        # Just log the error and continue

def _predict_image(image_path: str, prompt: str):
    """Synchronous function to call the Gradio client for image generation"""
    try:
        # Modify the prompt to preserve facial appearance
        enhanced_prompt = f"{prompt} keep the facial appearance same do not change the face"
        
        logger.info(f"Calling Gradio API with enhanced prompt: {enhanced_prompt}")
        
        result = client.predict(
            image=handle_file(image_path),
            prompt=enhanced_prompt,
            seed=0,
            randomize_seed=True,
            true_guidance_scale=2.8,
            num_inference_steps=8,
            rewrite_prompt=True,
            api_name="/infer"
        )
        
        logger.info(f"Gradio API returned successfully")
        return result
        
    except Exception as e:
        logger.error(f"Gradio client prediction failed: {type(e).__name__}: {str(e)}", exc_info=True)
        raise Exception(f"AI model prediction failed: {str(e)}")

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
