from fastapi import FastAPI, Request, Form, Response, File, UploadFile
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List
import os
import aiosqlite
import uuid
import base64
from groq import Groq
import google.generativeai as genai
from starlette.middleware.sessions import SessionMiddleware
from fastapi.responses import FileResponse
import speech_recognition as sr
from gtts import gTTS
from pydub import AudioSegment

app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key="your-secret-key")  # Session middleware
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")  # Static folder for logo

# Environment variables for API keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyDhIdC3HPpbrqvcQs_4QoBVBNGGCGvRzIc")

# Initialize Groq and Gemini clients
groq_client = Groq(api_key=GROQ_API_KEY)
genai.configure(api_key=GEMINI_API_KEY)

# Pydantic model for chat messages
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str

# Predefined tourism links
TOURISM_LINKS = {
    "general": "https://www.visitdubai.com/en",
    "attractions": "https://www.visitdubai.com/en/places-to-visit",
    "tours": "https://www.visitdubai.com/en/things-to-do/tours-and-attractions",
    "hotels": "https://www.visitdubai.com/en/places-to-stay",
}

# SQLite database initialization
async def init_db():
    async with aiosqlite.connect("chat_history.db") as db:
        await db.execute("CREATE TABLE IF NOT EXISTS history (session_id TEXT, role TEXT, content TEXT)")
        await db.execute("CREATE TABLE IF NOT EXISTS feedback (session_id TEXT, message_index INTEGER, rating TEXT)")
        await db.commit()

# Save message to SQLite
async def save_message(session_id: str, role: str, content: str):
    async with aiosqlite.connect("chat_history.db") as db:
        await db.execute("INSERT INTO history (session_id, role, content) VALUES (?, ?, ?)", (session_id, role, content))
        await db.commit()

# Get chat history from SQLite
async def get_history(session_id: str) -> List[Message]:
    async with aiosqlite.connect("chat_history.db") as db:
        cursor = await db.execute("SELECT role, content FROM history WHERE session_id = ?", (session_id,))
        rows = await cursor.fetchall()
        return [Message(role=row[0], content=row[1]) for row in rows]

# Save feedback to SQLite
async def save_feedback(session_id: str, message_index: int, rating: str):
    async with aiosqlite.connect("chat_history.db") as db:
        await db.execute("INSERT INTO feedback (session_id, message_index, rating) VALUES (?, ?, ?)", (session_id, message_index, rating))
        await db.commit()

# Intent detection function
def detect_intent(message: str) -> str:
    message = message.lower()
    if any(keyword in message for keyword in ["attraction", "place to visit", "things to see"]):
        return "attractions"
    elif any(keyword in message for keyword in ["tour", "guided tour", "activities"]):
        return "tours"
    elif any(keyword in message for keyword in ["hotel", "stay", "accommodation"]):
        return "hotels"
    else:
        return "general"

# Fetch data using Gemini API
async def fetch_tourism_info(query: str, intent: str) -> str:
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = f"Provide a brief overview (50 words or less) of {query} related to Dubai tourism."
        response = model.generate_content(prompt)
        info = response.text
        link = TOURISM_LINKS.get(intent, TOURISM_LINKS["general"])
        return f"{info}\n\nMore details: {link}"
    except Exception as e:
        return f"Couldn't fetch info. Error: {str(e)}"

# Generate response using Groq
async def generate_response(message: str, history: List[Message]) -> str:
    intent = detect_intent(message)
    if intent != "general":
        tourism_info = await fetch_tourism_info(message, intent)
        prompt = f"You are Afaq Tours Dubai, a professional chatbot. User asked: '{message}'. Use this info: {tourism_info}. Provide a concise response (50 words or less) with relevant chat history."
    else:
        prompt = f"You are Afaq Tours Dubai, a professional chatbot. User asked: '{message}'. Provide a concise response (50 words or less) about Dubai tourism. Include link: {TOURISM_LINKS['general']}. Use chat history: {history}"

    try:
        completion = groq_client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": "You are Afaq Tours Dubai, providing brief, accurate info on Dubai tourism."},
                *[{"role": msg.role, "content": msg.content} for msg in history],
                {"role": "user", "content": prompt}
            ],
            max_tokens=100
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Couldn't generate response. Error: {str(e)}"

# Root endpoint
@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    if "session_id" not in request.session:
        request.session["session_id"] = str(uuid.uuid4())
    session_id = request.session["session_id"]
    await init_db()
    chat_history = await get_history(session_id)
    return templates.TemplateResponse("index.html", {"request": request, "chat_history": chat_history})

# Chat endpoint
@app.post("/chat", response_class=HTMLResponse)
async def chat(request: Request, message: str = Form(...)):
    session_id = request.session.get("session_id", str(uuid.uuid4()))
    request.session["session_id"] = session_id
    await init_db()
    await save_message(session_id, "user", message)
    chat_history = await get_history(session_id)
    response = await generate_response(message, chat_history)
    await save_message(session_id, "assistant", response)
    chat_history = await get_history(session_id)
    return templates.TemplateResponse("index.html", {"request": request, "chat_history": chat_history})

# Clear history endpoint
@app.post("/clear_history", response_class=HTMLResponse)
async def clear_history(request: Request):
    session_id = request.session.get("session_id")
    if session_id:
        async with aiosqlite.connect("chat_history.db") as db:
            await db.execute("DELETE FROM history WHERE session_id = ?", (session_id,))
            await db.execute("DELETE FROM feedback WHERE session_id = ?", (session_id,))
            await db.commit()
    return templates.TemplateResponse("index.html", {"request": request, "chat_history": []})

# Image upload endpoint
@app.post("/upload_image", response_class=HTMLResponse)
async def upload_image(request: Request, image: UploadFile = File(...)):
    session_id = request.session.get("session_id", str(uuid.uuid4()))
    request.session["session_id"] = session_id
    await init_db()
    content = await image.read()
    encoded_image = base64.b64encode(content).decode("utf-8")
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = f"Describe this image in the context of Dubai tourism (50 words or less)."
        response = model.generate_content([prompt, {"inline_data": {"mime_type": image.content_type, "data": encoded_image}}])
        description = response.text
    except Exception as e:
        description = f"Couldn't analyze image. Error: {str(e)}"
    await save_message(session_id, "user", "Uploaded an image")
    await save_message(session_id, "assistant", description)
    chat_history = await get_history(session_id)
    return templates.TemplateResponse("index.html", {"request": request, "chat_history": chat_history})

# Feedback endpoint
@app.post("/feedback", response_class=HTMLResponse)
async def feedback(request: Request, message_index: int = Form(...), rating: str = Form(...)):
    session_id = request.session.get("session_id")
    if session_id:
        await save_feedback(session_id, message_index, rating)
    chat_history = await get_history(session_id)
    return templates.TemplateResponse("index.html", {"request": request, "chat_history": chat_history})

@app.post("/voice_chat")
async def voice_chat(request: Request, audio: UploadFile = File(...)):
    session_id = request.session.get("session_id", str(uuid.uuid4()))
    request.session["session_id"] = session_id
    await init_db()

    # Save uploaded file temporarily
    audio_path = f"temp_audio_{session_id}.wav"
    with open(audio_path, "wb") as f:
        f.write(await audio.read())

    # Convert to text
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data)
        except sr.UnknownValueError:
            return {"error": "Could not understand audio"}
        except sr.RequestError as e:
            return {"error": f"STT service failed: {e}"}

    # Save and generate response
    await save_message(session_id, "user", text)
    chat_history = await get_history(session_id)
    response = await generate_response(text, chat_history)
    await save_message(session_id, "assistant", response)

    # Convert response to speech
    tts = gTTS(response)
    response_audio_path = f"response_{session_id}.mp3"
    tts.save(response_audio_path)

    return FileResponse(response_audio_path, media_type="audio/mpeg", filename="response.mp3")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=8000)