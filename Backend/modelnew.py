from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from PIL import Image
import requests
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, AutoModelForObjectDetection, BitsAndBytesConfig, AutoModelForSeq2SeqLM
from ultralytics import YOLO
import numpy as np
import gc
import torchvision.transforms as transforms
import warnings
from typing import List, Dict, Optional
import os
import base64
import io

warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)

class ImageChatbot:
    def __init__(self, model_paths: Dict[str, str]):
        self.model_paths = model_paths
        self.yolo_model: Optional[YOLO] = None
        self.git_model: Optional[AutoModelForCausalLM] = None
        self.git_processor: Optional[AutoProcessor] = None
        self.llm_model: Optional[AutoModelForSeq2SeqLM] = None
        self.llm_tokenizer: Optional[AutoTokenizer] = None
        self.conversation_history: List[str] = []
        self.current_image: Optional[str] = None
        self.current_objects: Optional[List[List[float]]] = None
        self.current_caption: Optional[str] = None
        self.models_loaded = False

    def load_models(self):
        if not self.models_loaded:
            if not self.yolo_model:
                self.yolo_model = YOLO(self.model_paths['yolo'])
            
            if not self.git_processor or not self.git_model:
                self.git_processor = AutoProcessor.from_pretrained(self.model_paths['git_processor'])
                self.git_model = AutoModelForCausalLM.from_pretrained(self.model_paths['git_model'])
            
            if not self.llm_tokenizer or not self.llm_model:
                self.llm_tokenizer = AutoTokenizer.from_pretrained(self.model_paths['llm_tokenizer'])
                self.llm_model = AutoModelForSeq2SeqLM.from_pretrained(self.model_paths['llm_model'])
            
            self.models_loaded = True
            return "Models loaded successfully"
        else:
            return "Models already loaded"

    def preprocess_image(self, image: Image.Image) -> Image.Image:
        return image.convert('RGB')

    def detect_objects(self, image: Image.Image) -> Optional[List[List[float]]]:
        results = self.yolo_model(image)
        self.current_objects = results[0].boxes.data.tolist()
        return self.current_objects if self.current_objects else None

    def generate_caption(self, image: Image.Image) -> str:
        inputs = self.git_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            generated_ids = self.git_model.generate(pixel_values=inputs.pixel_values, max_length=50)
        self.current_caption = self.git_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return self.current_caption

    def _build_context(self) -> str:
        context = "\n".join(self.conversation_history[-8:])  # Keep only last 8 exchanges
        if self.current_caption:
            context += f"\nIMPORTANT - Image caption: {self.current_caption}"
        if self.current_objects:
            object_names = [str(obj[6]) for obj in self.current_objects if len(obj) > 6]
            if object_names:
                context += f"\nAdditional information - Some objects in the image: {', '.join(object_names[:5])}"
        context += "\nPlease focus primarily on the image caption when responding to the user's question."
        return context

    def generate_response(self, user_input: str) -> str:
        context = self._build_context()
        prompt = f"You are an AI assistant that helps analyze images. Here's the context:\n\n{context}\n\nHuman: {user_input}\nAI:"
        
        input_ids = self.llm_tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.llm_model.device)
        with torch.no_grad():
            output = self.llm_model.generate(input_ids, max_length=200, num_return_sequences=1)
        response = self.llm_tokenizer.decode(output[0], skip_special_tokens=True)
        
        self._update_conversation_history(user_input, response)
        return response


    def _update_conversation_history(self, user_input: str, response: str):
        self.conversation_history.append(f"Human: {user_input}")
        self.conversation_history.append(f"AI: {response}")
        if len(self.conversation_history) > 8:  # Keep only last 4 exchanges (8 messages)
            self.conversation_history = self.conversation_history[-8:]

    def process_image(self, image: Image.Image) -> str:
        try:
            self.detect_objects(image)
            self.generate_caption(image)
            return f"Image processed. Caption: {self.current_caption}"
        except Exception as e:
            return f"Error processing image: {str(e)}"

    def chat(self, user_input: str, image: Optional[Image.Image] = None) -> str:
        if not all([self.yolo_model, self.git_model, self.llm_model]):
            return "Models are not loaded. Please call load_models() first."
        
        if image:
            return self.process_image(image)
        else:
            return self.generate_response(user_input)

    def cleanup(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        for attr in ['yolo_model', 'git_model', 'git_processor', 'llm_model', 'llm_tokenizer']:
            if hasattr(self, attr):
                setattr(self, attr, None)

        self.conversation_history.clear()
        self.current_image = None
        self.current_objects = None
        self.current_caption = None

        gc.collect()

model_paths = {
    'yolo': r"yolov10l.pt",
    'git_processor': "microsoft/git-large-coco",
    'git_model': "microsoft/git-large-coco",
    'llm_tokenizer': "google/flan-t5-base",
    'llm_model': "google/flan-t5-base"
}

chatbot = ImageChatbot(model_paths)

@app.route('/warmup', methods=['GET'])
def warmup():
    result = chatbot.load_models()
    return jsonify({"status": result})

@app.route('/send-to-flask', methods=['POST'])
def process_message():
    if not chatbot.models_loaded:
        chatbot.load_models()
    
    data = request.json
    prompt = data.get('prompt', '')
    image_data = data.get('image')

    if image_data:
        # Decode base64 image
        image_data = image_data.split(',')[1]  # Remove the "data:image/jpeg;base64," part
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        response = chatbot.chat(prompt, image)
    else:
        response = chatbot.chat(prompt)

    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True, port=3000)