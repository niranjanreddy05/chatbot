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
from flask import Flask, request, jsonify
from flask_cors import CORS

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

    def load_models(self):
        if not self.yolo_model:
            self.yolo_model = YOLO(self.model_paths['yolo'])
        
        if not self.git_processor or not self.git_model:
            self.git_processor = AutoProcessor.from_pretrained(self.model_paths['git_processor'])
            self.git_model = AutoModelForCausalLM.from_pretrained(self.model_paths['git_model'])
        
        if not self.llm_tokenizer or not self.llm_model:
            self.llm_tokenizer = AutoTokenizer.from_pretrained(self.model_paths['llm_tokenizer'])
            self.llm_model = AutoModelForSeq2SeqLM.from_pretrained(self.model_paths['llm_model'])

    def preprocess_image(self, image_data: str) -> Image.Image:
        image_bytes = base64.b64decode(image_data.split(',')[1])
        return Image.open(io.BytesIO(image_bytes)).convert('RGB')

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

    def generate_response(self, user_input: str) -> str:
        context = self._build_context()
        prompt = f"{context}\nHuman: {user_input}\nAI:"
        
        input_ids = self.llm_tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.llm_model.device)
        with torch.no_grad():
            output = self.llm_model.generate(input_ids, max_length=200, num_return_sequences=1)
        full_response = self.llm_tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Extract only the AI's response
        ai_response = full_response.split("AI:")[-1].strip()
        
        self._update_conversation_history(user_input, ai_response)
        return ai_response

    def _build_context(self) -> str:
        context = "\n".join(self.conversation_history[-8:])  # Keep only last 8 exchanges
        if self.current_caption:
            context += f"\nImage caption: {self.current_caption}"
        if self.current_objects:
            context += f"\nDetected objects: {', '.join([str(obj[6]) for obj in self.current_objects if len(obj) > 6])}"
        return context

    def _update_conversation_history(self, user_input: str, response: str):
        self.conversation_history.append(f"Human: {user_input}")
        self.conversation_history.append(f"AI: {response}")
        if len(self.conversation_history) > 16:  # Keep only last 8 exchanges
            self.conversation_history = self.conversation_history[-16:]

    def process_image(self, image_data: str) -> str:
        try:
            image = self.preprocess_image(image_data)
            self.detect_objects(image)
            self.generate_caption(image)
            return f"Image processed. Caption: {self.current_caption}"
        except Exception as e:
            return f"Error processing image: {str(e)}"

    def chat(self, user_input: str, image_data: Optional[str] = None) -> str:
        if not all([self.yolo_model, self.git_model, self.llm_model]):
            return "Models are not loaded. Please call load_models() first."
        
        if image_data:
            return self.process_image(image_data)
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
    'yolo': r"yolov10m.pt",
    'git_processor': "microsoft/git-large-coco",
    'git_model': "microsoft/git-large-coco",
    'llm_tokenizer': "google/flan-t5-base",
    'llm_model': "google/flan-t5-base"
}

chatbot = ImageChatbot(model_paths)
chatbot.load_models()

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    image = data.get('image')
    prompt = data.get('prompt')

    if image:
        response = chatbot.chat(prompt, image_data=image)
    else:
        response = chatbot.chat(prompt)

    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(port=5000)