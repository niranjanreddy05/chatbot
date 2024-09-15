import base64
import io
from flask import Flask, request, jsonify
import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, AutoModelForSeq2SeqLM
from ultralytics import YOLO
from flask import Flask, jsonify
from flask_cors import CORS, cross_origin

app = Flask(__name__)
CORS(app)


app = Flask(__name__)

class ImageChatbot:
    def __init__(self):
        self.yolo_model = None
        self.git_model = None
        self.git_processor = None
        self.llm_model = None
        self.llm_tokenizer = None
        self.conversation_history = []
        self.current_image = None
        self.current_objects = None
        self.current_caption = None

    def load_models(self):
        if self.yolo_model is None:
            self.yolo_model = YOLO("yolov8m.pt")

        if self.git_processor is None or self.git_model is None:
            self.git_processor = AutoProcessor.from_pretrained("microsoft/git-large-coco")
            self.git_model = AutoModelForCausalLM.from_pretrained("microsoft/git-large-coco")

        if self.llm_tokenizer is None or self.llm_model is None:
            self.llm_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
            self.llm_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

    def preprocess_image_yolo(self, image):
        return Image.open(io.BytesIO(image)).convert('RGB')

    def preprocess_image_git(self, image):
        image = Image.open(io.BytesIO(image)).convert('RGB')
        return self.git_processor(images=image, return_tensors="pt")

    def detect_objects(self, image):
        image = self.preprocess_image_yolo(image)
        results = self.yolo_model(image)
        self.current_objects = results[0].boxes.data.tolist()
        return [result.boxes.data.tolist() for result in results]

    def generate_caption(self, image):
        inputs = self.preprocess_image_git(image)
        with torch.no_grad():
            generated_ids = self.git_model.generate(pixel_values=inputs.pixel_values, max_length=50)
        self.current_caption = self.git_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return self.current_caption

    def generate_response(self, user_input):
        context = "\n".join(self.conversation_history)
        if self.current_caption:
            context += f"\nImage caption: {self.current_caption}"
        if self.current_objects:
            context += f"\nDetected objects: {', '.join([str(obj[6]) for obj in self.current_objects if len(obj) > 6])}"

        prompt = f"{context}\nHuman: {user_input}\nAI:"

        input_ids = self.llm_tokenizer.encode(prompt, return_tensors="pt").to(self.llm_model.device)
        with torch.no_grad():
            output = self.llm_model.generate(input_ids, max_length=200, num_return_sequences=1)
        response = self.llm_tokenizer.decode(output[0], skip_special_tokens=True)

        self.conversation_history.append(f"Human: {user_input}")
        self.conversation_history.append(f"AI: {response}")

        return response

    def process_image(self, image):
        self.current_image = image
        self.detect_objects(image)
        self.generate_caption(image)

    def chat(self, user_input):
        if not self.yolo_model or not self.git_model or not self.llm_model:
            return "Models are not loaded. Please call load_models() first."

        if user_input.lower().startswith("process image:"):
            # Not used in the HTTP request; image is directly processed
            return f"Image processed. Caption: {self.current_caption}"
        else:
            return self.generate_response(user_input)

    def cleanup(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        del self.yolo_model
        del self.git_model
        del self.git_processor
        del self.llm_model
        del self.llm_tokenizer

        self.yolo_model = None
        self.git_model = None
        self.git_processor = None
        self.llm_model = None
        self.llm_tokenizer = None
        self.conversation_history = []
        self.current_image = None
        self.current_objects = None
        self.current_caption = None

# Initialize chatbot and load models
chatbot = ImageChatbot()
chatbot.load_models()

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    image_base64 = data.get('image')
    prompt = data.get('prompt', '')

    if image_base64:
        # Remove any prefix and decode the base64 image
        if image_base64.startswith('data:image/jpeg;base64,'):
            image_base64 = image_base64.replace('data:image/jpeg;base64,', '')
        elif image_base64.startswith('data:image/png;base64,'):
            image_base64 = image_base64.replace('data:image/png;base64,', '')
        image_bytes = base64.b64decode(image_base64)
        chatbot.process_image(image_bytes)
        response = chatbot.chat(prompt)
    else:
        response = chatbot.chat(prompt)

    return jsonify({"response": response})

@app.route('/cleanup', methods=['POST'])
def cleanup():
    chatbot.cleanup()
    return jsonify({"status": "cleanup completed"})

if __name__ == '__main__':
    app.run(port=5000)