import os
import uuid
from termcolor import colored
from googletrans import Translator

from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from flask import Flask, request, jsonify


app = Flask(__name__)


ROOT_FOLDER = os.path.abspath("images")
ALLOWED_EXTENSIONS = {".jpg", ".png", ".jpeg"}
os.makedirs(ROOT_FOLDER, exist_ok=True)


def describe_image(image_path):
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, return_tensors="pt")
    output = model.generate(**inputs)
    description = processor.decode(output[0], skip_special_tokens=True)
    return description

def check_file(ruta_archivo):
    return os.path.isfile(ruta_archivo) and os.path.getsize(ruta_archivo) > 0

@app.route("/image2text", methods=["POST"])
def upload_video():
    if "img" not in request.files:
        return jsonify({"error": "No se envió ningún archivo"}), 400
    
    img = request.files["img"]
    if not img or img.filename=="":
        return jsonify({"ERROR": "Empty file"}), 400
    
    file_extension = os.path.splitext(img.filename)[1].lower()
    if file_extension not in ALLOWED_EXTENSIONS:
        return jsonify({"ERROR": "Unsupported file type"}), 400
    
    session_id = str(uuid.uuid4())
    session_upload_folder = os.path.join(ROOT_FOLDER, session_id)
    os.makedirs(session_upload_folder, exist_ok=True)
    img_path = os.path.join(session_upload_folder, f"img_LR{file_extension}")
    img.save(img_path)

    traductor = Translator()
    print(colored(f"Image saved at {img_path}", "red"))
    text_EN = describe_image(image_path=img_path)
    return jsonify({"description-EN": f"{text_EN}",
                    "description-ES": f"{traductor.translate(text_EN, src='en', dest='es').text}"}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8880,debug=True)
