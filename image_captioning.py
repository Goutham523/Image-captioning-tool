from flask import Flask, render_template, request
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
from PIL import Image
import torch

app = Flask(__name__)

model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

max_length = 16
num_beams = 4
num_return_sequences = 4  # Number of different sequences to generate
gen_kwargs = {"max_length": max_length, "num_beams": num_beams, "num_return_sequences": num_return_sequences}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the uploaded image and number of captions to generate
        uploaded_file = request.files['file']
        num_captions = int(request.form['num_captions'])
        
        # Generate captions for the uploaded image
        images = []
        i_image = Image.open(uploaded_file)
        if i_image.mode != "RGB":
            i_image = i_image.convert(mode="RGB")

        images.append(i_image)

        pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device)

        captions = []
        for i in range(num_captions):
            output_ids = model.generate(pixel_values, **gen_kwargs)

            preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            preds = [pred.strip() for pred in preds]

            caption = preds[i]
            captions.append(caption)

        # Render the results template with the generated captions
        return render_template('results.html', captions=captions, num_captions=num_captions)
    
    # Render the index template with the upload form
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)