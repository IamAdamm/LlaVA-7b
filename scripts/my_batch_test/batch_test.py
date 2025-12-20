import os
import json
import torch
from PIL import Image

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token

project_root = '/data/llava/LLaVA'
prompts = os.path.join(project_root, 'playground/data/prompts/myPrompts/prompts.json')
images = os.path.join(project_root, 'images')
results_file = os.path.join(project_root, 'results/results.json')

os.makedirs(os.path.dirname(results_file), exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "/data/llava/models/llava-v1.5-7b"

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name="llava-v1.5-7b",
    device=device
)

model.eval()

with open(prompts, 'r') as f:
    prompt_data = json.load(f)

results = []

for current_image in os.listdir(images):
    image_path = os.path.join(images, current_image)
    
    # Load the image using PIL
    image = Image.open(image_path).convert('RGB')
    image_tensor = process_images([image], image_processor, model.config)[0]
    image_tensor = image_tensor.unsqueeze(0).to(device).half()
    image_sizes = [image.size]
    
    for prompt_entry in prompt_data:
        prompt_text = prompt_entry.get('prompt', '') if isinstance(prompt_entry, dict) else prompt_entry

        conv = conv_templates["llava_v1"].copy()
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + prompt_text)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(
            prompt,
            tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors="pt"
        ).unsqueeze(0).to(device)

        with torch.inference_mode():
            outputs = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=image_sizes,
                max_new_tokens=128,
                temperature=0.2,
                do_sample=False,
                use_cache=True
            )

        answer = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].strip()

        results.append({
            'image': current_image,
            'prompt': prompt_text,
            'answer': answer
        })

with open(results_file, 'w') as f:
    json.dump(results, f, indent=4)

print(f"Works. Results saved to {results_file}")