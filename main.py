from huggingface_hub import InferenceClient
from PIL import Image,ImageDraw
HF_API_KEY = "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"  # Replace
client=InferenceClient(api_key=HF_API_KEY)
results=client.object_detection(image="image.png",model="facebook/detr-resnet-50")
image=Image.open("image.png")
draw=ImageDraw.Draw(image)
for obj in results:
    box=obj["box"]
    label=obj["label"]
    score=obj["score"]
    x1,y1=box["xmin"],box["ymin"]
    x2,y2=box["xmax"],box["ymax"]
    draw.rectangle((x1,y1,x2,y2), outline="red", width=2)
    draw.text((x1, y1), f"{label}: {score:.2f}", fill="white")
image.save("output_image.png")
print("Image processed and saved as output_image.png")
