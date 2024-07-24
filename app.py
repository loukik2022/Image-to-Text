import gradio as gr
import time
from inference import generate_caption

def run(image):
    caption = generate_caption(image)
    return caption

interface = gr.Interface(
    fn=run,
    inputs=["image"],   
    outputs=["text"],
    title="Image-to-Text",
    description="Upload an image to generate a caption.",
    allow_flagging='never',
)

interface.launch()