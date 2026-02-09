import uuid
import gradio as gr
import edge_tts
import asyncio

def speak(text: str):
    fn = f"tts_{uuid.uuid4().hex}.mp3"
    asyncio.run(edge_tts.Communicate(text, "en-GB-SoniaNeural").save(fn))
    return fn

demo = gr.Interface(
    fn=speak,
    inputs=gr.Textbox(value="Say something"),
    outputs=gr.Audio(type="filepath"),
)

demo.launch()
