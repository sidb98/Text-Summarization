# %pip install fastapi uvicorn -q
from model_config import T5Model
from fastapi import FastAPI
import uvicorn
import gradio as gr


# load best model from checkpoint
best_model_path = "checkpoints/best-checkpoint-v1.ckpt"
best_model = T5Model.load_from_checkpoint(best_model_path)
best_model.eval().to("cpu")


def summarize(text):
    summary = best_model.genrate_summary(text)
    return summary[0]


iface = gr.Interface(
    fn=summarize,
    inputs="textbox",
    outputs="text",
    title="T5 News Article Summarization",
    description="Summarize news articles with T5",
)

if __name__ == "__main__":
    iface.launch(share=True)

# FAST API
# app  = FastAPI()

# @app.get("/")
# def read_root():
#     return {"message": "Welcome to the T5 Summarization API"}

# @app.get("/summarize/{text}")
# def summarize(text):
#     summary = best_model.genrate_summary(text)
#     return summary

# if __name__ == "__main__":
#     uvicorn.run(app, host='127.0.0.1', port=8000, log_level="info")
