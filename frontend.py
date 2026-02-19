import gradio as gr
import requests
import json

API_URL = "http://127.0.0.1:8000"

# Upload File to Backend
def upload_document(file):
    try:
        files = {"file": open(file.name, "rb")}
        response = requests.post(f"{API_URL}/upload", files=files)
        return response.json()
    except Exception as e:
        return {"error": str(e)}


# Streaming Chat
def chat(message, history):
    if not message or not message.strip():
        yield "", history
        return

    # Initialize history if empty
    if history is None:
        history = []

    # Add user message and a placeholder for the assistant
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": "Searching..."}) # Initial placeholder
    yield "", history

    try:
        payload = {"input": message}
        response = requests.post(
            f"{API_URL}/chat/stream",
            json=payload,
            stream=True,
            timeout=60
        )

        if response.status_code != 200:
            history[-1]["content"] = f"⚠️ Server Error: {response.text}"
            yield "", history
            return

        partial = ""
        for line in response.iter_lines():
            if not line: continue
            
            decoded = line.decode("utf-8").strip()
            
            # LangServe sends data in "data: <content>" format
            if decoded.startswith("data:"):
                content = decoded[5:].strip() # Remove "data:"
                
                # If it's a JSON string (like "Hello"), we need to unquote it
                try:
                    content = json.loads(content)
                except:
                    pass # Use as-is if not valid JSON

                # Skip metadata objects {}
                if isinstance(content, dict) or not content:
                    continue
                
                partial += str(content)
                history[-1]["content"] = partial
                yield "", history

    except Exception as e:
        history[-1]["content"] = f"⚠️ Connection interrupted: {str(e)}"
        yield "", history

# UI
with gr.Blocks(title="Smart AI/RAG Assistant") as demo:

    gr.Markdown("# 🚀 Smart AI/RAG Assistant")
    gr.Markdown("FastAPI + LangServe Backend + Streaming")

    gr.Markdown("## 📂 Upload PDF or DOCX Document")
    file_input = gr.File(file_types=[".pdf", ".docx"])
    upload_btn = gr.Button("Upload & Index" , variant="primary")
    upload_status = gr.Textbox(label="Status", lines=3, interactive=False)

    upload_btn.click(upload_document, file_input, upload_status)

    gr.Markdown("## 💬 Chat")
    chatbot = gr.Chatbot(height=450)
    message = gr.Textbox(placeholder="Ask anything...")
    send = gr.Button("Send" , variant="primary")

    send.click(chat, [message, chatbot], [message, chatbot])
    message.submit(chat, [message, chatbot], [message, chatbot])


if __name__ == "__main__":
    demo.launch()
