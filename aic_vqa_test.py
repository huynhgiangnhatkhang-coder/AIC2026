import base64
import json
import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import requests

# LM Studio Server Configuration
API_URL = "http://aicpc.sytes.net:1234/v1/chat/completions"


def encode_image_to_base64(image_path):
    """Encodes a local image file to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


class AICVisualQAApp:

    def __init__(self, root):
        self.root = root
        self.root.title("AIC HCM 2026 - Query Type 2 (Q&A) Tester")
        self.root.geometry("850x750")
        self.root.minsize(700, 600)

        self.selected_image_path = None

        self.setup_ui()

    def setup_ui(self):
        # Top Frame: File Selector
        top_frame = ttk.LabelFrame(self.root, text=" 1. Keyframe Selection ", padding=10)
        top_frame.pack(fill="x", padx=15, pady=8)

        self.btn_browse = ttk.Button(
            top_frame, text="📁 Choose Picture / Keyframe", command=self.choose_image
        )
        self.btn_browse.pack(side="left", padx=5)

        self.lbl_file_path = ttk.Label(
            top_frame, text="No image selected", foreground="gray"
        )
        self.lbl_file_path.pack(side="left", padx=10, fill="x", expand=True)

        # Middle Frame: Image Preview
        preview_frame = ttk.LabelFrame(self.root, text=" Image Preview ", padding=10)
        preview_frame.pack(fill="both", expand=True, padx=15, pady=5)

        self.lbl_preview = ttk.Label(
            preview_frame,
            text="No image preview available",
            anchor="center",
            background="#f0f0f0",
        )
        self.lbl_preview.pack(fill="both", expand=True)

        # Question Frame
        question_frame = ttk.LabelFrame(
            self.root, text=" 2. Question / Prompt (Query Type 2) ", padding=10
        )
        question_frame.pack(fill="x", padx=15, pady=8)

        self.txt_question = ttk.Entry(question_frame, font=("Arial", 11))
        self.txt_question.insert(
            0,
            "Trong video quay cảnh bữa tiệc, người phụ nữ mặc váy đỏ đang cầm ly màu gì?",
        )
        self.txt_question.pack(fill="x", side="left", expand=True, padx=(0, 10))

        self.btn_send = ttk.Button(
            question_frame, text="🚀 Send to LLM", command=self.start_query_thread
        )
        self.btn_send.pack(side="right")

        # Bottom Frame: Response
        response_frame = ttk.LabelFrame(
            self.root, text=" 3. LLM Response ", padding=10
        )
        response_frame.pack(fill="both", expand=True, padx=15, pady=8)

        self.txt_response = tk.Text(
            response_frame,
            height=6,
            wrap="word",
            font=("Arial", 11),
            bg="#ffffff",
        )
        self.txt_response.pack(fill="both", expand=True)

        # Status Bar
        self.status_var = tk.StringVar(value="Ready")
        self.status_bar = ttk.Label(
            self.root,
            textvariable=self.status_var,
            relief="sunken",
            anchor="w",
            padding=5,
        )
        self.status_bar.pack(fill="x", side="bottom")

    def choose_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Video Keyframe",
            filetypes=[
                ("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("All Files", "*.*"),
            ],
        )
        if file_path:
            self.selected_image_path = file_path
            self.lbl_file_path.config(
                text=os.path.basename(file_path), foreground="black"
            )
            self.display_image_preview(file_path)

    def display_image_preview(self, path):
        try:
            img = Image.open(path)
            # Resize image keeping aspect ratio
            img.thumbnail((450, 300))
            self.photo_img = ImageTk.PhotoImage(img)
            self.lbl_preview.config(image=self.photo_img, text="")
        except Exception as e:
            messagebox.showerror("Error", f"Could not load image: {e}")

    def start_query_thread(self):
        if not self.selected_image_path:
            messagebox.showwarning(
                "Missing Image", "Please select an image/keyframe first!"
            )
            return

        question = self.txt_question.get().strip()
        if not question:
            messagebox.showwarning("Missing Prompt", "Please enter a question!")
            return

        # Disable button while querying to prevent multiple clicks
        self.btn_send.config(state="disabled")
        self.status_var.set("Sending request to LM Studio server...")
        self.txt_response.delete("1.0", tk.END)
        self.txt_response.insert(tk.END, "Waiting for model response...\n")

        # Run request in background thread to avoid freezing the GUI
        thread = threading.Thread(target=self.send_query_request, args=(question,))
        thread.daemon = True
        thread.start()

    def send_query_request(self, question):
        try:
            base64_image = encode_image_to_base64(self.selected_image_path)

            # OpenAI-compatible Vision payload for LM Studio
            payload = {
                "model": "local-model",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an AI assistant for video visual question answering (VQA). You must NOT output internal thoughts or reasoning. Skip all analysis and output ONLY the final answer precisely and concisely.",
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": question},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                },
                            },
                        ],
                    },
                ],
                "temperature": 0.2,
                # "max_tokens": 1500,
                "stream": False,
            }

            headers = {"Content-Type": "application/json"}

            response = requests.post(
                API_URL, headers=headers, json=payload, timeout=60
            )

            if response.status_code == 200:
                result = response.json()
                answer = result["choices"][0]["message"]["content"].strip()
                self.update_ui_result(answer, "Completed successfully.")
            else:
                error_msg = f"HTTP Error {response.status_code}:\n{response.text}"
                self.update_ui_result(error_msg, "Error occurred.")

        except requests.exceptions.ConnectionError:
            self.update_ui_result(
                f"Connection Error: Unable to connect to LM Studio at {API_URL}.\nPlease ensure the server is running and accessible.",
                "Connection failed.",
            )
        except Exception as e:
            self.update_ui_result(f"Error: {str(e)}", "Request failed.")

    def update_ui_result(self, text, status):
        def _update():
            self.txt_response.delete("1.0", tk.END)
            self.txt_response.insert(tk.END, text)
            self.status_var.set(status)
            self.btn_send.config(state="normal")

        self.root.after(0, _update)


if __name__ == "__main__":
    root = tk.Tk()
    app = AICVisualQAApp(root)
    root.mainloop()