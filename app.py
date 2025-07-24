from flask import Flask, render_template, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    print("Received message:", msg)
    input = msg
    return get_chat_response(input)

def get_chat_response(input):
    # encode the user input, add the eos_token and return a tensor in Pytorch
    new_user_input_ids = tokenizer.encode(str(input) + tokenizer.eos_token, return_tensors='pt')
    # generate a response
    chat_history_ids = model.generate(new_user_input_ids, max_length=300, pad_token_id=tokenizer.eos_token_id)
    # decode and return the response
    return tokenizer.decode(chat_history_ids[:, new_user_input_ids.shape[-1]:][0], skip_special_tokens=True)

if __name__ == "__main__":
    app.run()
