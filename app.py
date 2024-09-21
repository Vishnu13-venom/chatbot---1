with open('app.py', 'w') as f:
    f.write('''
import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the fine-tuned model and tokenizer
model_path = "./fine_tuned_model"  # Replace with the path where your fine-tuned model is stored or Hugging Face model name
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

# Set up the Streamlit interface
st.title("Chatbot with Fine-Tuned GPT-2 Model")
st.write("Ask anything related to the data, and the chatbot will respond based on the fine-tuned model.")

# User input for the chatbot
user_input = st.text_input("You: ", placeholder="Type your question here...")

# Generate a response from the model
if user_input:
    # Encode the user input and generate a response
    input_ids = tokenizer.encode(f"<|startoftext|> {user_input}", return_tensors='pt')
    output = model.generate(
        input_ids,
        max_length=200,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        early_stopping=True
    )

    # Decode the generated response
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Display the chatbot response
    st.write(f"Chatbot: {response}")
    ''')
