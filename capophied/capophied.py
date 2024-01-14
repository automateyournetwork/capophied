import torch
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer

# Check if CUDA is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Display the device being used
if device.type == 'cuda':
    st.write("Using GPU")
else:
    st.write("Using CPU")

# Load the model and tokenizer, and move the model to the device
model = AutoModelForCausalLM.from_pretrained("/app/phi-2", torch_dtype="auto", trust_remote_code=True).to(device)
tokenizer = AutoTokenizer.from_pretrained("/app/phi-2", trust_remote_code=True)

# Streamlit UI
st.title("Capophied")
prompt = st.text_area("Enter your prompt", "Where is the CN Tower and what can you tell me about")

if st.button("Generate Output"):
    with torch.no_grad():
        # Encode input and move tensors to the device
        token_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(device)
        output_ids = model.generate(
            token_ids,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.3
        )
    
    # Decode output and move it back to CPU for displaying
    output = tokenizer.decode(output_ids[0][token_ids.size(1):], skip_special_tokens=True)
    
    st.text("Generated Output:")
    st.write(output)