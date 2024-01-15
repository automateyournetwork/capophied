import torch
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer

# Check if CUDA is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Display the device being used
if device.type == 'cuda':
    st.write("Your system is using the GPU")
else:
    st.write("Your system is using the CPU")

# Load the model and tokenizer, and move the model to the device
model = AutoModelForCausalLM.from_pretrained("/app/phi-2", torch_dtype="auto", trust_remote_code=True).to(device)
tokenizer = AutoTokenizer.from_pretrained("/app/phi-2", trust_remote_code=True)

# Streamlit UI
st.title("Capophied")
user_prompt = st.text_area("Enter your prompt", "Where is the CN Tower?")

if st.button("Generate Output"):
    formatted_prompt = f"Instruct: {user_prompt}\nOutput:"
    with torch.no_grad():
        # Encode input and move tensors to the device
        token_ids = tokenizer.encode(formatted_prompt, return_tensors="pt").float.to(device)
        output_ids = model.generate(
            token_ids,
            max_new_tokens=512,
            no_repeat_ngram_size=2,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            num_return_sequences=1,           
        )
    
    # Decode output and move it back to CPU for displaying
    output = tokenizer.decode(output_ids[0][token_ids.size(1):], skip_special_tokens=True)
    
    st.text("Generated Output:")
    st.write(output)
