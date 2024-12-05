import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Load the model and tokenizer
def load_model():
    print("Loading model...")
    model_name = "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"  # Replace with your model base
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    base_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    model = PeftModel.from_pretrained(base_model, "dapraws/daprawsgpt-ft-qlora")
    return model, tokenizer

def generate_response(model, tokenizer, user_message):
    instruction_string = (
        "daprawsGPT, functioning as a virtual FAQ assistant for Telkom University, "
        "communicates in clear, accessible language, escalating to technical depth upon request. "
        "It reacts to feedback aptly and ends responses with its signature '–daprawsGPT'."
    )
    prompt = f"[INST] {instruction_string} \n{user_message}\n[/INST]"
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Generate response
    outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"), max_new_tokens=280)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def main():
    print("Welcome to daprawsGPT CLI Chatbot!")
    print("Type your question or type 'exit' to quit.")
    
    # Load model
    model, tokenizer = load_model()
    model.eval()

    while True:
        # Get user input
        user_message = input("\nYou: ")
        if user_message.lower() == "exit":
            print("Goodbye!")
            break
        
        # Generate and print response
        response = generate_response(model, tokenizer, user_message)
        print(f"daprawsGPT: {response}")

if __name__ == "__main__":
    main()
