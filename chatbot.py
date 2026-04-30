from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 1. Load the pre-trained model and tokenizer
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

print("Chatbot is ready! Type 'quit' to exit.")

# 2. Chatting loop
chat_history_ids = None

for step in range(5):  # Limiting to 5 turns for this example
    user_input = input(">> User: ")
    
    if user_input.lower() == 'quit':
        break

    # Encode user input and add the EOS (End Of Sentence) token
    new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

    # Append the new user input to the chat history
    bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids

    # Generate a response
    chat_history_ids = model.generate(
        bot_input_ids, 
        max_length=1000, 
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=3,       
        do_sample=True, 
        top_k=100, 
        top_p=0.7,
        temperature=0.8
    )

    # Decode and print the response
    bot_output = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    print(f"Bot: {bot_output}")