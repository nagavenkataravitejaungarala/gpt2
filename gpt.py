import streamlit as st
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

def process_csv(file_path):
    df = pd.read_csv(file_path)
    qa_pairs = []

    for index, row in df.iterrows():
        question = row['question']
        answer = row['answer']
        qa_pairs.append(f"Question: {question}\nAnswer: {answer}\n")

    return qa_pairs

def load_dataset(file_path, tokenizer):
    qa_pairs = process_csv(file_path)
    tokenized_dataset = tokenizer(qa_pairs, truncation=True,
                                  padding='max_length', max_length=128,
                                  return_tensors="pt")
    return tokenized_dataset

def ask_question(question, model, tokenizer, max_length=128, num_return_sequences=1):
    prompt = f"Question: {question}\nAnswer:"
    inputs = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True)
    outputs = model.generate(
        inputs,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        no_repeat_ngram_size=3,
        do_sample=True,
        temperature=1.0,
        top_k=50,
        top_p=0.9,
        early_stopping=True,
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = answer.replace(prompt, "").strip()

    # Truncate the answer after the first newline character
    answer = answer.split("\n")[0]

    return answer

def main():
    st.title("GPT-2 Question Answering")

    # Load the pre-trained GPT-2 model and tokenizer
    model_name = "gpt2"
    tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    tokenizer.pad_token = tokenizer.eos_token

    # Load and preprocess the dataset
    train_dataset = load_dataset("/content/questions.csv", tokenizer)
    valid_dataset = load_dataset("/content/valid.csv", tokenizer)

    # Train the model (if needed)
    if len(train_dataset) > 0:
        model.train()
        model.resize_token_embeddings(len(tokenizer))

    # Fine-tune the model (if needed)
    if len(valid_dataset) > 0:
        model.eval()

    # Save the fine-tuned model
    model.save_pretrained("fine_tuned_BhagavatGita_gpt2")

    # Load the fine-tuned model
    fine_tuned_model = GPT2LMHeadModel.from_pretrained("fine_tuned_BhagavatGita_gpt2")

    question = st.text_input("Enter your question:")
    if st.button("Ask"):
        if question:
            answer = ask_question(question, fine_tuned_model, tokenizer)
            st.text_area("Answer:", answer)

if __name__ == "__main__":
    main()
