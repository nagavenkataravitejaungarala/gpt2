import streamlit as st
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from datasets import Dataset

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
    dataset = Dataset.from_dict(tokenized_dataset)
    return dataset

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
    # Load the pre-trained GPT-2 model and tokenizer
    model_name = "gpt2"
    tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    tokenizer.pad_token = tokenizer.eos_token

    # Load and preprocess the dataset
    train_dataset = load_dataset("/path/to/bhagavatgita_dataset.csv", tokenizer)

    # Configure and train the model using the Trainer class
    training_args = TrainingArguments(
        output_dir="output",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        eval_steps=100,
        save_steps=100,
        warmup_steps=0,
        logging_dir="logs",
        evaluation_strategy="steps",
        save_total_limit=3,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )

    trainer.train()
    # Save the fine-tuned model
    model.save_pretrained("fine_tuned_bhagavatgita_gpt2")

    # Load the fine-tuned model
    fine_tuned_model = GPT2LMHeadModel.from_pretrained("fine_tuned_bhagavatgita_gpt2")

    st.title("GPT-2 Question Answering")

    question = st.text_input("Enter your question:")
    if st.button("Ask"):
        if question:
            answer = ask_question(question, fine_tuned_model, tokenizer)
            st.write("Answer:", answer)

if __name__ == "__main__":
    main()
