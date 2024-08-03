import pandas as pd
from data_processing import load_data
from model import get_embeddings, train_model
from inference import match_speakers, evaluate
from transformers import Trainer, TrainingArguments

def main():
    # Load datasets
    train_df, database_df, test_df = load_data()

    # Train the model
    model, processor = train_model(train_df)

    # Extract embeddings for the database and test sets
    database_embeddings = {row['label']: get_embeddings(model, processor, f"Dataset/{row['fdir']}") for _, row in database_df.iterrows()}
    test_embeddings = {row['fdir']: get_embeddings(model, processor, f"Dataset/{row['fdir']}") for _, row in test_df.iterrows()}

    # Match speakers and evaluate
    predictions = match_speakers(test_embeddings, database_embeddings)
    accuracy, f1 = evaluate(test_df['label'].tolist(), predictions)

    print(f"Accuracy: {accuracy}")
    print(f"F1 Score: {f1}")

    # Prepare training arguments
    #training_args = TrainingArguments(
     #   output_dir='./results',
      #  num_train_epochs=3,
       # per_device_train_batch_size=16,
        #per_device_eval_batch_size=16,
        #warmup_steps=500,
        #weight_decay=0.01,
        #logging_dir='./logs',
        #logging_steps=10,
    #)

    # Prepare the Trainer
   # trainer = Trainer(
    #    model=model,
     #   args=training_args,
      #  train_dataset=train_df,
       # eval_dataset=test_df
    #)

    # Train the model
   # trainer.train()

    # Evaluate the model
    #metrics = trainer.evaluate()
    #print(metrics)

if __name__ == "__main__":
    main()
