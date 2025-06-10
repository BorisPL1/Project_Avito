import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, DataCollatorWithPadding


class SentenceClassifier:

    def load_from_file(self, model_path):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)           
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
            data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
            self.trainer = Trainer(model=model, data_collator=data_collator)           
            return self.trainer
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise

    def tokenize_function(self, text):
        return self.tokenizer(text["messages"])  
    
    def predict_tone(self, text: Dataset):
        try:
            text = text.map(self.tokenize_function, batched=True, num_proc=8)
            y_logits = self.trainer.predict(text, ignore_keys=['labels']).predictions
            text = text.add_column('tone', np.argmax(y_logits, axis=1))
            text.set_format('pandas')
            text = text.select_columns(['tone'])[:]
            return text
        
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            raise


