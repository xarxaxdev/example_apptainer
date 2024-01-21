#DATASET LOADING
import evaluate,torch,os
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import AutoTokenizer,DataCollatorWithPadding



dataset = load_dataset("xnli",'en')


#TOKENIZING
base_model = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(base_model)
def preprocess_function(examples):
    return tokenizer(examples["premise"], examples["hypothesis"] ,truncation=True)

tokenized_dataset = dataset.map(preprocess_function, batched=True) 


#COLLATION    
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

#EVALUATION
metric_name= "f1"
metric = evaluate.load(metric_name)
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels, average='macro')

id2label = {0: "Entailment", 1: "Neutral",2:'Contradiction'}
label2id = {"Entailment": 0, "Neutral": 1, 'Contradiction':2}


epochs = 1 
lr = 2e-5
bs= 16
physical_batch_size = 8
model = AutoModelForSequenceClassification.from_pretrained(
    base_model, num_labels=len(id2label), id2label=id2label, label2id=label2id
).to('cuda' if torch.cuda.is_available() else 'cpu')
model_name = f'{base_model}_lr{lr}_epochs{epochs}'
model_path = f"generated_models/{model_name}"
print(f'--------Training model {model_name}--------')
training_args = TrainingArguments(
    output_dir=model_path,
    learning_rate=lr,
    gradient_accumulation_steps=int(bs/physical_batch_size),
    per_device_train_batch_size=int(physical_batch_size),
    per_device_eval_batch_size=int(physical_batch_size),
    num_train_epochs=epochs,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    logging_strategy="epoch",
    save_strategy="no",
    metric_for_best_model=metric_name,
    load_best_model_at_end=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

print('------TRAINING FINISHED----------')
cur_path = os.path.split(os.path.realpath(__file__))[0]
datafile = os.path.join(cur_path, model_path)
if not os.path.exists(datafile):
    os.mkdir(datafile)
    #trainer.save_model(datafile)
metrics_values = {f'val_{metric_name}':[],'val_loss':[],'tra_loss':[]}
for metrics in trainer.state.log_history:
    if f'eval_{metric_name}' in metrics:
        metrics_values['val_loss'].append(round(metrics['eval_loss'],3))
        metrics_values[f'val_{metric_name}'].append(round(metrics[f'eval_{metric_name}'],3))
    elif 'loss' in metrics :
        metrics_values['tra_loss'].append(round(metrics['loss'],3))

def print_metrics():
    out = model_name + '\n'
    out += '\t'.join(['epoch'] + [str(i+1) for i in range(epochs)])
    for m in metrics_values:
        out += '\n' + '\t'.join([m]+[str(i) for i in metrics_values[m]])
    eval_res = max(metrics_values[f'val_{metric_name}'])
    print(eval_res)
    out += f'\nBest {metric_name} on evaluation is {eval_res}'
    test_res = trainer.evaluate(tokenized_dataset["test"])
    print(test_res)
    out += f'\nBest {metric_name} on testing is {round(test_res[f"eval_{metric_name}"],3)}'
    return out

with open(datafile+'/metrics.csv','w') as f:
    f.write(print_metrics())

