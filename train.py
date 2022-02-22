from random import seed
from prepare import prepare_datasets
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer, AutoConfig
from model import CustomLongformerForTokenClassification
from datasets import concatenate_datasets
from trainer_fb import FBTrainer
import wandb
from utils_fb import postprocess_fb_predictions2, score_feedback_comp3, seed_everything
import os
import numpy as np
def train():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    kfold_tokenized_datasets, N_LABELS, data_collator, kfold_examples, tokenizer = prepare_datasets()
    kfold_num = len(kfold_tokenized_datasets)
    for fold in range(kfold_num):
        valid_datasets = kfold_tokenized_datasets[fold]
        train_datasets = concatenate_datasets([kfold_tokenized_datasets[i].flatten_indices() for i in range(kfold_num) if i!=fold])
        valid_examples = kfold_examples[fold]
        config = AutoConfig.from_pretrained('allenai/longformer-large-4096')
        config.num_labels = N_LABELS
        model = CustomLongformerForTokenClassification.from_pretrained('allenai/longformer-large-4096', config = config)
        training_args = TrainingArguments(
            output_dir = './output/longformer-large-multidropout_fold'+ str(fold),
            evaluation_strategy = 'epoch',
            per_device_train_batch_size = 2,
            per_device_eval_batch_size = 2,
            gradient_accumulation_steps = 1,
            learning_rate = 1e-5,
            weight_decay = 0.01,
            max_grad_norm = 10,
            num_train_epochs = 6,
            warmup_ratio = 0.1,
            logging_strategy = 'steps',
            logging_steps = 50,
            save_strategy = 'epoch',
            save_total_limit = 1,
            seed = 42,
            dataloader_num_workers = 2,
            load_best_model_at_end = True,
            metric_for_best_model = 'f1',
            group_by_length = True,
            report_to = 'wandb',
        )
        def compute_metrics(eval_examples, eval_preds):
            f1, class_scores = score_feedback_comp3(pred_df = eval_preds, 
            gt_df = eval_examples, 
            return_class_scores=True
            )
            class_scores['eval_f1']=f1
            return class_scores
        trainer = FBTrainer(
            model,
            training_args,
            train_dataset=train_datasets,
            eval_dataset=valid_datasets,
            eval_examples=valid_examples,
            data_collator=data_collator,
            tokenizer=tokenizer,
            post_process_function = postprocess_fb_predictions2,
            compute_metrics=compute_metrics
        )
        run = wandb.init(project='Feedback-prize', entity='donggunseo', name='longformer-large-multidropout-fold'+str(fold))
        trainer.train()
        run.finish()
        trainer.save_model('best_model2/longformer-large-multidropout_fold'+ str(fold))

if __name__ == "__main__":
    seed_everything(42)
    train()
