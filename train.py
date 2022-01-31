from prepare import prepare_datasets
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer, AutoConfig
from datasets import concatenate_datasets
from trainer_fb import FBTrainer
import wandb
from utils_fb import postprocess_fb_predictions
import os

def train():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    kfold_tokenized_datasets, l2i, i2l, N_LABELS, data_collator, kfold_examples, tokenizer = prepare_datasets()
    kfold_num = len(kfold_tokenized_datasets)
    for fold in range(kfold_num):
        valid_datasets = kfold_tokenized_datasets[fold]
        train_datasets = concatenate_datasets([kfold_tokenized_datasets[i].flatten_indices() for i in range(kfold_num) if i!=fold])
        valid_examples = kfold_examples[fold]
        config = AutoConfig.from_pretrained('allenai/longformer-base-4096')
        config.num_labels = N_LABELS
        model = AutoModelForTokenClassification.from_pretrained('allenai/longformer-base-4096', config = config)
        training_args = TrainingArguments(
            output_dir = './output/longformer-baseline_fold'+ str(fold),
            evaluation_strategy = 'epoch',
            per_device_train_batch_size = 8,
            per_device_eval_batch_size = 8,
            gradient_accumulation_steps = 2,
            learning_rate = 5e-5,
            weight_decay = 0.01,
            max_grad_norm = 10,
            num_train_epochs = 8,
            warmup_ratio = 0.1,
            logging_strategy = 'steps',
            logging_steps = 50,
            save_strategy = 'epoch',
            save_total_limit = 1,
            seed = 42,
            # eval_steps = 50,
            # save_steps = 50,
            dataloader_num_workers = 2,
            load_best_model_at_end = True,
            metric_for_best_model = 'f1',# need to fix
            group_by_length = True,
            report_to = 'wandb',
        )
        def post_process_function(eval_examples, eval_datasets, predictions):
            predictions = postprocess_fb_predictions(
                eval_examples = eval_examples,
                eval_datasets = eval_datasets,
                predictions=predictions
            )
            formatted_predictions = [
                {'id': i, 'class':c, 'predictionstring':p} for i,c,p in predictions.items()
            ]
            return formatted_predictions
        def compute_metrics():
            return None
        trainer = FBTrainer(
            model,
            training_args,
            train_dataset=train_datasets,
            eval_dataset=valid_datasets,
            eval_examples=valid_examples,
            data_collator=data_collator,
            tokenizer=tokenizer,
            post_process_function = post_process_function,
            compute_metrics=compute_metrics
        )
        run = wandb.init(project='Feedback-prize', entity='donggunseo', name='longformer-fold'+str(fold))
        trainer.train()
        run.finish()
        trainer.save_model('best_model/longformer-baseline_best')