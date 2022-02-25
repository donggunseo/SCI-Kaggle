from transformers import DataCollatorForTokenClassification
from dataclasses import dataclass
import torch

@dataclass
class CustomDatacollator(DataCollatorForTokenClassificaton):

        def torch_call(self, features):

            label_name = "label" if "label" in features[0].keys() else "labels"
            labels = [feature[label_name] for feature in features] if label_name in features[0].keys() else None
            batch = dict()
            batch['input_ids'] = [feature['input_ids'] for feature in features]
            batch['attention_mask'] = [feature['attention_mask'] for feature in features]

            batch_max = max([len(ids) for ids in batch['input_ids']])
            padding_side = self.tokenizer.padding_side
            if padding_side == "right":
                batch['input_ids'] = [s + (batch_max - len(s))*[self.tokenizer.pad_token_id] for s in batch['input_ids']]
                batch['attention_mask'] = [s + (batch_max - len(s))*[0] for s in batch['attention_mask']]
            else : 
                batch['input_ids'] = [(batch_max - len(s))*[self.tokenizer.pad_token_id] + s for s in batch['input_ids']]
                batch['attention_mask'] = [(batch_max - len(s))*[0] + s for s in batch['attention_mask']]
            
            if labels is None:
                batch = {k: torch.tensor(v, dtype=torch.int64) for k, v in batch.items()}
                return batch
            if padding_side == "right":
                batch[label_name] = [
                    list(label) + [self.label_pad_token_id] * (batch_max - len(label)) for label in labels
                ]
            else:
                batch[label_name] = [
                    [self.label_pad_token_id] * (batch_max - len(label)) + list(label) for label in labels
                ]
            batch = {k: torch.tensor(v, dtype=torch.int64) for k, v in batch.items()}
            return batch