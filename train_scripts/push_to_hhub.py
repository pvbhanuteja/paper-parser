# %%

from datasets import load_dataset 
from PIL import Image, ImageDraw, ImageFont
import torch
import numpy as np
from transformers import AutoModelForTokenClassification, AutoProcessor

# %%
dataset = load_dataset("pvbhanuteja/funsd_dataset")
from datasets.features import ClassLabel

features = dataset["train"].features
column_names = dataset["train"].column_names
image_column_name = "image"
text_column_name = "tokens"
boxes_column_name = "bboxes"
label_column_name = "ner_tags"

# In the event the labels are not a `Sequence[ClassLabel]`, we will need to go through the dataset to get the
# unique labels.
def get_label_list(labels):
    unique_labels = set()
    for label in labels:
        unique_labels = unique_labels | set(label)
    label_list = list(unique_labels)
    label_list.sort()
    return label_list

if isinstance(features[label_column_name].feature, ClassLabel):
    label_list = features[label_column_name].feature.names
    # No need to convert the labels since they are already ints.
    id2label = {k: v for k,v in enumerate(label_list)}
    label2id = {v: k for k,v in enumerate(label_list)}
else:
    label_list = get_label_list(dataset["train"][label_column_name])
    id2label = {k: v for k,v in enumerate(label_list)}
    label2id = {v: k for k,v in enumerate(label_list)}
num_labels = len(label_list)

# %%
processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base",
                                          apply_ocr=False)

model = AutoModelForTokenClassification.from_pretrained("/mnt/nvme-data1/bhanu/code-bases/papertoweb/train_scripts/funsd-finetuned/best_model",
                                                        id2label=id2label,
                                                        label2id=label2id)

model.push_to_hub("llmv3-base-funsd")
# %%
