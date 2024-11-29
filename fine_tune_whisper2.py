#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os

os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '7'


from datasets import load_dataset, DatasetDict

common_voice = DatasetDict()

common_voice["train"] = load_dataset("imTak/Economy", split="train")
common_voice["test"] = load_dataset("imTak/Economy", split="test")

print(common_voice)

from transformers import WhisperFeatureExtractor

# feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
model_id = "imTak/whisper_large_v3_ko_ft_ft"
feature_extractor = WhisperFeatureExtractor.from_pretrained(model_id)


# ### Load WhisperTokenizer

# The Whisper model outputs a sequence of _token ids_. The tokenizer maps each of these token ids to their corresponding text string. For Hindi, we can load the pre-trained tokenizer and use it for fine-tuning without any further modifications. We simply have to
# specify the target language and the task. These arguments inform the
# tokenizer to prefix the language and task tokens to the start of encoded
# label sequences:

# In[6]:


from transformers import WhisperTokenizer, WhisperTokenizerFast

# tokenizer = WhisperTokenizerFast.from_pretrained(model_id, language="Korean", task="transcribe")
tokenizer = WhisperTokenizer.from_pretrained(model_id, language="Korean", task="transcribe")


# In[7]:


print(common_voice["train"][1])


# In[8]:


input_str = common_voice["train"][0]["text"]
labels = tokenizer(input_str).input_ids
decoded_with_special = tokenizer.decode(labels, skip_special_tokens=False)
decoded_str = tokenizer.decode(labels, skip_special_tokens=True)

print(f"Input:                 {input_str}")
print(f"Decoded w/ special:    {decoded_with_special}")
print(f"Decoded w/out special: {decoded_str}")
print(f"Are equal:             {input_str == decoded_str}")


from transformers import WhisperProcessor

# processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="Hindi", task="transcribe")
processor = WhisperProcessor.from_pretrained(model_id, language="Korean", task="transcribe")
# processor = WhisperProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)


print(common_voice["train"][0])


# In[11]:


def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["name"]

    # compute log-Mel input features from input audio array
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids
    batch["labels"] = tokenizer(batch["text"]).input_ids
    return batch


common_voice = common_voice.map(prepare_dataset, remove_columns=common_voice.column_names["train"], num_proc=8)


from transformers import WhisperForConditionalGeneration

# model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained(model_id)



# We can disable the automatic language detection task performed during inference, and force the model to generate in Hindi. To do so, we set the [langauge](https://huggingface.co/docs/transformers/en/model_doc/whisper#transformers.WhisperForConditionalGeneration.generate.language)
# and [task](https://huggingface.co/docs/transformers/en/model_doc/whisper#transformers.WhisperForConditionalGeneration.generate.task)
# arguments to the generation config. We'll also set any [`forced_decoder_ids`](https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.generation_utils.GenerationMixin.generate.forced_decoder_ids)
# to None, since this was the legacy way of setting the language and
# task arguments:

# In[14]:


model.generation_config.language = "ko"
model.generation_config.task = "transcribe"

model.generation_config.forced_decoder_ids = None

import torch

from dataclasses import dataclass
from typing import Any, Dict, List, Union

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


# Let's initialise the data collator we've just defined:

# In[16]:


data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)


# ### Evaluation Metrics

# In[17]:


import evaluate

metric = evaluate.load("wer")


# We then simply have to define a function that takes our model
# predictions and returns the WER metric. This function, called
# `compute_metrics`, first replaces `-100` with the `pad_token_id`
# in the `label_ids` (undoing the step we applied in the
# data collator to ignore padded tokens correctly in the loss).
# It then decodes the predicted and label ids to strings. Finally,
# it computes the WER between the predictions and reference labels:

# In[18]:


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


# ### Define the Training Configuration

# In the final step, we define all the parameters related to training. For more detail on the training arguments, refer to the Seq2SeqTrainingArguments [docs](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Seq2SeqTrainingArguments).

# In[19]:


from transformers import Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir="whisper_large_v3_ko_ft_ft_Economy",  # change to a repo name of your choice
    per_device_train_batch_size=32,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    resume_from_checkpoint='whisper_large_v3_ko_ft_ft_Economy/checkpoint-4000',
    restore_callback_states_from_checkpoint=True,
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=8000,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=32,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=2000,
    eval_steps=2000,
    logging_steps=100,
    report_to=["wandb"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=True,
)


# **Note**: if one does not want to upload the model checkpoints to the Hub,
# set `push_to_hub=False`.

# We can forward the training arguments to the 🤗 Trainer along with our model,
# dataset, data collator and `compute_metrics` function:

# In[20]:


from transformers import Seq2SeqTrainer

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=common_voice["train"],
    eval_dataset=common_voice["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor,
)


# We'll save the processor object once before starting training. Since the processor is not trainable, it won't change over the course of training:

# In[21]:


processor.save_pretrained(training_args.output_dir)


# ### Training

# Training will take approximately 5-10 hours depending on your GPU or the one
# allocated to this Google Colab. If using this Google Colab directly to
# fine-tune a Whisper model, you should make sure that training isn't
# interrupted due to inactivity. A simple workaround to prevent this is
# to paste the following code into the console of this tab (_right mouse click_
# -> _inspect_ -> _Console tab_ -> _insert code_).

# ```javascript
# function ConnectButton(){
#     console.log("Connect pushed");
#     document.querySelector("#top-toolbar > colab-connect-button").shadowRoot.querySelector("#connect").click()
# }
# setInterval(ConnectButton, 60000);
# ```

# The peak GPU memory for the given training configuration is approximately 15.8GB.
# Depending on the GPU allocated to the Google Colab, it is possible that you will encounter a CUDA `"out-of-memory"` error when you launch training.
# In this case, you can reduce the `per_device_train_batch_size` incrementally by factors of 2
# and employ [`gradient_accumulation_steps`](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Seq2SeqTrainingArguments.gradient_accumulation_steps)
# to compensate.
# 
# To launch training, simply execute:

# In[ ]:


trainer.train("whisper_large_v3_ko_ft_ft_Economy/checkpoint-4000")


# Our best WER is 32.0% - not bad for 8h of training data! We can make our model more accessible on the Hub with appropriate tags and README information.
# You can change these values to match your dataset, language and model
# name accordingly:

# In[ ]:


kwargs = {
    "dataset_tags": "imTak/Economy",
    "dataset": "Economy",  # a 'pretty' name for the training dataset
    "dataset_args": "config: ko, split: test",
    "language": "ko",
    "model_name": "Whisper large v3 turbo Korean-Economy",  # a 'pretty' name for our model
    "finetuned_from": "imTak/whisper_large_v3_ko_ft_ft",
    "tasks": "automatic-speech-recognition",
}


# The training results can now be uploaded to the Hub. To do so, execute the `push_to_hub` command and save the preprocessor object we created:

# In[ ]:


trainer.push_to_hub(**kwargs)


# ```shell
# ct2-transformers-converter --model imTak/whisper_large_v3_ko_ft_ft --output_dir faster-whisper_large_v3_ko_ft_ft     --copy_files tokenizer.json preprocessor_config.json --quantization int8_float32
# ```

# In[ ]:





# In[ ]:




