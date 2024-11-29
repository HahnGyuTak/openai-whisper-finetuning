#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os

os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '6'


# # Fine-Tune Whisper For Multilingual ASR with 🤗 Transformers

# In[2]:


# !pip install --upgrade --quiet pip
# !pip install --upgrade --quiet datasets[audio] transformers==4.45.2 accelerate evaluate jiwer tensorboard gradio


# We strongly advise you to upload model checkpoints directly the [Hugging Face Hub](https://huggingface.co/)
# whilst training. The Hub provides:
# - Integrated version control: you can be sure that no model checkpoint is lost during training.
# - Tensorboard logs: track important metrics over the course of training.
# - Model cards: document what a model does and its intended use cases.
# - Community: an easy way to share and collaborate with the community!
# 
# Linking the notebook to the Hub is straightforward - it simply requires entering your
# Hub authentication token when prompted. Find your Hub authentication token [here](https://huggingface.co/settings/tokens):

# In[3]:


# from huggingface_hub import notebook_login
# # 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
# notebook_login()


# ## Load Dataset

# Using 🤗 Datasets, downloading and preparing data is extremely simple.
# We can download and prepare the Common Voice splits in just one line of code.
# 
# First, ensure you have accepted the terms of use on the Hugging Face Hub: [mozilla-foundation/common_voice_11_0](https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0). Once you have accepted the terms, you will have full access to the dataset and be able to download the data locally.
# 
# Since Hindi is very low-resource, we'll combine the `train` and `validation`
# splits to give approximately 8 hours of training data. We'll use the 4 hours
# of `test` data as our held-out test set:

# In[ ]:


from datasets import load_dataset, DatasetDict

common_voice = DatasetDict()

common_voice["train"] = load_dataset("imTak/korean-speak-Develop", split="train")
common_voice["test"] = load_dataset("imTak/korean-speak-Develop", split="test")

print(common_voice)


# Most ASR datasets only provide input audio samples (`audio`) and the
# corresponding transcribed text (`sentence`). Common Voice contains additional
# metadata information, such as `accent` and `locale`, which we can disregard for ASR.
# Keeping the notebook as general as possible, we only consider the input audio and
# transcribed text for fine-tuning, discarding the additional metadata information:

# ## Prepare Feature Extractor, Tokenizer and Data

# We'll load the feature extractor from the pre-trained checkpoint with the default values:

# In[5]:


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


# ### Combine To Create A WhisperProcessor

# To simplify using the feature extractor and tokenizer, we can _wrap_
# both into a single `WhisperProcessor` class. This processor object
# inherits from the `WhisperFeatureExtractor` and `WhisperProcessor`,
# and can be used on the audio inputs and model predictions as required.
# In doing so, we only need to keep track of two objects during training:
# the `processor` and the `model`:

# In[9]:


from transformers import WhisperProcessor

# processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="Hindi", task="transcribe")
processor = WhisperProcessor.from_pretrained(model_id, language="Korean", task="transcribe")
# processor = WhisperProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)


# ### Prepare Data

# Let's print the first example of the Common Voice dataset to see
# what form the data is in:

# In[10]:


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


# We can apply the data preparation function to all of our training examples using dataset's `.map` method. The argument `num_proc` specifies how many CPU cores to use. Setting `num_proc` > 1 will enable multiprocessing. If the `.map` method hangs with multiprocessing, set `num_proc=1` and process the dataset sequentially.

# In[ ]:


common_voice = common_voice.map(prepare_dataset, remove_columns=common_voice.column_names["train"], num_proc=8)


# ## Training and Evaluation

# Now that we've prepared our data, we're ready to dive into the training pipeline.
# The [🤗 Trainer](https://huggingface.co/transformers/master/main_classes/trainer.html?highlight=trainer)
# will do much of the heavy lifting for us. All we have to do is:
# 
# - Load a pre-trained checkpoint: we need to load a pre-trained checkpoint and configure it correctly for training.
# 
# - Define a data collator: the data collator takes our pre-processed data and prepares PyTorch tensors ready for the model.
# 
# - Evaluation metrics: during evaluation, we want to evaluate the model using the [word error rate (WER)](https://huggingface.co/metrics/wer) metric. We need to define a `compute_metrics` function that handles this computation.
# 
# - Define the training configuration: this will be used by the 🤗 Trainer to define the training schedule.
# 
# Once we've fine-tuned the model, we will evaluate it on the test data to verify that we have correctly trained it
# to transcribe speech in Hindi.

# ### Load a Pre-Trained Checkpoint

# We'll start our fine-tuning run from the pre-trained Whisper `small` checkpoint,
# the weights for which we need to load from the Hugging Face Hub. Again, this
# is trivial through use of 🤗 Transformers!

# In[ ]:


from transformers import WhisperForConditionalGeneration

# model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained(model_id)



# We can disable the automatic language detection task performed during inference, and force the model to generate in Hindi. To do so, we set the [langauge](https://huggingface.co/docs/transformers/en/model_doc/whisper#transformers.WhisperForConditionalGeneration.generate.language)
# and [task](https://huggingface.co/docs/transformers/en/model_doc/whisper#transformers.WhisperForConditionalGeneration.generate.task)
# arguments to the generation config. We'll also set any [`forced_decoder_ids`](https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.generation_utils.GenerationMixin.generate.forced_decoder_ids)
# to None, since this was the legacy way of setting the language and
# task arguments:

# In[ ]:


model.generation_config.language = "ko"
model.generation_config.task = "transcribe"

model.generation_config.forced_decoder_ids = None


# ### Define a Data Collator

# The data collator for a sequence-to-sequence speech model is unique in the sense that it
# treats the `input_features` and `labels` independently: the  `input_features` must be
# handled by the feature extractor and the `labels` by the tokenizer.
# 
# The `input_features` are already padded to 30s and converted to a log-Mel spectrogram
# of fixed dimension by action of the feature extractor, so all we have to do is convert the `input_features`
# to batched PyTorch tensors. We do this using the feature extractor's `.pad` method with `return_tensors=pt`.
# 
# The `labels` on the other hand are un-padded. We first pad the sequences
# to the maximum length in the batch using the tokenizer's `.pad` method. The padding tokens
# are then replaced by `-100` so that these tokens are **not** taken into account when
# computing the loss. We then cut the BOS token from the start of the label sequence as we
# append it later during training.
# 
# We can leverage the `WhisperProcessor` we defined earlier to perform both the
# feature extractor and the tokenizer operations:

# In[ ]:


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

# In[ ]:


data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)


# ### Evaluation Metrics

# In[ ]:


import evaluate

metric = evaluate.load("wer")


# We then simply have to define a function that takes our model
# predictions and returns the WER metric. This function, called
# `compute_metrics`, first replaces `-100` with the `pad_token_id`
# in the `label_ids` (undoing the step we applied in the
# data collator to ignore padded tokens correctly in the loss).
# It then decodes the predicted and label ids to strings. Finally,
# it computes the WER between the predictions and reference labels:

# In[ ]:


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

# In[ ]:


from transformers import Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir="whisper_large_v3_turbo_korean_Develop",  # change to a repo name of your choice
    per_device_train_batch_size=32,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-5,
    resume_from_checkpoint='whisper_large_v3_turbo_korean_Develop/checkpoint-2000',
    restore_callback_states_from_checkpoint=True,
    warmup_steps=500,
    max_steps=8000,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=32,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=1000,
    eval_steps=500,
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

# In[ ]:


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

# In[ ]:


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


trainer.train("whisper_large_v3_turbo_korean_Develop/checkpoint-2000")


# Our best WER is 32.0% - not bad for 8h of training data! We can make our model more accessible on the Hub with appropriate tags and README information.
# You can change these values to match your dataset, language and model
# name accordingly:

# In[ ]:


kwargs = {
    "dataset_tags": "imTak/korean-speak-Develop",
    "dataset": "Develop",  # a 'pretty' name for the training dataset
    "dataset_args": "config: ko, split: test",
    "language": "ko",
    "model_name": "Whisper large v3 turbo Korean-Develop",  # a 'pretty' name for our model
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




