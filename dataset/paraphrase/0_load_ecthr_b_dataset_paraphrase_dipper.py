import os
import json
import string
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["HF_HUB_MAX_WORKERS"] = "1"
os.environ["HF_HUB_TIMEOUT"] = "180"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

'''
from huggingface_hub import snapshot_download
snapshot_download("kalpeshk2011/dipper-paraphraser-xxl",
                  resume_download=True, max_workers=1)
snapshot_download("google/t5-v1_1-xxl",
                  resume_download=True, max_workers=1)
'''

import torch
import re
import nltk
import pysbd
import time
from datasets import load_dataset
from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    BitsAndBytesConfig,
)
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
from evaluate import load

nltk.download('punkt')
nltk.download('punkt_tab')

ECHR_ARTICLES = [
    "Article 2",
    "Article 3",
    "Article 5",
    "Article 6",
    "Article 8",
    "Article 9",
    "Article 10",
    "Article 11",
    "Article 14",
    "Article 1 of Protocol 1"
]

# Load tokenizer for LLaMA
max_seq_length = 4096 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

_, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit", #"unsloth/Llama-3.2-3B-Instruct", # or choose "unsloth/Llama-3.2-1B-Instruct"
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
        # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
    )
tokenizer.pad_token_id = tokenizer.eos_token_id


bertscore = load("bertscore")

# Load the dataset
dataset = load_dataset("lex_glue", "ecthr_b")

# Sample sizes for each split
sample_sizes = {
    "train": 9000,
    "validation": 1000,
    "test": 1000
}

# Ensure output directory exists
output_dir_json = os.path.join("preprocessed", "new_combination", "json", "paraphrase")
os.makedirs(output_dir_json, exist_ok=True)
output_dir_jsonl = os.path.join("preprocessed", "new_combination", "jsonl")
os.makedirs(output_dir_jsonl, exist_ok=True)

# Function to check token length
def is_valid_ntokens(example):
    text = " ".join(example["text"])

    labels_map = list(string.ascii_uppercase[:len(ECHR_ARTICLES)])
    choices = [f"({label}) {article}" for label, article in zip(labels_map, ECHR_ARTICLES)]

    question = text.strip() + "\nChoices:\n" + "\n".join(choices) + "\nAnswer:"

    tokenized = tokenizer.encode(question, add_special_tokens=False)

    return len(tokenized) <= 4050
    
def chunk_by_tokens(sentences, tokenizer, max_tokens, add_special_tokens=True, overlap_sents=0):
    """
    Greedy packer: builds chunks of sentences whose tokenized length
    does not exceed `max_tokens`. Optional sentence overlap between chunks.
    """
    chunks, curr = [], []
    for s in sentences:
        if not curr:
            # Start a fresh chunk with this sentence (truncate singletons if needed)
            toks = tokenizer(s, add_special_tokens=add_special_tokens).input_ids
            if len(toks) <= max_tokens:
                curr = [s]
            else:
                # Fallback: hard-wrap a single overlong sentence
                # (rare for normal prose, but robust)
                ids = tokenizer(s, add_special_tokens=add_special_tokens).input_ids
                # keep chopping until all ids are consumed
                start = 0
                while start < len(ids):
                    end = min(start + max_tokens, len(ids))
                    piece = tokenizer.decode(ids[start:end], skip_special_tokens=True)
                    chunks.append(piece.strip())
                    start = end
                curr = []
        else:
            candidate = " ".join(curr + [s])
            cand_len = len(tokenizer(candidate, add_special_tokens=add_special_tokens).input_ids)
            if cand_len <= max_tokens:
                curr.append(s)
            else:
                # flush current chunk
                chunks.append(" ".join(curr).strip())
                # optionally add sentence overlap to next chunk
                if overlap_sents > 0:
                    curr = curr[-overlap_sents:] + [s]
                    # if even with overlap it's too big, reduce overlap
                    while True:
                        cand_len = len(tokenizer(" ".join(curr), add_special_tokens=add_special_tokens).input_ids)
                        if cand_len <= max_tokens or not curr:
                            break
                        curr = curr[1:]  # drop earliest overlapped sentence
                else:
                    curr = [s]
                # handle the (rare) case new curr already exceeds limit
                cand_len = len(tokenizer(" ".join(curr), add_special_tokens=add_special_tokens).input_ids)
                if cand_len > max_tokens:
                    # flush single sentence as in the singleton fallback
                    toks = tokenizer(curr[0], add_special_tokens=add_special_tokens).input_ids
                    start = 0
                    while start < len(toks):
                        end = min(start + max_tokens, len(toks))
                        piece = tokenizer.decode(toks[start:end], skip_special_tokens=True)
                        chunks.append(piece.strip())
                        start = end
                    curr = []
                    
    if curr:
        chunks.append(" ".join(curr).strip())
    return chunks
    
class DipperParaphraser(object):
    #def __init__(self, model="kalpeshk2011/dipper-paraphraser-xxl", verbose=True):
    def __init__(
        self,
        model_id: str = "kalpeshk2011/dipper-paraphraser-xxl",
        verbose: bool = True,
        use_4bit: bool = False,          # flip to False to turn off quantization
        compute_dtype: str = "bfloat16" # or "float16" if your GPU lacks bf16
    ):
        time1 = time.time()
        
        self.tokenizer = AutoTokenizer.from_pretrained('google/t5-v1_1-xxl')

        quant_config = None
        if use_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=getattr(__import__("torch"), compute_dtype),
            )

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id,
            quantization_config=quant_config,
            device_map="cuda",
        )
        if verbose:
            print(f"{model_id} model loaded in {time.time() - time1}")
        self.model.cuda()
        self.model.eval()

    def paraphrase(self, input_text, lex_diversity, order_diversity, prefix="", sent_interval=3, **kwargs):
        """Paraphrase a text using the DIPPER model.

        Args:
            input_text (str): The text to paraphrase. Make sure to mark the sentence to be paraphrased between <sent> and </sent> blocks, keeping space on either side.
            lex_diversity (int): The lexical diversity of the output, choose multiples of 20 from 0 to 100. 0 means no diversity, 100 means maximum diversity.
            order_diversity (int): The order diversity of the output, choose multiples of 20 from 0 to 100. 0 means no diversity, 100 means maximum diversity.
            **kwargs: Additional keyword arguments like top_p, top_k, max_new_tokens #max_length.
        """
        assert lex_diversity in [0, 20, 40, 60, 80, 100], "Lexical diversity must be one of 0, 20, 40, 60, 80, 100."
        assert order_diversity in [0, 20, 40, 60, 80, 100], "Order diversity must be one of 0, 20, 40, 60, 80, 100."

        lex_code = int(100 - lex_diversity)
        order_code = int(100 - order_diversity)

        input_text = " ".join(input_text.split())
        sentences = sent_tokenize(input_text)
        prefix = " ".join(prefix.replace("\n", " ").split())
        output_text = ""

        for sent_idx in range(0, len(sentences), sent_interval):
            curr_sent_window = " ".join(sentences[sent_idx:sent_idx + sent_interval])
            final_input_text = f"lexical = {lex_code}, order = {order_code}"
            if prefix:
                final_input_text += f" {prefix}"
            final_input_text += f" <sent> {curr_sent_window} </sent>"

            final_input = self.tokenizer([final_input_text], return_tensors="pt")
            final_input = {k: v.cuda() for k, v in final_input.items()}

            with torch.inference_mode():
                outputs = self.model.generate(**final_input, **kwargs)
            outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            prefix += " " + outputs[0]
            output_text += " " + outputs[0]

        return output_text

dp = DipperParaphraser()
seg = pysbd.Segmenter(language="en", clean=False)
MAX_INPUT_TOKENS = 512

# Process each split
for split, size in sample_sizes.items():
    filtered = dataset[split].filter(is_valid_ntokens)
    small_split = filtered.shuffle(seed=42).select(range(min(size, len(filtered))))

    converted_samples = []
    for example in tqdm(small_split, desc=f"Paraphrasing {split}"):
        text = " ".join(example["text"])
        labels = example.get("labels", [])  # indices of matching articles
        
        sentences = seg.segment(text)
        chunks = chunk_by_tokens(sentences, tokenizer, MAX_INPUT_TOKENS, add_special_tokens=True, overlap_sents=0)
        
        paraphrased_sentences = []
        for i, sentence in enumerate(chunks):
            # prefix is the previous sentence (or empty if this is the first one)
            prefix = chunks[i-1] if i > 0 else ""
            
            new_sentence = dp.paraphrase(sentence, lex_diversity=20, order_diversity=20, prefix="", do_sample=True, top_p=0.75, top_k=None, max_new_tokens=MAX_INPUT_TOKENS)
            paraphrased_sentences.append(new_sentence)
        new_text = " ".join(paraphrased_sentences)
        
        # Calculate BERT score
        predictions = [new_text]
        references = [text]
        
        led_results = bertscore.compute(predictions=predictions, references=references, model_type="allenai/led-base-16384", lang="en")
        led_bertscore_res = led_results["f1"][0]

        # Build choice labels: (A) Article 2 ...
        labels_map = list(string.ascii_uppercase[:len(ECHR_ARTICLES)])
        choices = [f"({label}) {article}" for label, article in zip(labels_map, ECHR_ARTICLES)]

        # Question format with choices
        question = text.strip() + "\nChoices:\n" + "\n".join(choices) + "\nECtHR Task B Answer:"
        new_question = new_text.strip() + "\nChoices:\n" + "\n".join(choices) + "\nECtHR Task B Answer:"

        correct_label_indices = labels
        answer_matching = ", ".join(f"({labels_map[i]})" for i in correct_label_indices)
        answer_not_matching = ", ".join(
            f"({labels_map[i]})" for i in range(len(ECHR_ARTICLES)) if i not in correct_label_indices
        )
        
        converted_samples.append([
            {"role": "user", "content": question, "new_content": new_question, "led_bertscore": led_bertscore_res},
            {"role": "assistant", "content": answer_matching}
        ])
        
        # Write to standard JSON
        output_path_json = os.path.join(output_dir_json, f"ecthr_b_paraphrase_{split}_{str(size)}.json")
        with open(output_path_json, "w", encoding="utf-8") as f:
            json.dump({"conversations": converted_samples}, f, indent=2, ensure_ascii=False)

    output_path_jsonl = os.path.join(output_dir_jsonl, f"ecthr_b_paraphrase_{split}_{str(size)}.jsonl")
    small_split.to_json(output_path_jsonl, lines=True)
        



