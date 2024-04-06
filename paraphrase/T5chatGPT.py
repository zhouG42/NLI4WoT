# pip install transformers

import numpy as np
import pandas as pd
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base")
device = "cpu"
model = AutoModelForSeq2SeqLM.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base").to(device)

def paraphrase(
    question,
    num_beams=5,
    num_beam_groups=5,
    num_return_sequences=5,
    repetition_penalty=10.0,
    diversity_penalty=3.0,
    no_repeat_ngram_size=2,
    temperature=0.7,
    max_length=128
):
    input_ids = tokenizer(
        f'paraphrase: {question}',
        return_tensors="pt", padding="longest",
        max_length=max_length,
        truncation=True,
    ).input_ids

    outputs = model.generate(
        input_ids, temperature=temperature, repetition_penalty=repetition_penalty,
        num_return_sequences=num_return_sequences, no_repeat_ngram_size=no_repeat_ngram_size,
        num_beams=num_beams, num_beam_groups=num_beam_groups,
        max_length=max_length, diversity_penalty=diversity_penalty
    )

    res = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return res

with open('/content/input.txt', 'a+') as t2:
  with open('/content/input.txt', 'r+') as t1:

    for i in range(35):
      phrase = t1.readline()
      print(t1.tell())
      print(t2.tell())
      print("---------------------")
      print(phrase)

      para1 = paraphrase(phrase)

      if para1 != None:
          for p1 in para1:
            print(p1)
            if p1 != phrase:
              t2.write('\n')
              t2.write(p1)
      else:
        pass
