import joblib
#import preprocess_style
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import streamlit as st
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer
import unicodedata

#out_cantillation = r"all_cantillation.pkl"
#cantillation = joblib.load(out_cantillation)
#cantillation.add("p")

device = "cuda:0"
device = "cpu"

st.title("TAAMI: Biblical Cantillation Prediction")


label_encoding_dict = joblib.load("label_encoding_dict.pkl") #{v:i for i, v in enumerate(cantillation)}
idx2label = {v:k for k,v in label_encoding_dict.items()}

tokenizer = AutoTokenizer.from_pretrained("onlplab/alephbert-base")
model = AutoModelForTokenClassification.from_pretrained("./checkpoint-5000", num_labels=len(label_encoding_dict))

default_text = "בראשית ברא אלהים את השמים ואת הארץ"


def predict(text, model=model, tokenizer=tokenizer):
    tokens = tokenizer.tokenize(text)
    tokenizer(text)
    res = model(torch.Tensor(tokenizer(text)["input_ids"]).view(1, -1).long().to(device))
    preds = res.logits.argmax(axis=-1).view(-1).tolist()
    result = [unicodedata.name(idx2label[el]) for el in preds]
    result = result[1:-1]
    return result, tokens
    
    
def predict2(text, model=model, tokenizer=tokenizer):
    tokens = tokenizer.tokenize(text)
    tokenizer(text)
    res = model(torch.Tensor(tokenizer(text)["input_ids"]).view(1, -1).long().to(device))
    preds = res.logits.argmax(axis=-1).view(-1).tolist()
    result = [idx2label[el] for el in preds]
    result = result[1:-1]

    sent = ""
    for i, token in enumerate(tokens):
      if "##" in token:
        if sent[-1] == " ":
          sent = sent[:-2] + token[2:] + result[i] + " "
        elif sent[-1] == "-":
          sent = sent[:-1] + token[2:] + result[i] + " "
        if result[i] == "z":
          sent = sent[:-2] + "-"
      elif result[i] == "z":
        sent += token + "-"
      else:
        if "YETIV" in unicodedata.name(result[i]):
            sent += token[:1] + result[i] + token[1:] + " "
        else:
            sent += token + result[i] + " "
    return sent


st.markdown("""
<style>
textarea {
  unicode-bidi:bidi-override;
  direction: RTL;
}
</style>
    """, unsafe_allow_html=True)

st.markdown("""
<style>
p {
  unicode-bidi:bidi-override;
  direction: RTL;
  font-size: 30px;
  font-family: 'David Libre';
}
</style>
    """, unsafe_allow_html=True)
    
label = "Enter a verse here (no punctuation or vowels):"

#inp_text = st.text_input(label, value=default_text)
text = st.text_area(label, value=default_text)


#import eli5
#ex = eli5.explain_prediction(model, inp_text, vec=vec, target_names=labels)
#exhtml = eli5.formatters.html.format_as_html(ex)
#res = exhtml.replace("eli5-weights", "eli5weights").replace("\n", " ")
res = predict2(text)
#res = predict(text)

st.markdown(res, unsafe_allow_html=True)