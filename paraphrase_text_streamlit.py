# paraphrase_text_streamlit.py
import streamlit as st
import nltk
from transformers import T5ForConditionalGeneration, T5Tokenizer
import re
import language_tool_python

nltk.download('punkt')

# Load the fine-tuned model and tokenizer (on CPU)
model_path = "anasdahmani/t5-finetuned-quora-dataset"  # We'll update this path after uploading to GitHub
model = T5ForConditionalGeneration.from_pretrained(model_path)
tokenizer = T5Tokenizer.from_pretrained(model_path)

# Initialize LanguageTool for grammar correction with motherTongue set to Arabic
tool = language_tool_python.LanguageTool('en-US', motherTongue='ar')

def clean_text(text):
    text = re.sub(r'\s+', ' ', text.strip())
    text = re.sub(r'([a-zA-Z])([A-Z])', r'\1. \2', text)
    text = re.sub(r'([.!?])\s*([A-Za-z])', r'\1 \2', text)
    if not text.endswith(('.', '!', '?')):
        text += '.'
    return text

def dynamic_sentence_split(sentences, min_words=5, max_words=20):
    new_sentences = []
    i = 0
    while i < len(sentences):
        sentence = sentences[i]
        words = sentence.split()
        
        if len(words) > max_words:
            parts = re.split(r'(,\s*(and|but|or|nor|for|yet|so)\s*|;\s*)', sentence)
            if len(parts) > 1:
                temp_sentence = ""
                for part in parts:
                    if part.strip():
                        temp_sentence += part
                        temp_words = temp_sentence.split()
                        if len(temp_words) >= min_words or not temp_sentence.endswith((',', ';')):
                            new_sentences.append(temp_sentence.strip())
                            temp_sentence = ""
                if temp_sentence:
                    new_sentences.append(temp_sentence.strip())
                i += 1
                continue
        
        if len(words) < min_words and i + 1 < len(sentences):
            combined_sentence = sentence.rstrip('.!?') + " " + sentences[i + 1].lstrip()
            sentences[i + 1] = combined_sentence
            i += 1
        else:
            new_sentences.append(sentence)
            i += 1
    
    return new_sentences

def manual_sentence_split(text, num_sentences):
    words = text.split()
    total_words = len(words)
    
    if num_sentences <= 0:
        raise ValueError("Number of sentences must be greater than 0.")
    
    num_sentences = min(num_sentences, total_words)
    
    words_per_sentence = total_words // num_sentences
    remainder = total_words % num_sentences
    
    sentences = []
    start_idx = 0
    for i in range(num_sentences):
        extra_word = 1 if i < remainder else 0
        num_words = words_per_sentence + extra_word
        
        end_idx = start_idx + num_words
        sentence_words = words[start_idx:end_idx]
        sentence = " ".join(sentence_words)
        
        if not sentence.endswith(('.', '!', '?')):
            sentence += '.'
        sentences.append(sentence)
        
        start_idx = end_idx
    
    return sentences

def get_generation_params(creativity_level):
    if creativity_level.lower() == "low":
        return {"temperature": 0.7, "top_k": 50, "top_p": 0.7}
    elif creativity_level.lower() == "high":
        return {"temperature": 1.2, "top_k": 150, "top_p": 0.95}
    else:  # Medium (default)
        return {"temperature": 1.0, "top_k": 100, "top_p": 0.9}

def correct_grammar(text):
    matches = tool.check(text)
    corrected_text = language_tool_python.utils.correct(text, matches)
    return corrected_text

def paraphrase_text(input_text, split_mode="automatic", num_sentences=None, creativity_level="medium", apply_grammar_correction=True):
    cleaned_text = clean_text(input_text)
    
    if split_mode.lower() == "manual" and num_sentences is not None:
        sentences = manual_sentence_split(cleaned_text, num_sentences)
    else:
        initial_sentences = nltk.sent_tokenize(cleaned_text)
        sentences = dynamic_sentence_split(initial_sentences)
    
    st.write(f"Number of sentences: {len(sentences)}")
    st.write("Sentences:", sentences)
    
    gen_params = get_generation_params(creativity_level)
    
    paraphrased_sentences = []
    for sentence in sentences:
        input_text = f"rewrite: {sentence}"
        inputs = tokenizer(input_text, return_tensors="pt", max_length=256, padding="max_length", truncation=True)
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=256,
            do_sample=True,
            top_k=gen_params["top_k"],
            top_p=gen_params["top_p"],
            temperature=gen_params["temperature"],
            repetition_penalty=3.0
        )
        paraphrased_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        paraphrased_sentences.append(paraphrased_text)
    
    paraphrased_text = " ".join(paraphrased_sentences)
    
    if apply_grammar_correction:
        corrected_text = correct_grammar(paraphrased_text)
        return corrected_text
    return paraphrased_text

# Streamlit app interface
st.title("AI Text Paraphrasing Tool")
st.write("Enter your text below to paraphrase it using an AI-powered tool. Customize the settings to fit your needs!")

# Input text
input_text = st.text_area("Input Text", placeholder="Paste your text here...", height=150)

# Splitting mode
split_mode = st.radio("Sentence Splitting Mode", ["Automatic", "Manual"])

# Number of sentences (visible only if Manual is selected)
num_sentences = None
if split_mode == "Manual":
    num_sentences = st.number_input("Number of Sentences", min_value=1, value=5)

# Creativity level
creativity_level = st.selectbox("Creativity Level", ["Low", "Medium", "High"])

# Grammar correction toggle
grammar_correction = st.checkbox("Apply Grammar Correction", value=True)

# Paraphrase button
if st.button("Paraphrase"):
    if not input_text.strip():
        st.error("No text provided. Please provide a text to paraphrase.")
    else:
        with st.spinner("Paraphrasing your text..."):
            paraphrased_text = paraphrase_text(
                input_text,
                split_mode=split_mode,
                num_sentences=num_sentences,
                creativity_level=creativity_level,
                apply_grammar_correction=grammar_correction
            )
        
        st.subheader("Results")
        st.write(f"**Original Text:**\n{input_text}\n")
        if grammar_correction:
            st.write(f"**Paraphrased Text (with Grammar Correction, motherTongue=Arabic):**\n{paraphrased_text}")
        else:
            st.write(f"**Paraphrased Text (without Grammar Correction):**\n{paraphrased_text}")