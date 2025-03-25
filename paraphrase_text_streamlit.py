import streamlit as st
import os
import zipfile
import requests
import nltk
import language_tool_python
from transformers import T5ForConditionalGeneration, T5Tokenizer
import subprocess

# Download NLTK data
nltk.download('punkt')

# Set up paths
dataset_path = "/tmp/t5-finetuned-quora-cleaned-temp"

# Function to download the dataset from Kaggle
def download_dataset_from_kaggle():
    if not os.path.exists(dataset_path):
        st.write("Downloading dataset files from Kaggle...")
        
        # Set up Kaggle API credentials (to be provided via environment variables in Hugging Face Spaces)
        os.environ['KAGGLE_USERNAME'] = os.getenv('KAGGLE_USERNAME')
        os.environ['KAGGLE_KEY'] = os.getenv('KAGGLE_KEY')
        
        # Install kaggle package if not already installed
        subprocess.run(["pip", "install", "kaggle"])
        
        # Download the dataset
        dataset_slug = "anasdahmani/t5-finetuned-quora-cleaned-temp"
        subprocess.run(["kaggle", "datasets", "download", "-d", dataset_slug, "-p", "/tmp"])
        
        # Unzip the dataset
        zip_path = "/tmp/t5-finetuned-quora-cleaned-temp.zip"
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall("/tmp/t5-finetuned-quora-cleaned-temp")
        
        # Remove the zip file to save space
        os.remove(zip_path)
        
        st.write("Dataset files downloaded and extracted.")

# Load the model and tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    download_dataset_from_kaggle()
    model = T5ForConditionalGeneration.from_pretrained(dataset_path).half()  # Use half-precision to reduce memory usage
    tokenizer = T5Tokenizer.from_pretrained(dataset_path)
    return model, tokenizer

# Load grammar correction tool
@st.cache_resource
def load_grammar_tool():
    return language_tool_python.LanguageTool('en-US')

# Function to split text into sentences
def split_into_sentences(text, mode, num_sentences):
    sentences = nltk.sent_tokenize(text)
    if mode == "Manual":
        if num_sentences <= 0:
            num_sentences = 1
        avg_len = max(1, len(sentences) // num_sentences)
        grouped_sentences = [' '.join(sentences[i:i + avg_len]) for i in range(0, len(sentences), avg_len)]
    else:
        grouped_sentences = sentences
    return grouped_sentences

# Function to paraphrase text
def paraphrase_text(model, tokenizer, text, creativity_level, num_sentences, splitting_mode, apply_grammar_correction, mother_tongue):
    grouped_sentences = split_into_sentences(text, splitting_mode, num_sentences)
    
    st.write(f"Number of sentences: {len(grouped_sentences)}")
    st.write("Sentences:", grouped_sentences)
    
    paraphrased_sentences = []
    for sentence in grouped_sentences:
        if not sentence.strip():
            continue
        input_text = f"paraphrase: {sentence}"
        input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
        
        # Adjust generation parameters based on creativity level
        if creativity_level == "Low":
            temperature, top_k, top_p = 0.7, 50, 0.9
        elif creativity_level == "Medium":
            temperature, top_k, top_p = 1.0, 70, 0.95
        else:  # High
            temperature, top_k, top_p = 1.2, 100, 0.98
        
        outputs = model.generate(
            input_ids,
            max_length=512,
            num_beams=5,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            no_repeat_ngram_size=2,
            early_stopping=True
        )
        paraphrased = tokenizer.decode(outputs[0], skip_special_tokens=True)
        paraphrased_sentences.append(paraphrased)
    
    paraphrased_text = ' '.join(paraphrased_sentences)
    
    if apply_grammar_correction:
        tool = load_grammar_tool()
        matches = tool.check(paraphrased_text)
        paraphrased_text = tool.correct(paraphrased_text)
    
    return paraphrased_text

# Streamlit app
st.title("AI Text Paraphrasing Tool")

# Input text
input_text = st.text_area("Enter the text you want to paraphrase:", height=200)

# Options
splitting_mode = st.selectbox("Splitting Mode:", ["Automatic", "Manual"])
num_sentences = st.slider("Number of Sentences (if Manual):", 1, 10, 3)
creativity_level = st.selectbox("Creativity Level:", ["Low", "Medium", "High"])
apply_grammar_correction = st.checkbox("Apply Grammar Correction", value=True)
mother_tongue = st.selectbox("Mother Tongue (for grammar correction):", ["Arabic", "English", "Spanish", "French", "German"])

# Paraphrase button
if st.button("Paraphrase"):
    if input_text.strip():
        with st.spinner("Loading model and paraphrasing..."):
            model, tokenizer = load_model_and_tokenizer()
            paraphrased_text = paraphrase_text(
                model, tokenizer, input_text, creativity_level, num_sentences,
                splitting_mode, apply_grammar_correction, mother_tongue
            )
        
        st.write("### Original Text:")
        st.write(input_text)
        st.write("### Paraphrased Text:")
        st.write(paraphrased_text)
    else:
        st.error("Please enter some text to paraphrase.")
