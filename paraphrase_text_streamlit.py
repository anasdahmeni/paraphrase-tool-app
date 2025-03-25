import streamlit as st
import os
import zipfile
import requests
import nltk
import language_tool_python
from transformers import T5ForConditionalGeneration, T5Tokenizer
import subprocess
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download NLTK data
logger.info("Downloading NLTK punkt data...")
nltk.download('punkt')
logger.info("NLTK punkt data downloaded successfully.")

# Set up paths
dataset_path = "/tmp/t5-finetuned-quora-cleaned-temp"

# Function to download the dataset from Kaggle
def download_dataset_from_kaggle():
    if not os.path.exists(dataset_path):
        st.write("Downloading dataset files from Kaggle...")
        logger.info("Starting dataset download from Kaggle...")
        
        # Retrieve Kaggle API credentials from environment variables
        kaggle_username = os.getenv('KAGGLE_USERNAME')
        kaggle_key = os.getenv('KAGGLE_KEY')
        
        # Log the credentials for debugging
        logger.info(f"KAGGLE_USERNAME value: {kaggle_username}")
        logger.info(f"KAGGLE_KEY: {'Set' if kaggle_key else 'Not set'}")
        
        # Check if credentials are set
        if not kaggle_username or not kaggle_key:
            error_msg = (
                f"Kaggle API credentials are missing or invalid. "
                f"Current KAGGLE_USERNAME: {kaggle_username}, KAGGLE_KEY: {'Set' if kaggle_key else 'Not set'}. "
                "Please ensure KAGGLE_USERNAME and KAGGLE_KEY are set correctly in Hugging Face Spaces. "
                "Go to Settings > Variables and secrets, and verify the values. "
                "KAGGLE_USERNAME should be 'anasdahmani', and KAGGLE_KEY should be your Kaggle API key from kaggle.json."
            )
            st.error(error_msg)
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Set the environment variables for the Kaggle API
        os.environ['KAGGLE_USERNAME'] = kaggle_username
        os.environ['KAGGLE_KEY'] = kaggle_key
        
        # Install kaggle package if not already installed
        logger.info("Installing kaggle package...")
        subprocess.run(["pip", "install", "kaggle"])
        
        # Download the dataset
        dataset_slug = "anasdahmani/t5-finetuned-quora-cleaned-temp"
        logger.info(f"Downloading dataset: {dataset_slug}")
        result = subprocess.run(
            ["kaggle", "datasets", "download", "-d", dataset_slug, "-p", "/tmp"],
            capture_output=True,
            text=True
        )
        
        # Check if the download was successful
        if result.returncode != 0:
            error_msg = f"Failed to download dataset from Kaggle. Error: {result.stderr}"
            st.error(error_msg)
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        st.write("Dataset download output:", result.stdout)
        logger.info(f"Dataset download output: {result.stdout}")
        
        # Unzip the dataset
        zip_path = "/tmp/t5-finetuned-quora-cleaned-temp.zip"
        logger.info(f"Unzipping dataset: {zip_path}")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall("/tmp/t5-finetuned-quora-cleaned-temp")
        
        # Remove the zip file to save space
        os.remove(zip_path)
        
        st.write("Dataset files downloaded and extracted.")
        logger.info("Dataset files downloaded and extracted successfully.")

# Load the model and tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    logger.info("Loading model and tokenizer...")
    download_dataset_from_kaggle()
    try:
        model = T5ForConditionalGeneration.from_pretrained(dataset_path, device_map="auto").half()
        tokenizer = T5Tokenizer.from_pretrained(dataset_path)
        logger.info("Model and tokenizer loaded successfully.")
        return model, tokenizer
    except Exception as e:
        error_msg = f"Failed to load model and tokenizer. Error: {str(e)}"
        st.error(error_msg)
        logger.error(error_msg)
        raise

# Load grammar correction tool
@st.cache_resource
def load_grammar_tool():
    logger.info("Loading grammar correction tool...")
    tool = language_tool_python.LanguageTool('en-US')
    logger.info("Grammar correction tool loaded successfully.")
    return tool

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
    logger.info("Starting paraphrasing process...")
    grouped_sentences = split_into_sentences(text, splitting_mode, num_sentences)
    
    st.write(f"Number of sentences: {len(grouped_sentences)}")
    st.write("Sentences:", grouped_sentences)
    logger.info(f"Number of sentences: {len(grouped_sentences)}")
    logger.info(f"Sentences: {grouped_sentences}")
    
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
    
    logger.info("Paraphrasing completed successfully.")
    return paraphrased_text

# Streamlit app
st.title("AI Text Paraphrasing Tool")

# Input text
input_text = st.text_area("Enter the text you want to paraphrase:", height=200)

# Options
splitting_mode = st.selectbox("Splitting Mode:", ["Automatic", "Manual"])
num_sentences = st.number_input("Number of Sentences (if Manual):", min_value=1, value=3, step=1)
creativity_level = st.selectbox("Creativity Level:", ["Low", "Medium", "High"])
apply_grammar_correction = st.checkbox("Apply Grammar Correction", value=True)
mother_tongue = st.selectbox("Mother Tongue (for grammar correction):", ["Arabic", "English", "Spanish", "French", "German"])

# Paraphrase button
if st.button("Paraphrase"):
    if input_text.strip():
        with st.spinner("Loading model and paraphrasing..."):
            try:
                model, tokenizer = load_model_and_tokenizer()
                paraphrased_text = paraphrase_text(
                    model, tokenizer, input_text, creativity_level, num_sentences,
                    splitting_mode, apply_grammar_correction, mother_tongue
                )
                
                st.write("### Original Text:")
                st.write(input_text)
                st.write("### Paraphrased Text:")
                st.write(paraphrased_text)
            except Exception as e:
                st.error(f"An error occurred during paraphrasing: {str(e)}")
                logger.error(f"Paraphrasing error: {str(e)}")
    else:
        st.error("Please enter some text to paraphrase.")
