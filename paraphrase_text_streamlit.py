import streamlit as st
import os
import nltk
import language_tool_python
from transformers import T5ForConditionalGeneration, T5Tokenizer
import logging
import gdown
import time

# Check if sentencepiece is installed

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download NLTK data
logger.info("Downloading NLTK punkt and punkt_tab data...")
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)  # Added to download punkt_tab
logger.info("NLTK punkt and punkt_tab data downloaded successfully.")

# Set up paths
local_model_path = "/mount/src/paraphrase-tool-app/t5-finetuned-quora-cleaned-temp"  # Local directory to store the model files
os.makedirs(local_model_path, exist_ok=True)

# List of expected files
expected_files = [
    "config.json",
    "spiece.model",
    "model.safetensors",  # We know this is the weights file based on the logs
]

# Download model files from Google Drive
def download_model_files():
    logger.info("Downloading model files from Google Drive...")
    try:
        # Replace with your Google Drive folder ID
        folder_id = "1Ab-vaLKkrv0eflaVS-F1s1C3-IWaXR1I"  # Extracted from the logs
        url = f"https://drive.google.com/drive/folders/1dCTBzHbQcJca9Bub2gyYgrFfkW5Qj59l?usp=drive_link"
        gdown.download_folder(url, output=local_model_path, quiet=False)
        logger.info("Model files downloaded successfully to %s", local_model_path)
        st.write("Model files downloaded successfully to", local_model_path)

        # List files to confirm
        files = os.listdir(local_model_path)
        logger.info("Files in %s: %s", local_model_path, files)
        st.write("Files in model directory:", files)

        # Check for required files
        for required_file in expected_files:
            if required_file not in files:
                st.error(f"{required_file} is missing in the downloaded files!")
                raise FileNotFoundError(f"{required_file} is missing in {local_model_path}")
    except Exception as e:
        error_msg = f"Error downloading model files: {str(e)}"
        logger.error(error_msg)
        st.error(error_msg)
        raise

# Download the model files if they don't already exist
if not all(os.path.exists(os.path.join(local_model_path, f)) for f in expected_files):
    download_model_files()

# Wait until model.safetensors is fully downloaded
def wait_for_model_file():
    model_file = os.path.join(local_model_path, "model.safetensors")
    timeout = 300  # Wait up to 5 minutes
    start_time = time.time()
    while not os.path.exists(model_file):
        elapsed_time = time.time() - start_time
        if elapsed_time > timeout:
            raise TimeoutError(f"Timed out waiting for {model_file} to download after {timeout} seconds")
        logger.info("Waiting for model.safetensors to download...")
        st.write("Waiting for model.safetensors to download...")
        time.sleep(5)  # Check every 5 seconds

wait_for_model_file()

# Load the model and tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    logger.info("Loading model and tokenizer from %s...", local_model_path)
    try:
        # Load the model and tokenizer from the local directory
        model = T5ForConditionalGeneration.from_pretrained(local_model_path, device_map="auto")
        tokenizer = T5Tokenizer.from_pretrained(local_model_path)
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
    try:
        tool = language_tool_python.LanguageTool('en-US')
        logger.info("Grammar correction tool loaded successfully.")
        return tool
    except Exception as e:
        error_msg = f"Failed to load grammar tool. Error: {str(e)}"
        logger.error(error_msg)
        raise

# Function to split text into sentences
def split_into_sentences(text, mode, num_sentences):
    try:
        sentences = nltk.sent_tokenize(text)
    except LookupError as e:
        st.error(f"Failed to tokenize text: {str(e)}")
        logger.error(f"Failed to tokenize text: {str(e)}")
        raise
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
        
        try:
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
        except Exception as e:
            logger.error(f"Error during paraphrasing: {str(e)}")
            st.error(f"Error paraphrasing sentence: {sentence}. Error: {str(e)}")
            paraphrased_sentences.append(sentence)  # Fallback to original sentence
    
    paraphrased_text = ' '.join(paraphrased_sentences)
    
    if apply_grammar_correction:
        try:
            tool = load_grammar_tool()
            matches = tool.check(paraphrased_text)
            paraphrased_text = tool.correct(paraphrased_text)
        except Exception as e:
            logger.error(f"Error during grammar correction: {str(e)}")
            st.warning("Grammar correction failed. Returning text without corrections.")
    
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
