import os
import streamlit as st
import concurrent.futures
from dotenv import load_dotenv
from transformers import pipeline
from huggingface_hub import InferenceClient
from scrapers.web_scraper import scrape_url
from scrapers.file_scraper import process_file
from nlp_tasks import summarize_text, analyze_sentiment, run_qa

# Set environment and load token
os.environ["TORCH_DISABLE_LAZY_MODULE_LOADING"] = "1"
load_dotenv()
token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Initialize session state
if 'scraped_text' not in st.session_state:
    st.session_state.scraped_text = ""

# Streamlit app config
st.set_page_config(page_title="IntelliScrape", layout="wide")
st.title("IntelliScrape - Your AI Document Assistant")

# HF Model client
client = InferenceClient(
    model="HuggingFaceH4/zephyr-7b-beta",
    token=token
)

# Input method selection
option = st.radio("Choose input method:", ["Scrape Website", "Upload File", "Scrape Multiple URLs"])

# Input logic
if option == "Scrape Website":
    url = st.text_input("Enter Website URL:")
    if st.button("Scrape"):
        with st.spinner("Scraping..."):
            try:
                st.session_state.scraped_text = scrape_url(url)
                st.success("Scraping complete!")
            except Exception as e:
                st.error(f"❌ Error while scraping: {e}")

elif option == "Scrape Multiple URLs":
    urls = st.text_area("Enter URLs (comma-separated):")
    if st.button("Scrape All"):
        url_list = [u.strip() for u in urls.split(",") if u.strip()]
        all_text = ""
        with st.spinner("Scraping all..."):
            for u in url_list:
                try:
                    all_text += scrape_url(u) + "\n\n"
                except Exception as e:
                    st.error(f"Failed to scrape {u}: {e}")
        st.session_state.scraped_text = all_text
        st.success("All websites scraped!")

elif option == "Upload File":
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
    if uploaded_file:
        with st.spinner("Processing..."):
            try:
                st.session_state.scraped_text = process_file(uploaded_file)
                st.success("File processed!")
            except Exception as e:
                st.error(f"❌ Error while processing file: {e}")

st.text_area("Extracted Content", st.session_state.scraped_text, height=200)

# User query
user_query = st.text_input("Ask a question or request summary:")

# Run NLP tasks in parallel
if st.button("Run AI"):
    if st.session_state.scraped_text.strip() == "":
        st.warning("⚠️ Please scrape a website or upload a file before asking.")
    else:
        with st.spinner("Running tasks in parallel..."):
            try:
                text = st.session_state.scraped_text
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future_summary = executor.submit(summarize_text, text)
                    future_sentiment = executor.submit(analyze_sentiment, text)
                    future_qa = executor.submit(run_qa, client, text, user_query)

                    summary = future_summary.result()
                    sentiment = future_sentiment.result()
                    answer = future_qa.result()

                st.subheader("Summary")
                st.write(summary)

                st.subheader("Sentiment")
                st.write(f"Label: {sentiment['label']}, Confidence: {sentiment['score']:.2f}")

                st.subheader("AI Answer")
                st.write(answer)

            except Exception as e:
                st.error(f"AI Error: {e}")
