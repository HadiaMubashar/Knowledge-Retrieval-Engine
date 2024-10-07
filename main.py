import streamlit as st
import time
from models import process_wikipedia_page, generate_answer_llm  # Updated to use LLM-based answer generation


st.set_page_config(
    page_title="Knowledge Retrievel Engine",
    page_icon="logo.png",  
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Knowledge Retrieval Engine")

st.sidebar.title("Add Wikipedia Link")
wiki_link = st.sidebar.text_input("Enter Wikipedia Page URL", placeholder="https://en.wikipedia.org/wiki/Example")

main_placeholder = st.empty()

if st.sidebar.button("Process Wikipedia Page") and wiki_link:
    try:
        main_placeholder.text("Data Loading...Started...")
        time.sleep(1)
        main_placeholder.text("Generating Summary...")

        # Process the Wikipedia page and get summary and chunks
        title, summary, chunks = process_wikipedia_page(wiki_link)
        if not title:
            st.error("The Wikipedia page could not be found. Please check the URL.")
        else:
            # Store processed content for later use
            st.session_state.chunks = chunks
            st.session_state.title = title
            st.session_state.summary = summary

            st.subheader(f"Title: {title}")
            st.write("Summary:", summary)

            main_placeholder.text("Text Splitter...Started...")
            main_placeholder.text("Embedding Vector Started Building...")
            st.success("Wikipedia page processed and summarized successfully!")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
else:
    st.info("Please enter a Wikipedia link in the sidebar to start.")

if 'chunks' in st.session_state:
    query = st.text_input("Ask a question based on the Wikipedia page")

    if query:
        try:
            main_placeholder.text("Searching for relevant content...🔍")
            
            answer = generate_answer_llm(query, st.session_state.chunks)
            
            main_placeholder.text("Answer generated successfully!✅")
            st.subheader("Answer to your query:")
            st.write(answer)
        except Exception as e:
            st.error(f"An error occurred while generating the answer: {str(e)}")