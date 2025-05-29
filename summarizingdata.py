
from transformers import pipeline,BartTokenizer,BartForConditionalGeneration

import streamlit as st



# function to load the summararization pipeline using a small, fast model
# this funciton is cached to avoid reloading the model every time the app runs
@st.cache_resource
def load_model():
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    model=BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
    # load the summarization pipeline using a distilled that is lightweight
    return pipeline("summarization",model=model,tokenizer=tokenizer,framework="pt")
summarizer= load_model()



#........ streamlit App UI...........
# Title of the app
st.title("Epitomize app or precise the data  ")
# Text input area for user to paste content
text = st.text_area("you enter the data or paste at here",height=300)
# button to trigger summarization
if st.button("Summarize"):
    # check if the text input is not empty
    if text.strip():
        # show a spinner while the model processes the input
        with st.spinner("summarizing..."):
            summary=summarizer(text,max_length=150, min_length=50,do_sample=False)[0]['summary_text']
            st.subheader("summary")
            st.write(summary)
else:
    # show a warning if no input was provided
    st.warning("please enter some data")
