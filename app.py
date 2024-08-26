from dotenv import load_dotenv
import os
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from ctransformers import AutoModelForCausalLM
from pyngrok import ngrok

# Load environment variables
load_dotenv()

# Set environment variables
os.environ["HUGGINGFACE_HUB_TOKEN"] = os.environ.get('HUGGINGFACE_HUB_TOKEN')

# Fetch and set the NGROK_AUTH_TOKEN
NGROK_AUTH_TOKEN = os.environ.get('NGROK_AUTH_TOKEN')
ngrok.set_auth_token(NGROK_AUTH_TOKEN)  # Set the authentication token

## Function to get response from LLama 2 model
def getLLamaresponse(input_text, no_words, blog_style):
    try:
        # LLama2 model
        model_id = "TheBloke/Llama-2-7B-chat-GGML"
        config = {'max_new_tokens': 256, 'repetition_penalty': 1.1, 
          'temperature': 0.1, 'stream': True}
        llm = AutoModelForCausalLM.from_pretrained(model_id, 
      model_type="llama",                                           
      #lib='avx2', for cpu use
      gpu_layers=130, #110 for 7b, 130 for 13b
      **config)

        # Prompt Template
        template = """
            Write a blog for {blog_style} job profile on the topic {input_text}
            within {no_words} words.
        """

        prompt = PromptTemplate(input_variables=["blog_style", "input_text", 'no_words'],
                                template=template)

        # Generate the response from the LLama 2 model
        response = llm(prompt.format(blog_style=blog_style, input_text=input_text, no_words=no_words))
        return response
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

# Streamlit app setup
st.set_page_config(page_title="Generate Blogs",
                   page_icon='🤖',
                   layout='centered',
                   initial_sidebar_state='collapsed')

st.header("Generate Blogs 🤖")

input_text = st.text_input("Enter the Blog Topic")

# Creating two columns for additional fields
col1, col2 = st.columns([5, 5])

with col1:
    no_words = st.text_input('Number of Words')
with col2:
    blog_style = st.selectbox('Writing the blog for',
                              ('Researchers', 'Data Scientist', 'Common People'), index=0)

submit = st.button("Generate")

# Generate the blog content on button click
if submit:
    response = getLLamaresponse(input_text, no_words, blog_style)
    if response:
        st.write(response)