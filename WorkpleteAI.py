import streamlit as st
from selenium import webdriver
from transformers import pipeline, set_seed, GPT2LMHeadModel, GPT2Tokenizer
import time

# Load GPT-2 model and tokenizer
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# Define web driver path
driver_path = "chromedriver.exe"

# Define the function to interact with the browser
def interact_with_browser(prompt):
    # Define the Chrome options
    options = webdriver.ChromeOptions()
    options.add_argument("--start-maximized")
    
    # Create a new instance of the Chrome driver
    driver = webdriver.Chrome(executable_path=driver_path, options=options)
    
    # Load the Redfin website
    driver.get("https://www.redfin.com/")
    
    # Find the search input box and enter the prompt
    search_box = driver.find_element_by_name('searchInputBox')
    search_box.send_keys(prompt)
    search_box.submit()
    
    # Wait for the page to load
    time.sleep(3)
    
    # Extract the search result and feed it to GPT-2 model
    result = driver.find_element_by_class_name("homecard").text
    input_text = prompt + "\n" + result
    set_seed(42)
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    output = model.generate(input_ids, max_length=200, do_sample=True, temperature=0.7)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Close the browser instance
    driver.quit()
    
    return generated_text

# Define the Streamlit app interface
st.title("Workplete AI Tool")

st.sidebar.markdown("# User Input")
prompt = st.sidebar.text_area("Enter your prompt here:")

if st.sidebar.button("Submit"):
    st.markdown("# Result")
    result = interact_with_browser(prompt)
    st.write(result)
