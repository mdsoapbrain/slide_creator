import openai
import streamlit as st
import pickle
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os
from stqdm import stqdm
from modules.ppt_generation import create_ppt
from annotated_text import annotated_text, annotation
import json
from PIL import Image


im = Image.open("assets/main_icon.jpg")

st.set_page_config(
    page_title="GPT Slide Creator",
    page_icon=im,
    layout="wide",
    initial_sidebar_state="auto",
)

block_public_api = False

# Sidebar contents
max_cost_limit = 20
history_cost = json.load(open("total_cost.json", "r"))

with st.sidebar:
    st.title('🧠 GPT Slide Creator 🧠')
    st.markdown(f'''
    ## About
    This app is an LLM-powered slide creator built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model
    - [FAISS](https://github.com/facebookresearch/faiss)


    ## Contact
    [Haydn Cheong](https://www.linkedin.com/in/haydnc/)\n
    [Hung Sheng Shih](https://www.linkedin.com/in/danny-hung-sheng-shih-97528a176/)

    ## Feedback
    [Feedback]()

    ## Current usage
    {history_cost["total_cost"]} / {max_cost_limit}

    ''')
    # for i in stqdm(range(max_cost_limit), st_container=st.sidebar):
    #     if i == min(max_cost_limit, history_cost["total_cost"]):
    #         break

    st.progress(history_cost["total_cost"]/20, text="")

    st.markdown(f"""
    ## Usage in this session
    {0 if "gpt_api_cost" not in st.session_state else st.session_state["gpt_api_cost"]}
    """)
    
    if history_cost["total_cost"] >= max_cost_limit:
        block_public_api = True

    html = f"<a href='https://www.buymeacoffee.com/qmi0000011'><img style='max-width: 100%;' src='https://miro.medium.com/v2/resize:fit:1100/format:webp/1*CEZSIxeYr6PCxsN6Gr38MQ.png'></a>"
    st.markdown(html, unsafe_allow_html=True)
    add_vertical_space(5)
    st.write('Made with 🍣 by Sushi Go')

#load_dotenv()
# tab1, tab2 = st.tabs(["Public", "Personal"])



def remove_references(text):
    if "References" in text:
        text, _ = text.split("References", 1)
    return text

# Summarize text function
def summarize_text(text):
    chunk_size = 15000
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    summarized_text = ""
    for chunk in stqdm(chunks):
        print("Request send to OpenAI API")
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Please summarize, and devide into 5 sections (abstract, introductoin, methods, results, and discussion) with section title begins with * in the following text: {chunk}"},
        ]
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",
            messages=messages
        )
        summarized_text += response['choices'][0]['message']['content']
        usage = response["usage"]["total_tokens"]
        print("Response from OpenAI API received!")
    return summarized_text, usage

# Summarize text function
def summarize_text_2(text):
    chunk_size = 15000
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    summarized_text = ""
    for chunk in stqdm(chunks):
        print("Request send to OpenAI API")
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Please summarize, and devide into 5 sections (abstract, introduction, methods, results, and discussion) in the following text: {chunk}"},
        ]
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",
            messages=messages
        )
        summarized_text += response['choices'][0]['message']['content']
        print("Response from OpenAI API received!")
    return summarized_text

def auto_generate_slides(summarized_text: str) -> str:
    chunk_size = 15000
    chunks = [summarized_text[i:i+chunk_size] for i in range(0, len(summarized_text), chunk_size)]
    slide_structure = ""

    for chunk in stqdm(chunks):
        user_message = f"Assuming you are a researcher, please summarize the {chunk} systematically into the first study according to the sections, and must create a 6~10 pages slide structure with 5 points each slide. (Slide number as title, and '-' as contents), the content should be academic in your language."
        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo-16k',
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_message},
            ]
        )
        slide_structure += response.choices[0].message['content']
        usage = response["usage"]["total_tokens"]

    return slide_structure, usage

def extract_pdf_file(pdf):
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def process_text(text, file_path):
    st.info("Summarizing text")
    summarized_text, usage = summarize_text(text)
    with open(file_path, "w") as f:
        f.write(summarized_text)

    return summarized_text, usage

def organize_text(text):
    text_list = text.strip(' ').split('*')
    res_dict = {}
    for i in text_list:
        head = i.split(":")[0].strip(" ")
        body = ":".join(i.split(":")[1:]).replace("\n", " ")
        if head not in res_dict:
            res_dict[head] = [body]
        else:
            res_dict[head].append(body)

    res_str = ""
    for k, v in res_dict.items():
        res_str += f"{k}\n"
        v_str = '\n'.join(v)
        res_str += f"{v_str}" + "\n\n"

    return res_str

def organize_text_2(text):
    text_list = text.strip(' ').split('*')
    res_dict = {}
    defined_headers = ["abstract", "introduction", "method", "result", "discussion"]
    for i in text_list:
        head = i.split(":")[0].strip(" ")
        body = ":".join(i.split(":")[1:]).replace("\n", " ")
        for tmp_head in defined_headers:
            if tmp_head in head.lower():
                if tmp_head in res_dict:
                    res_dict[tmp_head].append(body)
                else:
                    res_dict[tmp_head] = [body]

    res_str = ""
    for k, v in res_dict.items():
        res_str += f"{k}\n"
        v_str = '\n'.join(v)
        res_str += f"{v_str}" + "\n\n"

    return res_str

def load_pdf_file(pdf_path):
    # creating a pdf file object
    pdfFileObj = open('pdf_path', 'rb')
    # creating a pdf reader object
    pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
    # creating a page object
    pageObj = pdfReader.getPage(0)
    # extracting text from page
    return pageObj.extractText()

def update_session_cost(new_cost):
    if "gpt_api_cost" in st.session_state:
        st.session_state["gpt_api_cost"] += new_cost
    else:
        st.session_state["gpt_api_cost"] = new_cost


_, col_0, _ = st.columns([1, 20,1])
_, col_1, _, col_2, _ = st.columns([1, 3,2,3,6])
_, col_3, _ = st.columns([1, 20,1])

def main():
    counter = 0

    with st.container():
        with col_0:
            st.title("Welcome to GPT Slide Creator!")
            st.markdown('''
            ## Empower Your Research with AI!

            Do you have a wealth of knowledge in a thesis paper that needs summarizing? Or perhaps you're looking for an effortless way to generate informative slides for your next presentation?

            GPT Slide Creator is here to help! Utilizing GPT models from OpenAI, our app can summarize your thesis paper, create structured slides for presentations, and even answer queries about your PDF file

            **How does it work?** 
            We apply state-of-the-art Language Model (LLM) technology to digest and interpret your input documents, providing you with simplified summaries and structured slide layouts, saving you hours of work.

            **Why GPT Slide Creator?** 
            Our system is built on leading technologies: 
            1. Leveraging OpenAI's GPT-3.5 Turbo model, A 4MB paper with 100k tokens typically costs as little as 0.1 USD per analysis

            2. It is quick and efficient, turning complex papers into understandable summaries and structured slides in no time (2 mins).

            3. It is built on LangChain for text processing and the powerful FAISS for fast similarity search. 

            ''')
            
            st.markdown("""---""")
            st.header("🧠 GPT Slide Creator 🧠")


            annotated_text(
                        annotation("Public API", "", background="#FFDD00", color="black"
                        , font_family="San Francisco"),
                        " is made available by the support by ",
                        annotation("our community", "", background="#BBCE00", color="black"
                        , font_family="San Francisco"),
                        " via ",
                        annotation("Buy me a coffee", "", background="#379f9f", color="black"
                        , font_family="San Francisco"),
                        "If the remaining fund is low",
                        " please use your personal key for now.",
                        annotation("Thanks!", "", background="#00A0C8", color="black"
                        , font_family="San Francisco"),
                        "."
                    )

        m = st.markdown("""
        <style>
        div.stButton > button:first-child {
            background-color: rgb(204, 49, 49);
        }
        </style>""", unsafe_allow_html=True)

        with col_1:
            tab1 = st.button("👌 Public key", disabled=block_public_api)

        with col_2:
            tab2 = st.button("🚧 Personal key")
            
        if tab1:
            st.session_state["api_mode"] = "public"

        if tab2:
            st.session_state["api_mode"] = "personal"

        if "api_mode" not in st.session_state:
            st.session_state["api_mode"] = "personal"

    with col_3:
        if st.session_state["api_mode"] == "personal":
            os.environ["OPENAI_API_KEY"] = ""
            tmp_openai_api_key = st.text_input("Enter your OpenAI API key", value="", type="password") 
            openai.api_key = tmp_openai_api_key
            os.environ["OPENAI_API_KEY"] = tmp_openai_api_key

        if st.session_state["api_mode"] == "public":
            tmp_openai_api_key = st.secrets["openai_api_key"]
            openai.api_key = tmp_openai_api_key
            os.environ["OPENAI_API_KEY"] = tmp_openai_api_key


        pdf = st.file_uploader("Upload your thesis paper", type='pdf')
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=15000,
            chunk_overlap=200,
            length_function=len
        )


        cost_dict = {}
        if "preprocessed" not in st.session_state and pdf is not None:

            with st.spinner("Reading PDF"):
                text = extract_pdf_file(pdf)

            with st.spinner("Processing data"):    
                with st.spinner("Cleaning text"):
                    text_cleaned = remove_references(text)
                    summarized_text_initial, summarized_total_tokens = process_text(text_cleaned, file_path="sum1.txt")
                    # Get cost
                    cost_dict["summary"] = float(summarized_total_tokens)/1000*0.004
                    history_cost = json.load(open("total_cost.json", "r"))
                    history_cost["total_cost"] += cost_dict["summary"] 
                    json.dump({"total_cost": history_cost["total_cost"]}, open("total_cost.json", "w"))
                    update_session_cost(cost_dict["summary"])

                    summarized_text_grouped = organize_text_2(summarized_text_initial)
                    print("!" * 50)
                    print(summarized_text_grouped)
                    with open("sum12.txt", "w") as f:
                        f.write(summarized_text_grouped)

                    summarized_text = summarized_text_grouped
                    st.session_state["summarized_text"] = summarized_text
                    st.session_state["preprocessed"] = True

            with st.expander("Click to see the summarized content"):
                st.write(summarized_text)
            # Create a download button for the summarized text
            st.download_button(
                "Download summarized text",
                data=summarized_text,
                file_name='summarized_text.pdf',
                mime='text/plain',
            )
        
        if st.button('Generate slides'):
            ppt_title = st.text_input("PPT title", value="Your subject")
            with st.spinner("Splitting text"):
                store_name = pdf.name[:-4]

            with st.spinner("Auto-generating slides"):
                # Get the summarized text
                summarized_text = st.session_state["summarized_text"]
                # Split the summarized text into lines
                summarized_text_lines = summarized_text.splitlines()
                # Check if there are less than 10 lines
                # Select the first 10 lines
                # Join the lines back into a string
                input_text = " ".join(summarized_text_lines)

            # Use the first 10 lines as input to auto_generate_slides
            slides, slides_total_tokens = auto_generate_slides(input_text)
            cost_dict["slides"] = float(slides_total_tokens)/1000*0.004
            history_cost = json.load(open("total_cost.json", "r"))
            history_cost["total_cost"] += cost_dict["slides"] 
            json.dump({"total_cost": history_cost["total_cost"]}, open("total_cost.json", "w"))
            update_session_cost(cost_dict["slides"])

            with st.expander("Click to see the content of the slides"):
                st.write(f'Name of the file uploaded: {store_name}')
                st.write(slides)

            with open("tmp_slides.txt", "w") as f:
                f.write(slides)

            with open("tmp_slides.txt", "r") as f:
                content = f.read()
            
            content_list = content.split('\n')

            slide_contents = []
            track_dict = {}
            count = 0
            
            tmp_title = ""
            tmp_content_list = []
            tmp_res_dict = {}
            last_k = ""
            for i in content_list:
                if i != '': 
                    if '-' not in i:
                        tmp_title += i + " "
                    else:
                        if tmp_title != "" and tmp_title not in tmp_res_dict:
                            tmp_res_dict[tmp_title] = [i]
                            last_k = tmp_title
                            tmp_title = ""
                        else:
                            tmp_res_dict[last_k].append(i)

            for k, v in tmp_res_dict.items():
                slide_contents.append({"title": k, "content": v})

            print("!"*30)
            print(tmp_res_dict)

            st.download_button(
                "Download summarized text",
                data=summarized_text,
                file_name='summarized_text.txt',
                mime='text/plain',
            )        
            # for idx, i in enumerate(content_list):
            #     if '-' not in i.lower():
            #         # tmp_title_key = i.split(":")[-1].replace("Title: ", "")
            #         if len(i.lower().replace("slide:", "").replace("title:", "").strip(" ")) > 0:
            #             # if tmp_title_key not in track_dict:
            #             slide_contents.append({"title": tmp_title_key, "content": []})
            #                 # track_dict[tmp_title_key] = count
            #                 # count += 1

            #     elif len(i) > 0 and '-' == i[0]:
            #         if len(slide_contents) > 0:
            #             slide_contents[-1]["content"].append(i.replace('- ', '').strip(' '))
            #     else:
            #         pass

            tmp_dict = {}
            for i in slide_contents:
                if i['title'] not in tmp_dict:
                    tmp_dict['title'] = i['content']
                else:
                    tmp_dict['title'] = [i['content']]


            # Check if slide_contents is empty
            if not slide_contents:
                st.warning('No slide content was generated.')
            else:
                valid_slide_contents = []
                for slide_content in slide_contents:
                    # extract title and content from slide_content
                    title = ""  # use empty string as default value
                    content = slide_content.get('content', [])  # use empty list as default value
                    # Check if content is not empty
                    if content:
                        valid_slide_contents.append(slide_content)
                
                if len(valid_slide_contents) > 0:
                    binary_output = create_ppt(ppt_title, valid_slide_contents, outfile_path="test_ppt.pptx")
                    # display success message
                    # If already summarized once
                    st.success('The PPT has been generated for download!')
                    st.download_button(label='Click to download',
                                        data=binary_output.getvalue(),
                                        file_name="generated.pptx")

                else:
                    st.warning(f'No content for slide title: {title}.')


        query = st.text_input("Ask questions about your PDF file:")
        if query:
            store_name = pdf.name[:-4]
            st.write(f'{store_name}')
            summarized_text = st.session_state["summarized_text"]
            chunks = text_splitter.split_text(text=summarized_text)
            if os.path.exists(f"{store_name}.pkl"):
                with open(f"{store_name}.pkl", "rb") as f:
                    VectorStore = pickle.load(f)
            else:
                with st.spinner("Embedding tokens"):
                    embeddings = OpenAIEmbeddings()
                    VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
                    with open(f"{store_name}.pkl", "wb") as f:
                        pickle.dump(VectorStore, f)

            docs = VectorStore.similarity_search(query=query, k=10)
            llm_model_selected = "gpt-3.5-turbo-16k"
            llm = OpenAI(model_name=llm_model_selected)
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                QA_total_tokens = cb.total_tokens

                if QA_total_tokens > 0:
                    cost_dict["QA"] = float(QA_total_tokens)/1000*0.004 
                    history_cost = json.load(open("total_cost.json", "r"))
                    history_cost["total_cost"] += cost_dict["QA"] 
                    json.dump({"total_cost": history_cost["total_cost"]}, open("total_cost.json", "w"))
                    update_session_cost(cost_dict["QA"])

            st.write(response)

            with open("response_record.txt", "w") as f:
                f.write(response)

            with open("QA_history.txt", "a") as f:
                f.write(f"Question: {query}\nResponse: {response}\n\n")

if __name__ == '__main__':
    main()
