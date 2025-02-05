import streamlit as st
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
import chains
from utils import safe_write
from langchain.callbacks import StreamingStdOutCallbackHandler
from chains import (
    product_manager_chain,
    tech_lead_chain,
    test_lead_chain,
    file_structure_chain,
    file_path_chain,
    code_chain,
    missing_chain,
    new_classes_chain
)

load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

base_url = "https://api.deepseek.com"

o1_mini = init_chat_model('o1-mini',
                         model_provider='openai',
                         temperature=1.0,
                         streaming=True,
                         callbacks=[StreamingStdOutCallbackHandler()])

o1 = init_chat_model('o1',
                         model_provider='openai',
                         temperature=1.0,
                         streaming=True,
                         callbacks=[StreamingStdOutCallbackHandler()])

o3_mini = init_chat_model('o3-mini',
                            model_provider='openai',
                            temperature=1.0,
                            streaming=True,
                            callbacks=[StreamingStdOutCallbackHandler()])

gemini_2_0_flash_thinking = init_chat_model('gemini-2.0-flash-thinking-exp-01-21',
                              model_provider='google_genai',
                              temperature=0.5,
                              streaming=True,
                              callbacks=[StreamingStdOutCallbackHandler()])

gemini_2_0_pro = init_chat_model('gemini-2.0-pro-exp-02-05',
                                model_provider='google_genai',
                                temperature=0.5,
                                streaming=True,
                                callbacks=[StreamingStdOutCallbackHandler()])

deep_seek_r1 = init_chat_model('deepseek-reasoner',
                              model_provider='openai',
                              temperature=0.5,
                              streaming=True,
                              base_url=base_url,
                              api_key=DEEPSEEK_API_KEY,
                              callbacks=[StreamingStdOutCallbackHandler()])

deep_seek_chat = init_chat_model('deepseek-chat',
                                model_provider='openai',
                                temperature=0.5,
                                streaming=True,
                                base_url=base_url,
                                api_key=DEEPSEEK_API_KEY,
                                callbacks=[StreamingStdOutCallbackHandler()])

claude_3_5_sonnet = init_chat_model('claude-3-5-sonnet-20241022',
                                    model_provider='anthropic',
                                    temperature=0.5,
                                    streaming=True,
                                    callbacks=[StreamingStdOutCallbackHandler()])

mistral_codestral = init_chat_model('codestral-latest',
                                    model_provider='mistralai',
                                    temperature=0.5,
                                    streaming=True,
                                    callbacks=[StreamingStdOutCallbackHandler()])

llama_3_3_groq = init_chat_model('llama-3.3-70b-versatile',
                                 model_provider='groq',
                                 temperature=0.5,
                                 streaming=True,
                                 callbacks=[StreamingStdOutCallbackHandler()])



st.title('Code Generator')

language = st.radio('Select Language:',
                    ['Python', 'Java', 'Rust', 'Kotlin', 'Go'])

st.markdown(
    '''
    <style>
        section[data-testid="stSidebar"] {
            width: 400px !important; # Set the width to your desired value
        }
    </style>
    ''',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.image('image/software_eng.png')
    add_radio = st.radio(
        'Select the LLM to be used!',
        ('o1_mini',
         'o1',
         'o3_mini',
         'gemini_2_0_flash_thinking',
         'gemini_2_0_pro',
         'claude_3_5_sonnet',
         'mistral_codestral',
         'llama_3_3_groq',
         'deep_seek_r1',
         'deep_seek_chat'))  
     
if add_radio == 'o1_mini':
    chains.set_llm(o1_mini)
elif add_radio == 'o1':
    chains.set_llm(o1)
elif add_radio == 'o3_mini':
    chains.set_llm(o3_mini)
elif add_radio == 'gemini_2_0_flash_thinking':
    chains.set_llm(gemini_2_0_flash_thinking)
elif add_radio == 'gemini_2_0_pro':
    chains.set_llm(gemini_2_0_pro)
elif add_radio == 'claude_3_5_sonnet':
    chains.set_llm(claude_3_5_sonnet)
elif add_radio == 'mistral_codestral':
    chains.set_llm(mistral_codestral)
elif add_radio == 'llama_3_3_groq':
    chains.set_llm(llama_3_3_groq)
elif add_radio == 'deep_seek_r1':
    chains.set_llm(deep_seek_r1)
elif add_radio == 'deep_seek_chat':
    chains.set_llm(deep_seek_chat)

request = st.text_area('Please Detail Your Desired Use Case for Code Generation! ', height=100)
st.write('Generate a microservice to manage Trades. Use Redis as the DB.')
app_name = st.text_input('Enter Project Name:')
submit = st.button('submit', type='primary')

if language and submit and request and app_name:

    dir_path = app_name + '/'

    requirements = product_manager_chain.invoke(request)['text']
    req_doc_path = dir_path + '/requirements' + '/requirements.txt'
    safe_write(req_doc_path, requirements)
    st.markdown(''' :blue[Business Requirements : ] ''', unsafe_allow_html=True)
    st.write(requirements)

    tech_design = tech_lead_chain.invoke({'language': language, 'input': request})['text']
    tech_design_path = dir_path + '/tech_design' + '/tech_design.txt'
    safe_write(tech_design_path, tech_design)
    st.markdown(''' :blue[Technical Design :] ''', unsafe_allow_html=True)
    st.write(tech_design)

    test_plan = test_lead_chain.invoke(requirements)['text']
    test_plan_path = dir_path + '/test_plan' + '/test_plan.txt'
    safe_write(test_plan_path, test_plan)
    st.markdown(''' :blue[Test Plan :] ''', unsafe_allow_html=True)
    st.write(test_plan)

    file_structure = file_structure_chain.invoke({'language': language, 'input': tech_design})['text']
    file_structure_path = dir_path + '/file_structure' + '/file_structure.txt'
    safe_write(file_structure_path, file_structure)
    st.markdown(''' :blue[File Names :] ''', unsafe_allow_html=True)
    st.write(file_structure)

    files = file_path_chain.invoke({'language': language, 'input': file_structure})['text']
    files_path = dir_path + '/files' + '/files.txt'
    safe_write(files_path, files)
    st.markdown(''' :blue[File Paths :] ''', unsafe_allow_html=True)
    st.write(files)

    files_list = files.split('\n')

    missing = True
    missing_dict = {
        file: True for file in files_list
    }

    code_dict = {}

    while missing:

        missing = False
        new_classes_list = []

        for file in files_list:

            code_path = os.path.join(dir_path, 'code', file)
            norm_code_path = code_path

            if not missing_dict[file]:
                safe_write(norm_code_path, code_dict[file])
                st.markdown(''' :red[Code & Unit Tests: 2nd Iteration] ''', unsafe_allow_html=True)
                st.write(code_dict[file])
                continue

            code = code_chain.predict(
                language=language,
                class_structure=tech_design,
                structure=file_structure,
                file=file,
            )

            code_dict[file] = code
            response = missing_chain.invoke({'language': language, 'code': code})
            if '<TRUE>' in response:
                missing = missing or missing_dict[file]
            else:
                safe_write(norm_code_path, code)
                st.markdown(''' :blue[Complete Code & Unit Tests: 1st Iteration] ''', unsafe_allow_html=True)
                st.write(code)
                continue

            if missing_dict[file]:
                new_classes = new_classes_chain.predict(
                    language=language,
                    class_structure=tech_design,
                    code=code
                )
                new_classes_list.append(new_classes)

        tech_design += '\n\n' + '\n\n'.join(new_classes_list)
