from langchain.chains import LLMChain
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.callbacks import StreamingStdOutCallbackHandler

load_dotenv()

llm = init_chat_model("o1-mini",
                      model_provider="openai",
                      temperature=1.0,
                      streaming=True,
                      callbacks=[StreamingStdOutCallbackHandler()])


def set_llm(llm):
    print("selected llm:", llm)
    llm = llm


prompt = """System: You are a product manager, and your job is to design software. You are provided a rough 
description of the software. Expand on this description and generate the complete set of functionalities needed to 
get that software to work. Don't hesitate to make design choices if the initial description doesn't provide enough 
information. Don't generate code or unit tests!!

Human: {input}

Complete software design:
"""

product_manager_chain = LLMChain.from_string(
    llm=llm,
    template=prompt
)

prompt = """System: You are a Software engineering technical lead writing code in {language}. Your job is to come up 
with a detailed description of all the necessary functions, classes, methods, unit tests for the code and attributes 
for the following software description. Make sure to design a software that incorporates all the best practices of 
software development. Make sure you describe how all the different classes and function interact between each other. 
The resulting software should be a fully functional. Produce Mermaid class and sequence diagrams also as an output. 
Don't generate code or unit tests!!

language: {language}

Software description: {input}

Software design:
"""

tech_lead_chain = LLMChain.from_string(
    llm=llm,
    template=prompt
)

prompt = """System: You are a software testing engineer and your job is to design test plans for software. Provide a 
detailed test plan and test cases to test the following software. Make sure to design a test plan that incorporates 
all the best practices of software testing and limit the number of test cases to the minimum.

Software description: {input}

Software test plan:

"""

test_lead_chain = LLMChain.from_string(
    llm=llm,
    template=prompt
)

prompt = """System: You are a software engineer and your job is to design software in {language}. Provide a detailed 
description of the file structure with the required folders and {language} files. Make sure to design a file 
structure that incorporates all the best practices of software development. Make sure you explain in which folder 
each file belong to. Folder and file names should not contain any white spaces. A human should be able to exactly 
recreate that file structure. Make sure that those files account for the design of the software.
Don't generate non-{language} files.

language: {language}

Software design: {input}

File structure:

"""

file_structure_chain = LLMChain.from_string(
    llm=llm,
    template=prompt
)

prompt = """System: Return the complete list of the file paths, including the folder structure using the following 
list of {language} files. Only return well formed file paths: ./<FOLDER_NAME>/<FILE_NAME>.py

Follow the following template:
<FILE_PATH 1>
<FILE_PATH 2>
...

language: {language}

Human: {input}

File paths list:

"""

file_path_chain = LLMChain.from_string(
    llm=llm,
    template=prompt
)

prompt = """System: You are a software engineer. Your job is to write {language} code. Write the code for the 
following file using the following description. Only return code! The code should be able to run in a {language} 
interpreter or compiler. Make sure to implement all the methods and functions. Make sure to import all the necessary 
packages. The code should be complete.

language: {language}

Files structure: {structure}

Software description: {class_structure}

File name: {file}

{language} Code for this file:

"""

code_chain = LLMChain.from_string(
    llm=llm,
    template=prompt
)

prompt = """
Return `<TRUE>` If the following {language} code contains non-implemented parts and return `<FALSE>` otherwise

If a {language} code contains `TODO` or `pass`, it means the code is not implemented.

language: {language}

code: {code}

Return `<TRUE>` if the code is not implemented and return `<FALSE>` otherwise:
"""

missing_chain = LLMChain.from_string(
    llm=llm,
    template=prompt
)

prompt = """System: You are a software engineer. The following {language} code may contain non-implemented functions. If 
the code contains non-implemented functions, describe what additional functions or classes you would need to 
implement those missing implementations. Provide a detailed description of those additional classes or functions 
that you need to implement. Make sure to design a software that incorporates all the best practices of software 
development.

language: {language}

Class description: {class_structure}

Code: {code}

Only return text if some functions are not implemented.

The new classes and functions needed:
"""

new_classes_chain = LLMChain.from_string(
    llm=llm,
    template=prompt
)
