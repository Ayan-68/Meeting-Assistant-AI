import torch  
import os  
import gradio as gr  
from langchain.llms import HuggingFaceHub  
from transformers import pipeline  
from langchain.prompts import PromptTemplate  
from langchain.chains import LLMChain  
from ibm_watson_machine_learning.foundation_models import Model  
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM  
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams  

my_credentials = {  
    "url": ""   #Your credentials
}  
params = {  
    GenParams.MAX_NEW_TOKENS: 800,  
    GenParams.TEMPERATURE: 0.1,  
}  

LLAMA2_model = Model(  
    model_id='meta-llama/llama-3-2-11b-vision-instruct',  
    credentials=my_credentials,  
    params=params,  
    project_id="",  #your project id
)  

llm = WatsonxLLM(LLAMA2_model)  

temp = """  
<s><<SYS>>  
List the key points with details from the context:   
[INST] The context : {context} [/INST]   
<</SYS>>  
"""  

pt = PromptTemplate(  
    input_variables=["context"],  
    template=temp  
)  

prompt_to_LLAMA2 = LLMChain(llm=llm, prompt=pt)  

def transcript_audio(audio_file):  
    pipe = pipeline(  
        "automatic-speech-recognition",  
        model="openai/whisper-tiny.en",  
        chunk_length_s=30,  
    )  
    transcript_txt = pipe(audio_file, batch_size=8)["text"]  
    result = prompt_to_LLAMA2.run(transcript_txt)  
    return result  

audio_input = gr.Audio(sources="upload", type="filepath")  
output_text = gr.Textbox()  

iface = gr.Interface(fn=transcript_audio,   
                     inputs=audio_input, outputs=output_text,   
                     title="Audio Transcription App",  
                     description="Upload the audio file")  

iface.launch(server_name="0.0.0.0", server_port=7860)  
