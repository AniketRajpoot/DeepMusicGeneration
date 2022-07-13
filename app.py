import os
import streamlit as st
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import plotly.graph_objects as go
# from midi2audio import FluidSynth
import time 

from app_utils import *
from midi2audio import FluidSynth

import warnings
warnings.filterwarnings('ignore')
# import bs4

# '''
#     TODO :
#     0) Format the code  
#     1) Download the checkpoints for genre based remixing : 'lakh_genre_model.pth'
#     2) In the genre continuation UI, add slider for Instrument Temperature & Output BPM and dropdown for Genre
#     3) Complete the note remixing section and embedd the required model 
#     4) What to do about s2s ? 
# '''

if __name__ == '__main__':
    
    # Basic Page Configurations
    st.set_page_config(
     page_title="Deep Music Generation",
     page_icon="🎙️",
     layout="centered",
     menu_items={
         'About': "##### This is an SER based Emotion prediction app which predicts the emotion by analyzing the input audio of human voice."
        }
    ) 

    # markdown for customization
    st.markdown("""
    <style>
    div.stButton > button:first-child {

        border-radius: 0%;
        height: 3em;
        width: 44em; 
    }

    .stProgress > div > div > div > div {
            background-color: #FF4B4B;
    }
    </style>""", unsafe_allow_html=True)

    # load all the models 
    print(f'Loading the models......')

    # 1) Music generation model with genre conditioning  
    genre_model = createGenreContinuationModel()

    # 2) S2S model     


    # 3) Remix model (MASK one I guess)
    remix_model = createRemixModel()

    print(f'Models loaded')

    ###################################################################################################################################
    ###################################################################################################################################
    ###################################################################################################################################
    
    # Main Body 
    
    # Title of the page
    st.title("Deep Music Generation")
    # Cover of the page 
    cover = Image.open('./images/cover2.png')
    st.image(cover)
    
    # give indent 
    st.write(" ")

    # Show metrics of the project 
    with st.container():
        col1, col2, col3 = st.columns(3)
        col1.metric("Models Embedded", "3", "+3")
        col2.metric("Instruments", "6", "+6")
        col3.metric("Genre Classes", "6", "+6")

    # give indent 
    st.write(" ")
    st.write(" ")
    
    ###################################################################################################################################
    ###################################################################################################################################
    ###################################################################################################################################

    # side navigation bar 
    with st.sidebar:
        
        nav = Image.open('./images/nav.jpg')
        st.image(nav)

        st.write(" ")

        st.title('Select the task type:')
        st.write(" ")
        
        option = st.selectbox(
                'Models',
                ('Music Generation', 'Instrument Interconversion', 'Music remixing'))

        st.info(f'Currently performing : {option}')

    # Main content of the page
    if(option == 'Music Generation'):

        st.subheader("Input")

        uploaded_file = st.file_uploader("Upload an audio file", 
                                        type=['mid'],
                                        accept_multiple_files = False,)
        # fs = FluidSynth()
        # fs.midi_to_audio(uploaded_file, 'output.wav')
        # print(type(uploaded_file), uploaded_file.name)

        if uploaded_file:
            with open("tempDir/uploadedMidi.mid","wb") as f:
                f.write(uploaded_file.getbuffer())
            
            data_vocab = deep_music_genre.MusicVocab.create()
            # deep_music_genre.MusicItem.from_file("tempDir/uploadedMidi.mid", data_vocab).to_stream().write('mp3', fp= "tempDir/uploadedFile.wav")
            # os.system('fluidsynth wt_183k_G.sf2 -F tempDir/uploadedMidi.mid tempDir/uploadedFile.wav')
            # fs.midi_to_audio("./tempDir/uploadedMidi.mid", "./tempDir/uploadedFile.wav")

            st.audio('tempDir/uploadedMidi.mid', format='audio/wav', start_time=0)
            
            st.write(" ")
            st.write(" ")

            st.subheader("Parameters")
            
            # temperature  
            temperature_notes = st.slider('Temperature (Notes)', 0.9, 2.5, 1.8)
            temperature_duration = st.slider('Temperature (Duration)', 0.9, 2.5, 1.8)
            temperature_instrument = st.slider('Temperature (Instrument)', 0.9, 2.5, 1.0)
            top_p = st.slider('Top p', 0.0, 1.0, 0.3)
            output_bpm = st.slider('Output BPM', 1, 240, 120)

            with st.container():
                col1, col2, col3 = st.columns(3)
                with col1:
                    num_tokens = st.number_input('Maximum Length', min_value = 0, max_value = 1024)
                with col2:
                    bars = st.number_input('cutoff beat', min_value = 4, max_value = 128)
                with col3:
                    mem_len = st.number_input('Memory Length', min_value = 512, max_value = 2048)
            
            ins_list = st.multiselect(
                'What are your favorite instruments',
                ['Piano', 'Guitar', 'Bass', 'Violin', 'Flute', 'Brass', 'Misc'],
                [])

            genre = st.selectbox('What genre do you like?',
            ('Auto', 'Pop', 'Folk', 'Jazz', 'Rock', 'Electronic'))

            st.write(" ")
            st.write(" ")

            st.subheader("Predict")

            if st.button('Run Prediction'):

                st.write(" ")

                # Generates the prediction and automatically saves the file 

                full = predictNwGenreModel(genre_model, 'tempDir/uploadedMidi.mid', top_p= top_p,
                genre = genre, temperature_notes = temperature_notes, temperature_duration = temperature_duration,
                temperature_ins = temperature_instrument, mem_len = num_tokens, allowed_ins= ins_list,
                cutoff_beat = bars, output_bpm = output_bpm)
                
                st.write("Generating Output File")
                my_bar = st.progress(0)

                for percent_complete in range(100):
                    time.sleep(0.1)
                    my_bar.progress(percent_complete + 1)
                
                full.to_stream(bpm = output_bpm).write('midi', fp= './outputs/genre_output.mid')
                
                st.write(" ")
                st.write(" ")

                st.subheader("Output")

                if(percent_complete == 99):
                    st.success(f'The output is saved as genre_output.mid in outputs folder')
                

                st.write(" ")
    elif(option == 'Music remixing'):

        st.subheader("Input")

        uploaded_file = st.file_uploader("Upload an audio file", 
                                        type=['mid'],
                                        accept_multiple_files = False)

        if uploaded_file:
            with open("tempDir/uploadedMidi.mid","wb") as f:
                f.write(uploaded_file.getbuffer())
            
            data_vocab = deep_music_genre.MusicVocab.create()
            # deep_music_genre.MusicItem.from_file("tempDir/uploadedMidi.mid", data_vocab).to_stream().write('mp3', fp= "tempDir/uploadedFile.wav")

            # os.system('fluidsynth wt_183k_G.sf2 -F tempDir/uploadedMidi.mid tempDir/uploadedFile.wav')


            # fs.midi_to_audio("./tempDir/uploadedMidi.mid", "./tempDir/uploadedFile.wav")

            st.audio('tempDir/uploadedMidi.mid', format='audio/wav', start_time=0)

            st.write(" ")
            st.write(" ")

            # parameters to be tweaked 
            st.subheader("Parameters")

            # temperature  
            temperature_notes = st.slider('Temperature (Notes)', 0.9, 2.5, 1.0)
            temperature_duration = st.slider('Temperature (Duration)', 0.9, 2.5, 1.0)

            # top-p 
            top_p = st.slider('Top p', 0.0, 1.0, 0.3)

            # percentage of notes/duration mask 
            mask_percentage = st.slider('Mask Percentage', 10, 100, 60)
            output_bpm = st.slider('Output BPM', 1, 240, 120)

            with st.container():
                col1, col2, col3 = st.columns(3)
                with col1:
                    genre = st.selectbox('What genre do you like?',
                    ('Auto', 'Pop', 'Folk', 'Jazz', 'Rock', 'Electronic'))
                with col2:
                    bars = st.number_input('cutoff beat', min_value = 4, max_value = 128)
                with col3:
                    # Remix type 
                    remix_type = st.selectbox(
                    'What do you want to remix?',
                    ('Notes', 'Duration'))
       
            st.write(" ")
            st.write(" ")

            st.subheader("Predict")

            if st.button('Run Prediction'):

                st.write(" ")
                
                full = predictMaskModel(remix_model, 'tempDir/uploadedMidi.mid', top_p = top_p,
                genre = genre, temperature_notes = temperature_notes, temperature_duration = temperature_duration,
                cutoff_beat = bars, output_bpm = output_bpm, pred_type = remix_type.lower(), mask_proportion = mask_percentage/100)
                
                st.write("Generating Output File")
                my_bar = st.progress(0)

                for percent_complete in range(100):
                    time.sleep(0.1)
                    my_bar.progress(percent_complete + 1)
                
                full.to_stream(bpm = output_bpm).write('midi', fp= f'./outputs/remix_{remix_type}_output.mid')

                st.write(" ")
                st.write(" ")

                st.subheader("Output")

                if(percent_complete == 99):
                    st.success(f'The output is saved as remix_{remix_type}_output.mid in outputs folder')
