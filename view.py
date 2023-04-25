#!/usr/bin/env python3
#--------------------------------------------- ghislain.bernard@gmail.com ---------------------------------------------#

# import logging
import sys
import timeit

import emoji

import PIL as pillow

import streamlit as st

#----------------------------------------------------------------------------------------------------------------------#

import module.regression_random_linear

#----------------------------------------------------------------------------------------------------------------------#

def main():

  # logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

  ########################

  st.set_page_config(layout='wide', page_title='PyCaret example')

  with open('style.css', encoding='utf-8') as file:
    st.markdown('<style>' + file.read() + '</style>', unsafe_allow_html=True)

  ########################

  st.title('PyCaret', 'title')
  st.markdown('##### {} laboratory'.format(emoji.emojize(':lab_coat:')))

  st.image(pillow.Image.open('pycaret.webp'), width=128)

  st.button('Rerun')

  timer = timeit.default_timer()

  ########################

  options = {'---': sys.modules[__name__], 'regression (random linear)': module.regression_random_linear}

  options[st.sidebar.selectbox('module', options.keys())].run()

  ########################

  st.markdown('---')

  st.markdown('##### -')
  st.markdown('##### {} in {:.3f} second(s)'.format(emoji.emojize(':rocket:'), timeit.default_timer() - timer))

  del timer

#----------------------------------------------------------------------------------------------------------------------#

def run():

  st.markdown('---')
  st.markdown('#### {} Dependancy'.format(emoji.emojize(':x-ray:')))

  st.table({'module': ['python', 'streamlit'], 'version': [sys.version.split(maxsplit=1)[0], st.__version__]})

#----------------------------------------------------------------------------------------------------------------------#

if __name__ == '__main__':
  main()

#--------------------------------------------- ghislain.bernard@gmail.com ---------------------------------------------#
