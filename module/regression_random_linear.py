#--------------------------------------------- ghislain.bernard@gmail.com ---------------------------------------------#

import os
import os.path
import sys

import emoji

import matplotlib as mpl
import matplotlib.pyplot as plt

import streamlit as st

import numpy as np
import pandas as pd

import pycaret.regression

#----------------------------------------------------------------------------------------------------------------------#

def run():

  plt.style.use('dark_background')

  palette = mpl.cm.get_cmap('Pastel2').colors

  plt.rcParams['image.cmap'] = 'Pastel2'
  plt.rcParams['axes.prop_cycle'] = plt.cycler(color=palette)

  plt.rcParams['figure.constrained_layout.use'] = True

  ########################

  st.sidebar.markdown('---')

  st.sidebar.markdown('### {} parameters'.format(emoji.emojize(':gear:')))

  width = st.sidebar.select_slider('width', [100, 200, 500, 1000, 2000, 5000, 10000, 50000, 100000], 500)
  noise = st.sidebar.slider('noise', 0.0, 0.25, 0.1, step=0.05)

  st.sidebar.markdown('---')

  ########################

  st.markdown('---')
  st.markdown('#### {} Dataset'.format(emoji.emojize(':chart_increasing:')))

  st.markdown('{} samples'.format(width))

  feature = np.random.normal(size=width)

  if noise:

    dataset = pd.DataFrame({
      'feature': feature,
      'target': feature + np.random.normal(scale=noise, size=feature.shape)
    })

  else:

    dataset = pd.DataFrame({'feature': feature, 'target': feature})

  st.write(dataset.head())

  ########################

  columns = st.columns([2, 3, 2])

  figure, axe = plt.subplots(figsize=(10, 10))
  axe.grid(False)
  axe.scatter(dataset['feature'], dataset['target'], alpha=0.5)
  columns[1].pyplot(figure)

  plt.close(figure)
  del figure, axe

  del columns

  # ########################

  st.markdown('---')
  st.markdown('#### {} Prediction'.format(emoji.emojize(':alembic:')))

  pycaret.regression.setup(data=dataset,
                           html=False,
                           session_id=0,
                           target='target',
                           train_size=0.8,
                           verbose=False)

  st.dataframe(pycaret.regression.pull())

  model = pycaret.regression.compare_models(exclude=['ada', 'catboost', 'xgboost'], verbose=False)
  st.table(pycaret.regression.pull())

  # st.write(model)
  # pycaret.regression.evaluate_model(model)

  # st.write(pycaret.regression.predict_model(model))

  # model = tf.keras.models.Sequential([tf.keras.layers.Dense(1)])

  # model.compile(loss=tf.keras.losses.mean_absolute_error, optimizer=tf.keras.optimizers.SGD())

  # ########################

  # columns = st.columns(5)

  # losses = []
  # for number in range(epoch):

  #   history = model.fit(features, targets, batch_size=10, verbose=0).history
  #   losses.append(history['loss'][0])

  #   targets_predicted = model.predict(features, verbose=0)

  #   with columns[number % len(columns)]:

  #     figure, axe = plt.subplots(figsize=(5, 5))
  #     axe.set_title('epoch={}'.format(number + 1))

  #     axe.scatter(features, targets_predicted, alpha=0.5, c=palette[1])
  #     st.pyplot(figure)

  #     plt.close(figure)
  #     del figure, axe

  # del columns

  # ########################

  # figure, axe = plt.subplots(figsize=(20, 6))
  # axe.set_xlabel('epoch')
  # axe.set_ylabel('loss')
  # axe.set_xlim([0, epoch + 1])
  # axe.set_xticks(range(1, epoch + 1))

  # axe.plot(range(1, epoch + 1), losses, '-o', c=palette[2])
  # st.pyplot(figure)

  # plt.close(figure)
  # del figure, axe

  ########################

  if os.path.exists('logs.log'):
    os.remove('logs.log')

  ########################


  st.markdown('---')

  st.markdown('#### {} Dependancy'.format(emoji.emojize(':x-ray:')))
  st.table({
    'module': ['python', 'matplotlib', 'streamlit', 'numpy', 'pandas', 'pycaret'],
    'version': [
      sys.version.split(maxsplit=1)[0], mpl.__version__, st.__version__, np.__version__, pd.__version__,
      pycaret.__version__
    ]
  })

#--------------------------------------------- ghislain.bernard@gmail.com ---------------------------------------------#
