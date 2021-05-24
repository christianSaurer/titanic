#RETRIEVES INPUT FILES FOR KAGGLE COMPETITION
#YOU NEED TO FIRST DOWNLOAD YOUR API TOKEN (SEE https://www.kaggle.com/docs/api)

from kaggle.api.kaggle_api_extended import KaggleApi
import shutil
import os

data_folder = os.getcwd() + '/data/input'

if not os.path.exists(data_folder):
    os.makedirs(data_folder)

api = KaggleApi()
api.authenticate()
api.competition_download_files('titanic', path=data_folder)
shutil.unpack_archive(data_folder + '/titanic.zip', data_folder)
