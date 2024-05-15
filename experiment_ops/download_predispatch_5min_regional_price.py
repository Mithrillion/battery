import numpy as np
import pandas as pd
from pathlib import Path
# import requests
from urllib import request

# create "extra_data/predispatch_price" folder if not exist
output_path = Path("../extra_data/p5min")
output_path.mkdir(parents=True, exist_ok=True)

def fill_url_template(year, month):
    base_url_template = f"https://nemweb.com.au/Data_Archive/Wholesale_Electricity/MMSDM/{year}/MMSDM_{year}_{month:02}/MMSDM_Historical_Data_SQLLoader/DATA/"
    predispath_price_filename = f"PUBLIC_DVD_P5MIN_REGIONSOLUTION_{year}{month:02}010000.zip"
    return base_url_template + predispath_price_filename

# download files
for year in [2022, 2023]:
    for month in range(1, 13):
        url = fill_url_template(year, month)
        print("downloading from", url)
        # download from url and save in output_path
        request.urlretrieve(url, output_path / url.split("/")[-1])