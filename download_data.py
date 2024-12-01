import requests 
from zipfile import ZipFile
import os



# https://github.com/Garrafao/WUGs/tree/main/scripts


"""
Downloads datasets into directory "./data", extracts downloaded zip files and deletes zip files. 
"""

def download_datasets(zipurls):
    # create data directory
    if not os.path.exists("./data"):
        os.makedirs("./data")
    # download data 
    for zipurl in zipurls:
        zipresponse = requests.get(zipurl)   # download zip file from URL
        zipfile_path = "./data/zipdata.zip"
        with open (zipfile_path, "wb") as f:   # create new file
            f.write(zipresponse.content)        # write URL zip content to new zip file 
        print("ZIP file downloaded.")
        with ZipFile("./data/zipdata.zip") as zf:      # open created zip file 
            zf.extractall(path="./data")                  # extract zip file to "./data" 
        print("ZIP file extracted.")
        os.remove(zipfile_path)                # delete zip file
        print("ZIP file deleted.")




if __name__=="__main__":
    # List of datasets: DWUG DE, DiscoWUG, RefWUG, DWUG EN, DWUG SV, DWUG LA, DWUG ES, ChiWUG, NorDiaChange, DWUG DE Sense 
    zipurls = ["https://zenodo.org/records/14028509/files/dwug_de.zip?download=1", "https://zenodo.org/records/14028592/files/discowug.zip?download=1", 
                "https://zenodo.org/records/5791269/files/refwug.zip?download=1", "https://zenodo.org/records/14028531/files/dwug_en.zip?download=1",
                "https://zenodo.org/records/14028906/files/dwug_sv.zip?download=1", "https://zenodo.org/records/5255228/files/dwug_la.zip?download=1",
                "https://zenodo.org/records/6433667/files/dwug_es.zip?download=1", "https://zenodo.org/records/10023263/files/chiwug.zip?download=1",
                "https://github.com/ltgoslo/nor_dia_change/archive/refs/heads/main.zip", "https://zenodo.org/records/14041715/files/dwug_de_sense.zip?download=1"]
    download_datasets(zipurls)