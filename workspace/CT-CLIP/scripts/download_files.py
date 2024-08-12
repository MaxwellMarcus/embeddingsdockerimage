import xnat
import sys
import os

#Download
if len( sys.argv ) == 4:
    subject_min = sys.argv[ 1 ]
    subject_max = sys.argv[ 2 ]
    batch = sys.argv[ 3 ]
else:
    subject_min = 3439
    subject_max = 20000
    batch = 18

session = xnat.connect( "http://18.188.8.197", user="admin", password="admin" )

project_name = "Train"

project = session.projects[ project_name ]

def download( project, subject_min, subject_max, dir_loc ):
    for i in range( subject_min, subject_max, 1 ):
        subject_name = f"train_{i}"
        subject = project.subjects[ subject_name ]
        for session in subject.experiments:
            if not "CTCLIP" in subject.experiments[ session ].resources:
                print( f"Downloading {subject_name}" )
                subject.experiments[ session ].download_dir( dir_loc )

#Prep

import preprocess
import pandas as pd
from multiprocessing import Pool
import shutil
from tqdm import tqdm

def prep( dir_loc ):
    split_to_preprocess = dir_loc #select the validation or test split
    nii_files = preprocess.read_nii_files(split_to_preprocess)


    num_workers = 1  # Number of worker processes

    for i in tqdm( range( len( nii_files ) ) ):
        preprocess.process_file( nii_files[ i ], dir_loc.replace( "input", "batch" ) )

n = 675
for i in range( subject_min, subject_max, batch ):
    print( "Downloading..." )
    os.mkdir( f"./input/{n}/" )
    download( project, i, i + batch, f"./input/{n}/" )
    prep( f"./input/{n}/" )
    n += 1
