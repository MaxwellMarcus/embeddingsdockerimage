import torch
from pymilvus import MilvusClient
import numpy as np
from transformer_maskgit import CTViT
from transformers import BertTokenizer, BertModel
from ct_clip import CTCLIP
from zero_shot import CTClipInference
import accelerate
import sys
import os
import xnat
import shutil

input_dir = "./inputs2/"
batch_dir = "./batch2/"
out_dir = "./out2/"

tokenizer = BertTokenizer.from_pretrained('microsoft/BiomedVLP-CXR-BERT-specialized',do_lower_case=True)
text_encoder = BertModel.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized")

text_encoder.resize_token_embeddings(len(tokenizer))


image_encoder = CTViT(
    dim = 512,
    codebook_size = 8192,
    image_size = 480,
    patch_size = 30,
    temporal_patch_size = 15,
    spatial_depth = 4,
    temporal_depth = 4,
    dim_head = 32,
    heads = 8
)

clip = CTCLIP(
    image_encoder = image_encoder,
    text_encoder = text_encoder,
    dim_image = 2097152,
    dim_text = 768,
    dim_latent = 512,
    extra_latent_projection = False,         # whether to use separate projections for text-to-image vs image-to-text comparisons (CLOOB)
    use_mlm=False,
    downsample_image_embeds = False,
    use_all_token_embeds = False

)

    # Process files using multiprocessing with tqdm progress bar
    #with Pool(num_workers) as pool:
    #    list(tqdm(pool.imap(preprocess.process_file, nii_files), total=len(nii_files)))


#Download
if len( sys.argv ) == 4:
    subject_min = sys.argv[ 1 ]
    subject_max = sys.argv[ 2 ]
    batch = sys.argv[ 3 ]
else:
    subject_min = 4000
    subject_max = 5000
    batch = 1

session = xnat.connect( "http://18.188.8.197", user="admin", password="admin" )

project_name = "Train"

project = session.projects[ project_name ]

def download( project, subject_min, subject_max, dir_loc ):
    for i in range( subject_min, subject_max, 1 ):
        subject_name = f"train_{i}"
        subject = project.subjects[ subject_name ]
        print( f"Downloading {subject_name}" )
        subject.download_dir( dir_loc )

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
        preprocess.process_file( nii_files[ i ], dir_loc.replace( input_dir, batch_dir ) )

#Upload

def upload( project ):
    url = "18.221.89.89"
    port = "19530"
    client = MilvusClient( uri = "http://" + url + ":" + port, token = "Milvus:root" )

    for i in os.listdir( out_dir ):
        if ".image." in i:
            image = os.path.join( out_dir, i )

            t = i.replace( ".image.", ".text." )
            text = os.path.join( out_dir, t )

            session_name = i[ :-len( "_*.image.npy" ) ]

            ct_session = project.experiments[ session_name ]

            if not "CTCLIP" in ct_session.resources:
                resource = session.classes.ResourceCatalog( parent=ct_session, label="CTCLIP" )

                resource.upload( text, t )
                resource.upload( image, i )

                print( "Uploading to Milvus..." )
                data = {
                        "Project": project_name,
                        "Subject": "_".join( session_name.split( "_" )[ :-1 ] ),
                        "Session": session_name,
                        "File": i.replace( ".image.npy", ".nii.gz" ),
                        "embedding": np.load( image )[ 0 ],
                        "text_embedding": np.load( text )[ 0 ]
                      }
                client.insert( collection_name="CTCLIP", data=data )

data_folder = "train_preprocessed"
reports_file = "reports.csv"
labels_file = "labels.csv"

print( "Loading model..." )
loaded = False
import time

n = 0
for i in range( subject_min, subject_max, batch ):
    print( "Downloading..." )
    os.mkdir( os.path.join( input_dir, str( n ) ) )
    download( project, i, i + batch, os.path.join( input_dir, str( n ) ) )
    prep( os.path.join( input_dir, str( n ) ) )

    if not loaded:
        clip.load("model.pt")
        loaded = True

    data_folder = os.path.join( batch_dir, n )
    
    print( "Running inference..." )

    inference = CTClipInference(
        clip,
        data_folder = data_folder,
        reports_file= reports_file,
        labels = labels_file,
        batch_size = 1,
        results_folder=out_dir,
        num_train_steps = 1,
    )


    inference.infer()

    session = xnat.connect( "http://18.188.8.197", user="admin", password="admin" )

    project_name = "Train"

    project = session.projects[ project_name ]

    print( "Uploading..." )
    try:
        upload( project )
        shutil.rmtree( out_dir )
    except Exception as e:
        print( "Exception!!!!!" )
        print( e )

    shutil.rmtree( os.path.join( input_dir, n ) )
    shutil.rmtree( os.path.join( batch_dir, n ) )

    n += 1

print( "Done." )
