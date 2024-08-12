import torch
from pymilvus import MilvusClient
import numpy as np
from transformer_maskgit import CTViT
from transformers import BertTokenizer, BertModel
from ct_clip import CTCLIPTEXT
from text_zero_shot import CTClipTextInference
import accelerate
import sys
import os
import xnat
import shutil

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

clip = CTCLIPTEXT(
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

#Upload

def upload( project ):
    url = "18.221.89.89"
    port = "19530"
    client = MilvusClient( uri = "http://" + url + ":" + port, token = "Milvus:root" )

    for i in os.listdir( "./textout" ):
        if ".metadata." in i:
            text = os.path.join( "./textout", i )

            session_name = i[ :-len( "_*.metadata.npy" ) ]

            ct_session = project.experiments[ session_name ]
            print( f"Uploading to {session_name}" )

            if not "CTCLIP_METADATA" in ct_session.resources:
                resource = session.classes.ResourceCatalog( parent=ct_session, label="CTCLIP_METADATA" )

                resource.upload( text, i )

                print( "Uploading to Milvus..." )
                data = {
                        "Project": project_name,
                        "Subject": "_".join( session_name.split( "_" )[ :-1 ] ),
                        "Session": session_name,
                        "File": i.replace( ".metadata.npy", ".nii.gz" ),
                        "embedding": np.load( text )[ 0 ],
                      }
                client.insert( collection_name="CTCLIP_METADATA", data=data )

data_folder = "test_preprocessed"
reports_file = "reports.csv"
labels_file = "labels.csv"

print( "Loading model..." )
loaded = False
import time
import psutil

for i in range( 12184, 50000, 10 ):
    print( f"Getting metadata embeddings for {i} through {i + 10}..." )

    if True:#str( i + 1 ) in os.listdir( "./batch" ):
        if not loaded:
            print( "Loading model..." )
            clip.load("model.pt")
            loaded = True

        data_folder = f"./batch/{i}"
        
        print( "Running inference..." )

        inference = CTClipTextInference(
            clip,
            data_folder = data_folder,
            reports_file= reports_file,
            labels = labels_file,
            batch_size = 1,
            row_min = i,
            row_max = i+10,
            results_folder="./textout/",
            metadata="metadata.csv",
            num_train_steps = 1
        )


        inference.infer()

        session = xnat.connect( "http://18.188.8.197", user="admin", password="admin" )

        project_name = "Train"

        project = session.projects[ project_name ]

        print( "Uploading..." )
        try:
            upload( project )
            shutil.rmtree( "./textout/" )
        except Exception as e:
            print( "Exception..." )
            print( e )

        
        if psutil.virtual_memory().percent > 95:
            quit()

print( "Done." )
