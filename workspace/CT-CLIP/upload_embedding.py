print( "Uploading Embeddings: " )

import sys
import os

from pymilvus import MilvusClient

import numpy as np

batch_size = int( sys.argv[ 8 ] )

project = sys.argv[ 1 ]
subject = sys.argv[ 2 ]
session = sys.argv[ 3 ]

url = sys.argv[ 5 ]
port = sys.argv[ 6 ]
token = sys.argv[ 7 ]

client = MilvusClient( uri = "http://" + url + ":" + port, token = "Milvus:root" )

print( "Uploading to Milvus..." )
for i in os.listdir( "/input/RESOURCES/CT-CLIP-LATENT" ):
    if ".image.npy" in i:
        print( np.load( os.path.join( "/input/RESOURCES/CT-CLIP-LATENT", i ) ) )
        print( np.load( os.path.join( "/input/RESOURCES/CT-CLIP-LATENT", i ) ).shape )

        data = {
                "Project": project,
                "Subject": subject,
                "Session": session,
                "File": i.replace( ".image.npy", ".nii.gz" ),
                "embedding": np.load( os.path.join( "/input/RESOURCES/CT-CLIP-LATENT", i ) )[ 0 ],
                "text_embedding": np.load( os.path.join( "/input/RESOURCES/CT-CLIP-LATENT", i.replace( ".image", ".text" ) ) )[ 0 ]
              }
        client.insert( collection_name="CTCLIP", data=data )



print( "Done." )

