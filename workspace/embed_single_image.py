print( "Creating Embeddings: " )

import torch
from MedCLIP.medclip.modeling_medclip import MedCLIPModel, MedCLIPVisionModelViT
from MedCLIP.medclip.dataset import MedCLIPProcessor
from PIL import Image
import sys
import os

from pymilvus import MilvusClient

# from pydicom import dcmread
import nibabel as nib
import numpy as np

if torch.cuda.is_available():
    device = torch.device( "cuda" )
else:
    device = torch.device( "cpu" )


print( "Loading Processor..." )
processor = MedCLIPProcessor()

print( "Loading model..." )
model = MedCLIPModel( vision_cls = MedCLIPVisionModelViT )
print( "Loading weights..." )
model.from_pretrained( input_dir="model_weights" )

if torch.cuda.is_available():
    model.cuda()


images = []
file_names = []

batch_size = int( sys.argv[ 8 ] )

url = sys.argv[ 5 ]
port = sys.argv[ 6 ]
token = sys.argv[ 7 ]

client = MilvusClient( uri = "http://" + url + ":" + port, token = "Milvus:root" )


print( "Getting images..." )
if "input" in os.listdir( "/" ):
    dcm = [ f for f in os.listdir( "/input" ) ]
    p = 0
    i = dcm[ len( dcm ) // 2 ]
    #ds = dcmread( os.path.join( "/input", i ) )
    #image = ds.pixel_array
    ds = nib.load( os.path.join( "/input", i ) )
    image = np.array( ds.get_fdata() )
    image = image[ :, :, image.shape[ 2 ] // 2 ]
    print( image.shape )
    if len( image.shape ) == 2:
        image = np.stack( [ image, image, image ], 2 )

    image = ( image - image.min() ) / np.clip( image.max() - image.min(), a_min = 1e-8, a_max = None )

    images.append( Image.fromarray( np.uint8( image * 255 ) ) )
    file_names.append( i )

inputs = processor(
        text = [ "" for i in range( len( images ) ) ],
        images = images,
        return_tensors = "pt",
        padding = True
        )

print( "Running model..." )
outputs = model( **inputs )

print( "Uploading to Milvus..." )
data = [
        {
         "Project": sys.argv[ 1 ],
         "Subject": sys.argv[ 2 ],
         "Session": sys.argv[ 3 ],
         "Scan": sys.argv[ 4 ],
         "File": file_names[ n ],
         "embedding": v
        } for n, v in enumerate( outputs[ "img_embeds" ].detach().numpy().tolist() )
]
if len( data ): client.insert( collection_name="MedCLIP", data=data )


print( "Done." )
