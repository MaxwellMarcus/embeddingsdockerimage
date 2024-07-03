import torch
from MedCLIP.medclip.modeling_medclip import MedCLIPModel, MedCLIPVisionModelViT
from MedCLIP.medclip.dataset import MedCLIPProcessor
from PIL import Image
import sys
import os

from pymilvus import MilvusClient

from pydicom import dcmread
import numpy as np

if torch.cuda.is_available():
    device = torch.device( "cuda" )
else:
    device = torch.device( "cpu" )


print( "Loading Processor..." )
processor = MedCLIPProcessor()

images = []

print( "Getting images..." )
if len( sys.argv ) > 5 and sys.argv[ 5 ] == "test":
    ds = dcmread( "test.dcm" )
    image = ds.pixel_array
    print( image.shape )
    if len( image.shape ) == 2:
        image = np.stack( [ image, image, image ], 2 )

    image = ( image - image.min() ) / np.clip( image.max() - image.min(), a_min = 1e-8, a_max = None )

    images.append( Image.fromarray( np.uint8( image ) ) )
    
else:
    if "input" in os.listdir( "/" ):
        for i in os.listdir( "/input" ):
            if ".dcm" in i:
                ds = dcmread( os.path.join( "/input", i ) )
                image = ds.pixel_array
                if len( image.shape ) == 2:
                    image = np.stack( [ image, image, image ], 2 )

                image = ( image - image.min() ) / np.clip( image.max() - image.min(), a_min = 1e-8, a_max = None )

                images.append( Image.fromarray( np.uint8( image ) ) )

print( "Loaded ", len( images ), " images" )

inputs = processor(
        text = [ "" for i in range( len( images ) ) ],
        images = images,
        return_tensors = "pt",
        padding = True
        )

print( "Loading model..." )
model = MedCLIPModel( vision_cls = MedCLIPVisionModelViT )
print( "Loading weights..." )
model.from_pretrained( input_dir="model_weights" )

if torch.cuda.is_available():
    model.cuda()

print( "Running model..." )
outputs = model( **inputs )
