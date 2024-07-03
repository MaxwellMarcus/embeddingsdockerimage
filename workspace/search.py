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

ds = dcmread( "test.dcm" ) # dcmread( "/Users/maxmarcus/xnat-docker-compose/xnat-data/archive/buttcancer/arc001/TCGA-CL-4957/SCANS/2/DICOM/1-20.dcm" )
image = ds.pixel_array

print( image.shape )
if len( image.shape ) == 2:
    image = np.stack( [ image, image, image ], 2 )

image = ( image - image.min() ) / np.clip( image.max() - image.min(), a_min = 1e-8, a_max = None )

print( image.min(), image.max() )

print( np.uint8( image*255 ).min(), np.uint8( image*255 ).max() )

images.append( Image.fromarray( np.uint8( image * 255 ) ) )
    

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
model.from_pretrained( input_dir="workspace/model_weights" )

if torch.cuda.is_available():
    model.cuda()

print( "Running model..." )
outputs = model( **inputs )


v = outputs[ "img_embeds" ]

print( v.min(), v.max() )

client = MilvusClient( uri = "http://18.118.101.73:19530", token = "Milvus:root" )
client.release_collection( collection_name="MedCLIP" )

print( client.list_indexes( collection_name="MedCLIP" ) )
client.drop_index( collection_name="MedCLIP", index_name="embedding" )

index_params = client.prepare_index_params()
index_params.add_index(
        field_name="embedding",
        index_type="IVF_FLAT",
        metric_type="L2",
        params={"nlist":1024}
)

client.create_index(
        collection_name="MedCLIP",
        index_params=index_params
)


print( "Loading collection..." )
client.load_collection( collection_name="MedCLIP" )

search_params = {
        "metric_type": "L2",
        "params": {}
        }

res = client.search(
        collection_name="MedCLIP", data=v.detach().numpy().tolist(), limit=3, search_params=search_params, output_fields=["Project", "Subject", "Session", "Scan", "File" ]
        )

print( res )

client.release_collection( collection_name="MedCLIP" )
