import torch
import numpy as np
from transformer_maskgit import CTViT
from transformers import BertTokenizer, BertModel
from ct_clip import CTCLIP
from zero_shot import CTClipInference
import accelerate
import sys
import os

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


data_folder = "train_preprocessed"
reports_file = "/input/RESOURCES/report/report.csv"
labels_file = "/input/RESOURCES/label/label.csv"

print( reports_file, os.path.isfile( reports_file ) )
print( labels_file, os.path.isfile( labels_file ) )

clip.load("model.pt")

batch_size = 1 #int( sys.argv[ 8 ] )

inference = CTClipInference(
    clip,
    data_folder = data_folder,
    reports_file= reports_file,
    labels = labels_file,
    batch_size = batch_size,
    results_folder="/out/",
    num_train_steps = 1,
)


inference.infer()

print( "Done." )
