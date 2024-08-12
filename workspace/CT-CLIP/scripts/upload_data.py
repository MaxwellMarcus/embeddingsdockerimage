import xnat
import os

session = xnat.connect( "http://http://18.188.8.197", user="admin", password="admin" )
project_name = "Train"
project = session.projects[ "train" ]

for i in os.listdir( "/out" ):
    if ".image." in i:
        image = os.path.join( "/out", i )

        t = i.replace( ".image.", ".text." )
        text = os.path.join( "/out", t )

        session_name = i[ :-len( "_*.image.npy" ) ]

        ct_session = project.experiments[ session_name ]

        resource = session.classes.ResourceCatalog( parent=ct_session, label="CTCLIP" )
        
        resource.upload( text, t )
        resource.upload( image, i )
