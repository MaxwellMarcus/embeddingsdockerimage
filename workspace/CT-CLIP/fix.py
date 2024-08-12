import os
import shutil

for i in os.listdir( "batch" ):
    n = int( i )
    n_min = n * 18
    n_max = n_min + 17
    for p in os.listdir( os.path.join( "batch", i ) ):
        q = int( p.split( "_" )[ 1 ] )
        if not (  q >= n_min and q <= n_max ):
            print( os.path.join( i, p ) )
            shutil.rmtree( os.path.join( "batch", i, p ) )
            shutil.rmtree( os.path.join( "input", i, p ) )
