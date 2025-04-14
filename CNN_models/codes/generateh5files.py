from hd5file import ascii_to_hd5
import sys 

filename = sys.argv[1]   
ascii_to_hd5 (filename,
              add_reverse=False,
              num_padding=0,
              add_shape=False)
                  
