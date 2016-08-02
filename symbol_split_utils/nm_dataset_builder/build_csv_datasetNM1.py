import os
from os import path
import subprocess
from argparse import ArgumentParser

def parse_args():
    def file_arg(value): 
        if not os.path.exists(value):
            if not os.path.exists(value):
                raise Exception() 
        return value
        
    parser = ArgumentParser()
    parser.add_argument('positives', type=file_arg,
                        help="path to directory with positive samples")
    parser.add_argument('negatives', type=file_arg,
                        help="path to directory with negative samples")
    parser.add_argument('--out', help="output_file")
    return parser.parse_args()
    
    
if __name__ == '__main__':
    args = parse_args()
    
    out_file = 'char_datasetNM1.csv'
    if args.out:
        out_file = args.out
    
    f = open(out_file, 'w')
    
    #obtain positive samples 
    traindbdir = args.positives
    
    for filename in os.listdir(traindbdir):
        print("processing "+filename);
        
        out = subprocess.check_output([path.join(path.dirname(__file__),
                                                 "extract_featuresNM1"),
                                       path.join(traindbdir,filename)]).decode()
        
        if ("Non-integer" in out):
            print("ERROR: Non-integer Euler number")
        
        else:
            if (out != ''):
                out = out.replace("\n","\nC,",out.count("\n")-1)
                f.write("C," + out)
    
    #obtain negative samples 
    traindbdir = args.negatives
    
    for filename in os.listdir(traindbdir):
        print("processing "+filename);
        
        out = subprocess.check_output([path.join(path.dirname(__file__),
                                                 "extract_featuresNM1"),
                                       path.join(traindbdir, filename)]).decode()
        
        if ("Non-integer" in out):
            print("ERROR: Non-integer Euler number")
        
        else:
            if (out != ''):
                out = out.replace("\n","\nN,",out.count("\n")-1)
                f.write("N," + out)
                
    f.close()
