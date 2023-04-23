import argparse 
import os
import shutil
import segmentation as seg

parser = argparse.ArgumentParser()
parser.add_argument('algos', type=str, nargs='*', help='segmentation algorithms (BPE, SP, WP, UNI, MFS)')
parser.add_argument('--vocab-size', '-v', type=int, help='desired vocabulary size', default=10000)
parser.add_argument('--fpath', '-f', type=str)
parser.add_argument('--dir', '-d', type=str, default=None)
args = parser.parse_args()

algos_map = {
    'BPE': seg.SegmenterBPE,
    'SP': seg.SegmenterSentencePiece,
    'WP': seg.SegmenterWordPiece,
    'UNI': seg.SegmenterUnigram,
    'MFS': seg.SegmenterMorfessor
}
algos = [algo for algo in args.algos if algo in algos_map]

fnames = []
if args.dir is not None: 
    fnames = [os.path.join(args.dir, fname) for fname in os.listdir(args.dir) if '_' not in fname]
else:
    fnames.append(args.fpath)

for algo in algos:
    for fname in fnames: 
        print(fname)
        with open(fname, 'r') as original:
            lines = [line.strip() for line in original]
            segmenter = algos_map[algo](lines, args.vocab_size)
            fpath_new = fname.split('.')[0]+'_'+algo+'.txt'
            try: 
                with open(fpath_new, 'w') as segmented:
                    for line in lines:
                        line = ['-'.join(segmenter.segment(word)) for word in line.split()]
                        segmented.write(' '.join(line)+'\n')
            except FileNotFoundError: 
                print(f'file not found ({fname})')
            
