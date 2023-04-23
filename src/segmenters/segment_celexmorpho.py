import argparse 
import os
import shutil
import segmentation as seg

parser = argparse.ArgumentParser()
parser.add_argument('--fallback_algo', '-a', type=str, default='CHAR', help='one of CHAR, MFS, NONE')
parser.add_argument('--fpath', '-f', type=str)
parser.add_argument('--dir', '-d', type=str, default=None)
args = parser.parse_args()

algos_map = {
    'CHAR': seg.CharacterSegmenter(),
    'MFS': seg.built_in('morfessor_tokens'),
    'NONE': seg.SegmenterDummy()
}
seg_fallback = algos_map[args.fallback_algo]

fnames = []
if args.dir is None: 
    fnames = [fname for fname in os.listdir(args.dir) if '_' not in fname]
else:
    fnames.append(args.fpath)

for fname in fnames: 
    try:
        with open(fname, 'r') as original:
            celex_dict = seg.load_celex_morpho()
            segmenter = seg.FallBackSegmenter(segmenters=[
                seg.LookUpSegmenter(celex_dict),
                seg_fallback
            ])

            fpath_new = fname.split('.')[0]+'_CELEX_'+args.fallback_algo+'.txt'

            lines = [line.strip() for line in original]
            with open(fpath_new, 'w') as segmented:
                for line in lines:
                    line = ['-'.join(segmenter.segment(word)) for word in line.split()]
                    segmented.write(' '.join(line)+'\n')
    except FileNotFoundError: 
        print(f'file not found ({fname})')