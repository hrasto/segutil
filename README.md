# Segmenters

``` python
import segmentation
import pandas as pd
import os

print(segmentation.list_built_in())

celex_morpho = segmentation.built_in_segmenter('celex_morpho')
print(celex_morpho.segment("cars"))

bpe = segmentation.built_in_segmenter('morfessor_tokens_5k')
print(bpe.segment("records"))

fb = segmentation.FallBackSegmenter([celex_morpho, bpe])
print(fb.segment("unhinged"))
```