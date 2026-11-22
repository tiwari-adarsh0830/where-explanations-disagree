# Where the data in this folder comes from

## ioi_lists.json

The names, places, objects and BABA templates are copied from `easy_transformer/ioi_dataset.py` in
https://github.com/redwoodresearch/Easy-Transformer, the code release for Wang et al. (2023), "Interpretability in the Wild:
a Circuit for Indirect Object Identification in GPT-2 small" (checked Nov 22, 2026: same 99 names, same places, objects and
templates). That repository is released under the MIT License, reproduced here as it requires:

```
MIT License

Copyright (c) 2022 neelnanda-io

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## gt_nouns.txt

The 120 nouns are the ones used by Hanna, Liu and Variengien (2023), "How does GPT-2 compute greater-than?", from their code
release https://github.com/hannamw/gpt2-greater-than. That repository has no license file (checked Nov 22, 2026). The same
author's dataset on Hugging Face, https://huggingface.co/datasets/mwhanna/greater-than, is released under the MIT License, but
it uses 45 of these 120 nouns, so it does not cover the whole list. The list is used here with attribution, for
non-commercial research. If the author asks for it to be removed, gt_check.py can take the nouns from his MIT dataset instead
(45 nouns), at the cost of rerunning the greater-than results.
