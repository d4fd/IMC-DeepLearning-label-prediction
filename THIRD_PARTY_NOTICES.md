# Upstream software and weights

The team code in this repository is MIT licensed. That license grants no rights
to private study data, third-party software or pretrained model weights.

- **VirTues**, Bunne Lab: <https://github.com/bunnelab/virtues>.
  The workflow was developed against commit
  `229688bba5cac5395cffa729dcf230e937da005d`. Upstream source is MIT licensed;
  consult its original license and acknowledgements when installing it.
- **Spora IO**, Bunne Lab: <https://github.com/bunnelab/spora-io>.
  The observed development revision was
  `2215b9d684a0d00e3d8401cd48898d67f0637ca9`. Consult upstream terms.
- **VirTues `virtues-sp32` pretrained weights**:
  <https://huggingface.co/bunnelab/virtues>. Their separate license is
  **CC BY-NC 4.0**, not MIT. Download separately only under applicable terms.
- Marker preparation uses the upstream VirTues sequence/embedding utilities and
  ESM2 (`esm2_t30_150M_UR50D`); their dependencies and externally downloaded
  sequences/models retain their respective terms.
- The reconstruction check follows the upstream VirTues demonstration with a
  project-owned T4 SDPA adapter. No upstream dataset, vendor tree, sequence,
  embedding or model-weight file is redistributed here.

Package dependency licenses remain those of their respective authors. This is
a source showcase, not an assertion that all dependency licenses are identical.

## VirTues notice for adapted reconstruction logic

The adapted upstream reconstruction example retains the following original
copyright and permission notice from the pinned VirTues source:

```text
MIT License

Copyright (c) ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland,
Laboratory of Artificial Intelligence in Molecular Medicine, 2025

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
