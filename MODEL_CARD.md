# Model card: resnet18_224

Organ classification on single abdominal CT slices, eleven classes.

| | |
|---|---|
| Version | v0.1.0 |
| Architecture | ResNet18, ImageNet-pretrained, single input channel |
| Input | 224 × 224 greyscale |
| Output | Eleven logits, one per organ class |
| Artefacts | fp32, int8 dynamic, int8 static |
| Licence | MIT (code); see *Data* below for the dataset |

## Intended use

A demonstration of an end-to-end serving pipeline: training, ONNX export,
post-training quantisation, containerised inference, and a browser demo. The
model exists so the pipeline has something to carry.

Reasonable uses are educational and technical: inspecting how the artefacts are
produced and validated, comparing precisions, or as a starting point for a
similar pipeline on another dataset.

### Out of scope

**Not for clinical or diagnostic use.** Nothing here has been validated for
patient care, and no regulatory process has been undertaken.

Also outside what this model supports:

- Raw clinical DICOM. The model expects slices already cropped and windowed the
  way MedMNIST prepared them.
- Any organ outside the eleven classes. The network always returns one of them
  and will assign high confidence to inputs containing none.
- Non-CT imagery, and CT of other body regions.
- Volumetric reasoning. Each slice is classified independently; the model has
  no notion of the scan it came from.

## Data

**OrganAMNIST** from MedMNIST v2 / MedMNIST+, at 224 × 224. Axial slices from
abdominal CT, cropped to individual organs from segmentation masks and labelled
with the organ they contain.

| Split | Images | Source volumes |
|---|---|---|
| Train | 34 561 | 115 |
| Validation | 6 491 | 16 |
| Test | 17 778 | 70 |

Classes: bladder, femur-left, femur-right, heart, kidney-left, kidney-right,
liver, lung-left, lung-right, pancreas, spleen.

**Splits are drawn at CT-scan level, not slice level**, so no patient appears in
more than one split. This matters more than it may look: slice-level splitting
puts adjacent slices of the same patient on both sides of the divide and is the
most common way medical imaging results end up inflated. The official splits are
used unchanged, which also keeps the numbers comparable to the published
baselines.

Class counts are not uniform, which is why balanced accuracy rather than plain
accuracy is used throughout.

### Provenance

Every training run records the archive's SHA-256, the `medmnist` package
version, split sizes, class names and licence into its `metrics.json`, next to
the results. That record is the project's substitute for a data-versioning
tool: the dataset is an immutable public artefact, so the question worth
answering is which bytes produced these numbers.

### Licence and citation

OrganAMNIST is distributed under CC BY 4.0. MedMNIST subsets inherit the licence
of their source dataset, and the authors ask that the source paper be cited
alongside theirs. OrganAMNIST derives from the Liver Tumor Segmentation
Benchmark.

```bibtex
@article{medmnistv2,
  title   = {MedMNIST v2 -- A large-scale lightweight benchmark for 2D and 3D
             biomedical image classification},
  author  = {Yang, Jiancheng and Shi, Rui and Wei, Donglai and Liu, Zequan and
             Zhao, Lin and Ke, Bilian and Pfister, Hanspeter and Ni, Bingbing},
  journal = {Scientific Data},
  volume  = {10}, number = {1}, pages = {41}, year = {2023},
  publisher = {Nature Publishing Group UK London}
}

@article{bilic2023liver,
  title   = {The Liver Tumor Segmentation Benchmark (LiTS)},
  author  = {Bilic, Patrick and Christ, Patrick and Li, Hongwei Bran and
             Vorontsov, Eugene and Ben-Cohen, Avi and Kaissis, Georgios and
             Szeskin, Adi and Jacobs, Colin and
             Mamani, Gabriel Efrain Humpire and Chartrand, Gabriel and others},
  journal = {Medical Image Analysis},
  volume  = {84}, pages = {102680}, year = {2023}, publisher = {Elsevier},
  doi     = {10.1016/j.media.2022.102680}
}
```

## Training

| | |
|---|---|
| Optimiser | AdamW, lr 3e-4, weight decay 0.05 |
| Schedule | Cosine decay, 2 warmup epochs |
| Budget | 30 epochs, fixed; no early stopping |
| Batch size | 128 |
| Precision | bf16 autocast |
| Seed | 42 |
| Selected | Epoch 25, on validation balanced accuracy |

The epoch budget is fixed rather than early-stopped, so runs at different
configurations remain comparable; the best checkpoint is retained regardless.

Augmentation is rotation within ±10° and translation within ±10 % per axis.
**No horizontal flipping**: abdominal organs are not laterally symmetric, and
the label set distinguishes left from right kidney and femur. A flip would map a
sample onto another class's appearance while keeping its original label, which
is label noise rather than augmentation.

Preprocessing is defined once and imported by both training and serving:
greyscale conversion, resize to 224 with bilinear resampling through PIL,
scaling to [0, 1], then standardisation with mean 0.5 and standard deviation
0.5. Those are the constants the MedMNIST reference implementation uses, which
keeps results comparable to the published baselines. Resize and augmentation are
composed into a single affine transform and applied in one resampling step.

The transform parameters travel inside each artefact as ONNX metadata, so they
cannot drift apart from the model they belong to.

## Evaluation

On the validation split, 6 491 images:

| Artefact | Size | Balanced accuracy | Agreement with fp32 | Max logit deviation |
|---|---|---|---|---|
| fp32 | 44.69 MB | 0.9929 | — | — |
| int8 dynamic | 11.23 MB | 0.9931 | 0.9954 | 10.19 |
| int8 static | 11.29 MB | 0.9929 | 0.9978 | 2.65 |

The differences between artefacts are within noise. Around thirty images change
prediction under quantisation, and the marginally higher int8 dynamic figure
reflects a few of them falling the right way rather than any improvement.

Quantisation is validated by agreement rate rather than by numeric closeness.
Quantised logits necessarily differ from fp32; what matters for a deployment is
whether any decision changed. The fp32 artefact itself is checked against the
torch model directly, with a maximum absolute logit deviation of 1.9e-06.

Static quantisation calibrated on 512 training images at seed 42. Calibration
draws from the training split on principle: estimating activation ranges is
fitting parameters to data.

### Test split

**Not yet evaluated.** Selection across artefacts was made on validation alone,
and the test split is read once, after that choice is committed. The commit
history shows the order.

<!-- TODO: fill in after running evaluate.py -->

### Per-class performance

<!-- TODO: per-class recall and confusion matrix, from evaluate.py -->

A single balanced accuracy can hide one class performing badly. For a medical
model that is exactly what a reader needs to be able to check, so per-class
recall belongs here and is not yet reported.

## Deployment

The static int8 artefact is the deployed default. Not for speed — it is only
about eighteen per cent faster than fp32, well short of what int8 delivers on
hardware with VNNI instructions — but for size and consistency. On a free-tier
dyno it holds a 0.9 ms latency spread against 8.4 ms for fp32, at a quarter of
the size.

| Artefact | Median latency | Spread |
|---|---|---|
| fp32 | 33.8 ms | 28.0 – 36.4 |
| int8 dynamic | 221.4 ms | 206 – 247 |
| int8 static | 27.9 ms | 27.5 – 28.4 |

Single-threaded on shared CPU, 224 × 224 input, batch size one. Preprocessing
adds 0.4 to 0.5 ms. Resident memory with all three artefacts loaded: 186 MB.

## Caveats

- Single run at one seed. No confidence intervals.
- One architecture at one resolution. Nothing else has been trained, so the
  reported figures say nothing about how the choice compares to alternatives.
- The near-ceiling accuracy reflects the benchmark, not the difficulty of organ
  recognition in practice. OrganAMNIST slices are pre-cropped around a single
  organ; the model is not localising anything.
- Confidence is uncalibrated. High softmax probabilities should not be read as
  reliability, particularly on inputs unlike the training distribution.
- Behaviour on out-of-distribution input is undefined and untested. The model
  has no reject option and will confidently classify an image of anything.
