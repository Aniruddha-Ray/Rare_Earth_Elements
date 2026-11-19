Geologically Constrained Deep Learning for Mineral Prospectivity Mapping



Research direction: From Hard-Constrained CNNs to a Geology-Aware Hybrid CNN–Vision Transformer

Status: Research in Progress / Prototype Validation



## Quick start (code and data)

```bash
# 1. environment (from the repo root, inside the REE venv)
pip install -r requirements.txt
pip install -e .

# 2. real dataset: northern Curnamona Province REE (Luo et al. 2026, DEEP-SEAM; Zenodo 10.5281/zenodo.17098677)
python scripts/download_curnamona.py      # -> data/external/curnamona/ (md5-verified, gitignored)
python scripts/prepare_curnamona.py --export-tifs

# 3. tests (shapes, gradient flow, prior fitting, data alignment)
pytest

# 4. experiments: leave-one-deposit-out CV + full prospectivity map
python scripts/run_experiment.py configs/shvcnn.json
```

Notebooks (thin drivers over the `geompm` package):

* `notebooks/01_brs_data_prep.ipynb`: Rajasthan BRS soil survey (fixed parsing, labels, leakage, spatial CV)
* `notebooks/02_curnamona_priors.ipynb`: Curnamona evidence layers, w and g priors
* `notebooks/03_models_and_training.ipynb`: SHCNN / geology-aware ViT / SHVCNN sanity checks and a short training run

**Dataset note.** Curnamona has 35 aligned evidence rasters (~0.2 km cells, 18,496 cells in the study area), faults and felsic granites (w), 615 multi-element rock samples (g) and **7 REE occurrences**. With so few positives, evaluation is leave-one-deposit-out with map-based metrics (area rank, success-rate curve), not a random train/test split. The fitted distance-decay exponents are weak (|k| < 0.3), so ablations with and without each prior are essential.

### Model comparison grid (`configs/`)

| config | architecture | hard w | geo-attention | g (FFN/head) | soft GeoLoss |
|---|---|---|---|---|---|
| cnn | CNN | - | - | - | - |
| cnn_soft_only | CNN | - | - | - | yes |
| shcnn | CNN | yes | - | head | yes |
| vit | ViT | - | - | - | - |
| geovit | ViT | - | product | yes | yes |
| cnn_vit | CNN+ViT | - | - | - | - |
| shvcnn_hard_only | CNN+ViT | yes | product | yes | - |
| shvcnn | CNN+ViT | yes | product | yes | yes |
| shvcnn_learned_affinity | CNN+ViT | yes | learned MLP | yes | yes |



Overview



Mineral Prospectivity Mapping (MPM) aims to identify areas that are more likely to contain economically significant mineralization using geological, geochemical, geophysical, and spatial evidence.



Conventional deep learning models can learn complex statistical relationships from exploration data, but they do not automatically respect geological principles. A model may therefore achieve strong predictive performance while producing predictions that are difficult to justify geologically.



This research investigates how geological domain knowledge can be embedded directly into deep learning, rather than treating geological information only as another input feature.



The work begins with a geologically constrained CNN using both soft and hard constraints and then extends the same philosophy toward a geology-aware Vision Transformer (ViT) and a proposed hybrid CNN–ViT architecture.



Research Motivation



A conventional CNN learns statistical relationships between input data and mineralization labels:



Geological / geochemical data

&#x20;           ↓

&#x20;      CNN learning

&#x20;           ↓

&#x20;       Prediction



This creates several challenges:



False positives may occur in geologically unfavorable regions.



High predictive accuracy does not necessarily imply geological plausibility.



Black-box predictions can be difficult for domain experts to trust.



CNNs primarily capture local spatial patterns, whereas geological structures can extend across large regions.



The central research question is:



Can geological knowledge be embedded into the internal learning and reasoning mechanisms of deep neural networks so that predictions become both data-driven and geologically informed?



1\. Geological Knowledge



Two complementary geological priors are used.



1.1 Spatial Prior — w



Mineralization density generally decreases as distance from an ore-controlling factor increases. A power-law relationship is used:



$$

w = \\frac{N d^k}{m\_{max}}

$$



where:



d = distance to an ore-controlling factor



N = proportionality constant



k = slope obtained from fitting log(m) against log(d)



m = mineralization density at distance d



m\_max = maximum mineralization density



Taking logarithms gives:



$$

\\log m = \\log N + k\\log d

$$



For a typical distance-decay relationship, k is negative. The resulting normalized w map acts as a spatial geological confidence/favorability map.



1.2 Geochemical Prior — g



Multi-element geochemical data may contain variables such as:



Au, Ag, Cu, Pb, W, Mo, ...



Principal Component Analysis (PCA) is used to compress correlated geochemical variables. A PCA component associated with mineralization is used as a latent geochemical indicator g.



The important distinction is:



w → spatial geological information

g → geochemical / semantic information



2\. Soft Geological Constraint



The standard CNN objective is augmented with a geological penalty:



$$

L\_{total}=L\_{CNN}+\\lambda L\_{geology}

$$



The geological loss compares the prediction with a geological prior representation:



f\_{sigmoid}\\left(\\sum\_i aw\_i+b\\right)

\\right|\_2

$$



The L2 norm measures the magnitude of disagreement between the prediction and the geological representation.



Conceptually:



CNN prediction

&#x20;     ↓

compare with geological prior

&#x20;     ↓

geological penalty

&#x20;     ↓

gradient update



The important point is that the soft constraint changes the optimization objective, not the forward architecture itself.



3\. Hard Geological Constraint



Hard constraints inject geological knowledge directly into the forward pass.



3.1 Hard-Constrained Convolution



The constrained feature map is:



$$

F\_g(x)=w\\left(ur(x)+v\\right)

$$



where:



r(x) = incoming feature map



u = trainable convolution weight



v = trainable bias



w = spatial geological prior



Thus:



CNN feature map

&#x20;     ×

spatial prior w

&#x20;     ↓

geologically modulated features



Geologically favorable regions can be amplified while unfavorable regions are suppressed.



3.2 Hard-Constrained Classifier



The classifier incorporates the geochemical prior:



$$

C\_g(x)=g(nl(x)+t)

$$



where l(x) is the flattened feature representation and g is the PCA-derived geochemical indicator.



Therefore the final decision uses both learned features and geochemical evidence.



4\. SHCNN Prototype



The Soft + Hard Constrained CNN combines both approaches:



Input

&#x20; ↓

Conv1 → ReLU → × w

&#x20; ↓

Conv2 → ReLU → × w

&#x20; ↓

MaxPool

&#x20; ↓

Flatten

&#x20; ↓

FC1 → × g

&#x20; ↓

FC2

&#x20; ↓

Sigmoid

&#x20; ↓

Mineralization probability



Training additionally uses:



L\_total = L\_CNN + λ L\_geology



Thus:



Hard constraint: geology affects the forward computation.



Soft constraint: geology affects optimization.



They are complementary rather than alternative approaches.



5\. Why Vision Transformers?



CNNs are strong at local feature extraction, but geological relationships can extend over large spatial distances:



fault systems



mineral belts



structural corridors



distant but geologically similar regions



Vision Transformers divide an image into patches and use self-attention to allow patches to interact directly.



Image

&#x20; ↓

Patch embedding

&#x20; ↓

Tokens

&#x20; ↓

Self-attention

&#x20; ↓

Global contextual representation



This motivates extending geological constraints into the Transformer.



6\. Geology-Aware Vision Transformer



The guiding principle is:



Use each geological prior where its mathematical nature matches the model mechanism.



Prior



Nature



Proposed location



w



Spatial



CNN modulation + ViT attention



g



Geochemical / semantic



ViT FFN + classifier



7\. Geological Attention



Standard self-attention is:



$$

Attention(Q,K,V)=

softmax\\left(\\frac{QK^T}{\\sqrt{d\_k}}\\right)V

$$



The proposed geological version adds an affinity matrix M:



$$

Attention\_{geo}(Q,K,V)=

softmax\\left(

\\frac{QK^T}{\\sqrt{d\_k}}+\\alpha M

\\right)V

$$



where:



M = geological affinity matrix



α = learnable scaling parameter



The geological prior therefore acts as an attention bias.



Why addition rather than multiplication?



A multiplicative formulation such as:



$$

A' = M \\odot A

$$



can strongly suppress learned attention relationships.



The additive formulation:



$$

A'=A+\\alpha M

$$



preserves the learned attention structure while gently steering it using geological information. This is more appropriate because the geological prior is empirical rather than an absolute rule.



8\. Geological Affinity Matrix



Each patch receives a spatial confidence value w\_i.



A simple affinity matrix is:



$$

M\_{ij}=w\_iw\_j

$$



Two highly favorable patches therefore receive a strong geological affinity.



A more flexible learnable formulation is:



$$

M\_{ij}=MLP(w\_i,w\_j,|w\_i-w\_j|)

$$



This allows the model to learn nonlinear geological compatibility between two patches.



The difference term:



$$

|w\_i-w\_j|

$$



provides information about how similar the two geological confidence values are.



9\. Geochemical Modulation in the ViT FFN



The Transformer Feed Forward Network performs semantic feature transformation.



Since g is a geochemical/mineralization indicator rather than a pairwise spatial relationship, it is proposed as a conditioning signal inside the FFN.



Conceptually:



Tokens

&#x20; ↓

Geological self-attention using w

&#x20; ↓

GELU / FFN

&#x20; ↓

Geochemical modulation using g

&#x20; ↓

Next representation



This follows the principle of placing spatial knowledge in spatial mechanisms and semantic knowledge in semantic feature transformation.



10\. Proposed SHVCNN Architecture



The proposed research architecture combines hierarchical CNN and ViT stages:



Input

&#x20; ↓

CNN Stage 1

&#x20; ↓

Hard geological modulation using w

&#x20; ↓

ViT Stage 1

&#x20; ├── Geological Attention using w

&#x20; └── FFN modulation using g

&#x20; ↓

CNN Stage 2

&#x20; ↓

Hard geological modulation using w

&#x20; ↓

ViT Stage 2

&#x20; ├── Geological Attention using w

&#x20; └── FFN modulation using g

&#x20; ↓

CNN Stage 3

&#x20; ↓

Hard geological modulation using w

&#x20; ↓

ViT Stage 3

&#x20; ├── Geological Attention using w

&#x20; └── FFN modulation using g

&#x20; ↓

Decoder

&#x20; ↓

Prospectivity Map



The CNN stages provide progressively richer local features, while the ViT stages model increasingly global spatial relationships.



11\. CNN–ViT Tensor Flow



A CNN feature tensor typically has shape:



(B, C, H, W)



It can be converted into Transformer tokens:



(B, C, H, W)

&#x20;      ↓

(B, H×W, C)

&#x20;      ↓

Linear projection

&#x20;      ↓

(B, N, D)



where:



B = batch size



N = number of tokens/patches



D = Transformer embedding dimension



After Transformer processing, tokens can be reshaped back into spatial features:



(B, N, D)

&#x20;      ↓

(B, D, H, W)



This allows another CNN stage to process the Transformer output.



12\. Local + Global Geological Reasoning



The proposed hybrid architecture is designed around complementary strengths.



CNN



local textures



local geological structures



spatial feature extraction



explicit w-based modulation



ViT



long-range dependencies



regional geological context



patch-to-patch relationships



w-based geological attention



Together:



CNN = local geological precision

ViT = global geological context



13\. Current Prototype Status



SHCNN



The SHCNN prototype has been used to validate:



hard spatial modulation using w



hard geochemical modulation using g



geological loss



gradient-based training compatibility



A synthetic training loop was used to verify that the model can train and that the loss decreases.



SHVCNN



The SHVCNN prototype validates:



CNN-based feature projection



CNN-to-token transformation



geological attention



geological affinity construction



FFN modulation using g



forward propagation



backward propagation



Important scope



The current SHVCNN prototype was validated as a classifier using synthetic data. The proposed decoder-based dense prospectivity mapping stage is intended for subsequent experiments with spatially labeled real geoscience data.



Synthetic validation demonstrates architectural feasibility; it does not establish real-world mineral prospectivity performance.



14\. Experimental Roadmap



Real Dataset



The next stage should use real:



geological maps



ore-controlling structures



mineral occurrence data



distance maps



multi-element geochemistry



spatial labels



Baselines



Recommended comparisons:



CNN

ViT

CNN + ViT

Soft-constrained CNN

Hard-constrained CNN

SHCNN

Geology-aware ViT

SHVCNN



Ablation Studies



Important experiments include:



Soft only

Hard only

Soft + Hard



CNN only

ViT only

CNN + ViT



Without w

With w



Without g

With g



Fixed geological affinity

Learnable geological affinity



Evaluation



Possible metrics include:



Precision



Recall



F1-score



ROC-AUC



PR-AUC



spatial localization quality



false-positive distribution



geological consistency



Qualitative evaluation with domain experts is also important.



15\. Limitations



Static priors



The current w and g are precomputed. Future work could allow the model to learn corrections to these priors.



Attention complexity



Dense geological attention requires an N × N affinity matrix, which can become expensive for high-resolution maps.



Potential solutions include:



windowed attention



sparse attention



hierarchical attention



low-rank affinity



graph-based geological attention



Empirical geological law



The distance-decay relationship is an empirical prior, not an absolute physical law. It should therefore guide the model rather than completely dictate its predictions.



Real-data validation



Synthetic validation demonstrates that the architecture is trainable but cannot establish geological predictive performance.



16\. Future Research



Potential extensions include:



Learnable Geological Priors



Initial geological prior

&#x20;       ↓

Neural refinement

&#x20;       ↓

Learned geological representation



Multi-Mineral Prospectivity



Different minerals may require different geological relationships. A future model could learn mineral-specific priors.



Geological Graph Attention



Ore-controlling structures can naturally be represented as spatial graphs, motivating:



CNN + ViT + GNN



Explainable Geological Attention



Attention maps and w maps can be visualized alongside geological maps to determine whether the learned relationships are geologically meaningful.



17\. Research Contribution



The research follows this progression:



Geologically constrained CNN

&#x20;         ↓

Understand soft + hard constraints

&#x20;         ↓

Implement SHCNN

&#x20;         ↓

Identify limitation of local CNN reasoning

&#x20;         ↓

Introduce Vision Transformer

&#x20;         ↓

w → geological attention

&#x20;         ↓

g → FFN modulation

&#x20;         ↓

Combine CNN + geology-aware ViT

&#x20;         ↓

Proposed SHVCNN

&#x20;         ↓

Real-data evaluation



The objective is not simply to replace CNNs with Transformers. Instead, the research attempts to preserve the domain-knowledge constraint philosophy while adding global reasoning capability.



18\. Repository Structure

```
.
├── README.md
├── pyproject.toml                 # installable package `geompm` (pip install -e .)
├── requirements.txt
├── configs/                       # one JSON per model / ablation
├── scripts/
│   ├── download_curnamona.py      # Zenodo download + md5 check
│   ├── prepare_curnamona.py       # build and cache the model-ready grid
│   └── run_experiment.py          # LODO CV + full map -> experiments/results/<name>/
├── src/geompm/
│   ├── config.py                  # dataclass configs (data, prior, model, train)
│   ├── geology/
│   │   ├── spatial_prior.py       # w = N d^k / m_max (Poisson-ML fit of k)
│   │   ├── geochem_prior.py       # g: CLR -> PCA -> indicator component -> IDW grid
│   │   └── geo_loss.py            # L = BCE + lambda * ||p - sigmoid(a w + b)||
│   ├── models/
│   │   ├── modulation.py          # hard constraints: F * w, l(x) * g
│   │   ├── shcnn.py
│   │   ├── geology_attention.py   # softmax(QK^T/sqrt(d) + alpha M)
│   │   ├── geology_vit.py         # GeoFFN, GeoTransformerBlock, GeoViTStage, GeoViT
│   │   ├── shvcnn.py
│   │   └── factory.py             # build_model(cfg) for every baseline / ablation
│   ├── data/
│   │   ├── raster.py              # alignment, rasterisation, km distance maps
│   │   ├── curnamona.py           # GeoGrid builder for the Curnamona REE dataset
│   │   ├── patches.py             # patch sampling (PU labels) + dihedral augmentation
│   │   └── brs.py                 # Rajasthan BRS survey: parsing and fold-safe features
│   └── training/
│       ├── trainer.py, predict.py # training loop, sliding-window map
│       ├── evaluation.py          # area rank, success-rate curve, AUCs
│       └── experiment.py          # leave-one-deposit-out CV
├── notebooks/
├── tests/
├── data/                          # gitignored: raw/, external/, processed/
└── experiments/results/           # gitignored outputs
```



19\. Reproducibility



Future experiments should record:



dataset version



preprocessing procedure



spatial resolution



patch size



model configuration



CNN/ViT depth



embedding dimension



number of attention heads



optimizer



learning rate



λ for GeoLoss



geological prior generation method



random seed



train/validation/test split



evaluation metrics



All comparisons should use identical preprocessing and data splits wherever possible.



20\. Research Status



Component



Status



Spatial prior w



Established



PCA geochemical prior g



Established



Soft geological loss



Implemented / validated



Hard-constrained CNN



Implemented / validated



SHCNN



Prototype validated



Geological attention



Prototype implemented



FFN modulation using g



Prototype implemented



SHVCNN forward pass



Validated



SHVCNN backward pass



Validated



Real geological dataset



Integrated (Curnamona REE, 7 occurrences)



Dense prospectivity decoder



Proposed (dense maps currently via sliding-window patch classification)



Ablation study



Configs ready (configs/), runs pending



Real-data quantitative evaluation



Pending



References



The project is primarily grounded in literature on:



Geologically constrained CNNs for mineral prospectivity mapping.



Geological constraints for deep learning-based prospectivity mapping.



CNN–Transformer approaches for mineral prospectivity mapping.



Vision Transformers.



Transformer self-attention.



Transformer-based dense prediction and segmentation.



PCA-based geochemical feature extraction.



Specific bibliographic entries should be maintained in a dedicated references.bib file and cited consistently in the final report.



Conclusion



This research explores a geology-aware approach to mineral prospectivity mapping in which domain knowledge is embedded directly into the learning process.



The original geologically constrained CNN demonstrates that geological information can influence both feature extraction and classification. The proposed extension carries the same philosophy into Vision Transformers.



The spatial prior w is used in spatial mechanisms such as convolution and attention, while the geochemical prior g is used to influence semantic feature transformation and classification. A geological loss additionally encourages predictions to remain consistent with the supplied prior.



The resulting architecture aims to combine:



local feature extraction + global contextual reasoning + geological domain knowledge.



The current synthetic prototypes establish architectural and gradient-flow feasibility. The next decisive milestone is evaluation on real geological and geochemical datasets, followed by controlled ablation studies and comparison with established CNN, ViT, and geologically constrained baselines.

