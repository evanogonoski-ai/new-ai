# V2.2 Experiment Plan: Generalization Test

## Motivation

V2.1 showed a consistent ~40% loss advantage for Resonance, but both models hit PPL ~1.1 because the corpus is small enough (~3.8 tokens/param) to memorize. We need a test that forces generalization — where the 40% loss advantage can manifest as a PPL difference.

## Step A: Author-Holdout Generalization Test

**Goal**: Hold out entire authors the model never sees during training. Evaluate on those held-out authors. This tests whether the model can generalize to unseen writing styles — where adaptive attention should matter.

### Holdout Authors (selected by Chief Scientist)
| Author | Category | Files | Size |
|--------|----------|-------|------|
| Plato | philosophy | `plato_apology.txt`, `plato_republic.txt`, `plato_symposium.txt` | 1.5 MB |
| Jane Austen | fiction | `jane_austen_pride_and_prejudice.txt` | 698 KB |
| Charles Darwin | science | `charles_darwin_origin_of_species.txt` | 951 KB |

**Total holdout**: ~3.1 MB (~17% of corpus). Each of the 3 real categories has a holdout representative. Synthetic and quran categories are unaffected.

**Remaining training data**: ~15.6 MB — still large enough for meaningful training, with 7+ other authors across philosophy, fiction, and science.

### Implementation Changes

#### 1. Modify `build_corpus.py` → produce train + holdout JSONL
- Add a `HOLDOUT_SOURCES` list mapping source prefixes to holdout status:
  ```python
  HOLDOUT_SOURCES = ['plato_', 'jane_austen_', 'charles_darwin_']
  ```
- Output two files instead of one:
  - `data/train_corpus.jsonl` — everything except holdout authors
  - `data/holdout_corpus.jsonl` — only holdout authors
- Add an `"author_holdout": true/false` field to each JSONL entry for traceability
- Print stats showing train vs holdout split per category

#### 2. Modify `category_dataset.py` → support loading train vs holdout
- Add `pretokenize_corpus()` call that accepts a JSONL path parameter (already does this)
- Produce two `.pt` files:
  - `data/train_tokenized.pt`
  - `data/holdout_tokenized.pt`
- No structural changes to `CategoryTextDataset` needed — it's already generic

#### 3. Create `neurogenesis_v2/scripts/train_v22_generalization.py`
New training script that:
- Loads **train** dataset for training (no holdout contamination)
- Loads **holdout** dataset for generalization evaluation
- Trains both Resonance and Baseline on the **same train split**
- Evaluates both on:
  - **Train set** (to confirm memorization behavior matches V2.1)
  - **Holdout set** (the real test — per-author PPL comparison)
- Logs per-author holdout metrics: loss, PPL, gates, iterations
- Produces analysis report comparing:
  - Train PPL: Resonance vs Baseline (expect both ~1.1, same as V2.1)
  - Holdout PPL: Resonance vs Baseline (**this is the key metric**)
  - Per-author holdout breakdown (Plato, Austen, Darwin separately)
  - Gate/iteration adaptation on holdout vs train (does gating change on unseen text?)

#### 4. Training configuration
- Same locked config as V2.1 (no architecture changes)
- Same 15,000 steps, lr=3e-4, batch_size=32
- Mitosis enabled (same as V2.1 default)
- Only data split changes — pure experiment design improvement

### Expected Outcomes
- **If entropy gating helps generalization**: Resonance will show meaningfully lower PPL on holdout authors than Baseline, even though both are similar on training data. The adaptive gates should activate differently on novel writing styles.
- **If no difference**: Both models generalize equally poorly/well, suggesting the architecture advantage is about optimization efficiency rather than generalization.

### Key Metrics to Report
1. Train PPL: Resonance vs Baseline (sanity check — should match V2.1)
2. **Holdout PPL: Resonance vs Baseline** (primary metric)
3. PPL gap: (Holdout PPL - Train PPL) for each model (generalization penalty)
4. Per-author holdout PPL breakdown
5. Gate/iteration stats: holdout vs train (adaptive behavior signal)

---

## Step B: Corpus Scaling (follow-up, not implemented now)

**Goal**: Scale corpus to 50-100M tokens while keeping 1.25M param model. Forces generalization because the model literally can't memorize everything.

**Not implemented in this plan** — Step A is faster and tests the same hypothesis. If Step A shows a clear generalization advantage, Step B provides the definitive scaling confirmation. If Step A shows no difference, Step B may not be worth the compute.

### What it would require (scoping only)
- Expand `corpus/raw/` with more Gutenberg texts (easy to source)
- Retrain tokenizer on expanded corpus (or verify existing tokenizer coverage)
- Same training script structure as Step A but with larger dataset
- May need more training steps (50K-100K) for convergence
- Token-to-parameter ratio would go from ~3.8 to ~40-80 (real generalization pressure)

---

## Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `neurogenesis_v2/training/build_corpus.py` | **Modify** | Add holdout split logic, output train + holdout JSONL |
| `neurogenesis_v2/training/category_dataset.py` | **Minor modify** | Add convenience function to load train/holdout pair |
| `neurogenesis_v2/scripts/train_v22_generalization.py` | **Create** | New training script with holdout evaluation |
| `data/train_corpus.jsonl` | **Generated** | Training data (no holdout authors) |
| `data/holdout_corpus.jsonl` | **Generated** | Holdout authors only |
| `data/train_tokenized.pt` | **Generated** | Pre-tokenized training set |
| `data/holdout_tokenized.pt` | **Generated** | Pre-tokenized holdout set |

## Execution Order
1. Modify `build_corpus.py` with holdout split
2. Run `build_corpus.py` to generate train + holdout JSONL
3. Modify `category_dataset.py` with convenience loader
4. Create `train_v22_generalization.py`
5. Run the experiment: `python neurogenesis_v2/scripts/train_v22_generalization.py --mode both --device cpu`
6. Analyze results and save report to `logs/v2.2_generalization_report.txt`
