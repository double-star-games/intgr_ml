# IntgrML CLI Reference Specification

**Version**: 1.2.1
**Binary**: `intgrmlc`
**Purpose**: Unified command-line interface for all IntgrML modules

---

## 1. Global Structure

### Syntax
```
intgrmlc <module> <action> [options]
```

### Modules
| Module | Description | Actions | Output Format |
|--------|-------------|---------|---------------|
| `boost` | Gradient boosted trees (binary/regression) | train, predict, determinism, export, parity | .sbf, .mcu |
| `boost-ovr` | One-vs-Rest multiclass boosting (v1.2+) | train, predict | .ovr |
| `forest` | Random forests | train, predict | .sff |
| `linear` | Linear/logistic regression | train, predict | .slf |
| `bayes` | Naive Bayes classifier | train, predict | .sbn |
| `reduce` | Principal component analysis (PCA) | train, transform | .srf |
| `edge` | Universal model inference (auto-detect) | predict | N/A |

### Global Flags
| Flag | Description |
|------|-------------|
| `--help`, `-h` | Show help (global or module-specific) |

### Common Options (All Modules)
| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--data <path>` | string | required | Input CSV file path |
| `--out <path>` | string | required* | Output file path (model or predictions) |
| `--model <path>` | string | required† | Input model file (predict/transform only) |
| `--label <col>` | int/string | -1 | Label column (index or name, -1=last) |
| `--no-header` | flag | false | CSV has no header row |
| `--seed <n>` | uint32 | 42 | Random seed for reproducibility |

\* Required for train/transform, optional for predict
† Required for predict/transform only

### Quantization Options (boost, boost-ovr, forest, bayes)
| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--clip-min <f>` | float | -10.0 | Minimum feature clipping value |
| `--clip-max <f>` | float | 10.0 | Maximum feature clipping value |
| `--precision <f>` | float | 0.02 | Quantization bin width |
| `--per-feature-clips <path>` | string | null | JSON file with per-feature [min,max] arrays |

---

## 2. Module: `boost`

**Purpose**: Binary classification or regression using gradient boosted decision trees
**Format**: Q16.16 fixed-point accumulator, int8 quantization
**File Extension**: `.sbf`

### Train Action
```
intgrmlc boost train --data <csv> --out <sbf> [options]
```

#### Options
| Option | Type | Range | Default | Description |
|--------|------|-------|---------|-------------|
| `--trees <n>` | uint32 | 1-10000 | 100 | Number of boosting trees |
| `--depth <n>` | uint16 | 1-20 | 6 | Maximum tree depth |
| `--min-leaf <n>` | uint16 | 1-∞ | 8 | Minimum samples per leaf node |
| `--lr <f>` | float | 0.0-1.0 | 0.125 | Learning rate (shrinkage) |
| `--accum <format>` | string | q16_16 | q16_16 | Accumulator format (v1.1+) |
| `--task <type>` | string | binary\|regression | binary | Task type |

#### Constraints
- `--trees`: Must be 1-10000
- `--depth`: Must be 1-20
- `--lr`: Must be 0.0-1.0
- Labels: Binary {0,1} for classification, continuous for regression

### Predict Action
```
intgrmlc boost predict --model <sbf> --data <csv> [--out <csv>]
```

#### Options
| Option | Description |
|--------|-------------|
| `--out <path>` | Output predictions CSV (optional, defaults to stdout) |

#### Output Format
CSV with single column:
```
prediction
-15
23
-8
...
```
Values are Q16.16 fixed-point logits (binary) or continuous predictions (regression).

### Determinism Action (v1.2.1+)
```
intgrmlc boost determinism --data <csv> --out <sbf> --runs <n> [options]
```

**Purpose**: Verify bit-exact deterministic training across multiple runs

#### Options
| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--runs <n>` | uint32 | 3 | Number of training runs to compare |
| All training options | - | - | Same as `train` action |

#### Behavior
1. Trains N identical models with same seed and hyperparameters
2. Computes SHA-256 hash of each serialized model
3. Compares hashes for bit-exact equality
4. Outputs: Model file (first run), hash verification results

#### Output Example
```
Run 1: 9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08
Run 2: 9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08
Run 3: 9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08
✓ All hashes match - deterministic training verified
```

#### Exit Codes
- 0: All hashes match (deterministic)
- 5: Hash mismatch detected (non-deterministic)

### Export Action (v1.2.1+)
```
intgrmlc boost export --model <sbf> --out <mcu>
```

**Purpose**: Export IntgrBoost model to MCU-optimized binary format for embedded deployment

#### Options
| Option | Type | Description |
|--------|------|-------------|
| `--model <sbf>` | string | Input IntgrBoost model file |
| `--out <mcu>` | string | Output MCU binary file |

#### MCU Format Characteristics
- **File Extension**: `.mcu`
- **Header**: 40 bytes (includes learning rate, bias, checksum)
- **Endianness**: Little-endian
- **Compression**: None (optimized for embedded parsing)
- **Size**: Typically 10-30% smaller than .sbf format

#### Output Format (Binary)
```
Header (40 bytes):
  - Magic: "INTGMCU1" (8 bytes)
  - model_type: u16 (0=boost, 1=forest, 2=ovr)
  - version: u16
  - num_trees: u32
  - num_features: u32
  - num_classes: u32
  - learning_rate_fp: i32 (Q8.8)
  - prescaled: u8 (1 if leaves pre-scaled)
  - reserved: u8[3]
  - bias_fp: i32 (Q0)
  - checksum: u32 (CRC-32)

Tree Data:
  - Serialized trees (variable length)
```

#### Use Cases
- Deploy models on microcontrollers (STM32, ESP32, etc.)
- IoT edge devices with RAM constraints
- Real-time embedded inference

### Parity Action (v1.2.1+)
```
intgrmlc boost parity --model <sbf> --data <csv> [--samples <n>]
```

**Purpose**: Verify bit-exact parity between desktop (.sbf) and MCU (.mcu) inference

#### Options
| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--model <sbf>` | string | required | IntgrBoost model file |
| `--data <csv>` | string | required | Test data for parity check |
| `--samples <n>` | uint32 | 100 | Number of samples to verify |

#### Behavior
1. Exports model to temporary MCU format
2. Runs desktop inference on N samples
3. Runs MCU inference on same N samples
4. Compares predictions sample-by-sample
5. Reports match percentage and first mismatch (if any)

#### Output Example (Success)
```
Loading model: model.sbf
Exporting to MCU format...
Testing parity on 100 samples...

Desktop vs MCU Predictions:
  Sample 0: Desktop=-46, MCU=-46 ✓
  Sample 1: Desktop=23, MCU=23 ✓
  Sample 2: Desktop=-10, MCU=-10 ✓
  ...

Parity: 100/100 samples match (100.00%)
✓ Bit-exact parity verified
```

#### Output Example (Failure)
```
Loading model: model.sbf
Exporting to MCU format...
Testing parity on 100 samples...

Desktop vs MCU Predictions:
  Sample 0: Desktop=-46, MCU=-10 ✗ MISMATCH
  Sample 1: Desktop=23, MCU=5 ✗ MISMATCH
  ...

Parity: 0/100 samples match (0.00%)
✗ Parity verification FAILED
First mismatch at sample 0: Desktop=-46, MCU=-10
```

#### Exit Codes
- 0: 100% parity (all samples match)
- 5: Parity failure (any mismatch detected)

#### Notes
- This command is critical for validating embedded deployment
- Mismatches indicate bugs in MCU export or inference
- Run this before deploying models to production embedded systems

---

## 3. Module: `boost-ovr`

**Purpose**: K-class multiclass classification using One-vs-Rest
**Architecture**: K independent binary IntgrBoost models
**Format**: Q16.16 accumulator per head, int8 quantization
**File Extension**: `.ovr`

### Train Action
```
intgrmlc boost-ovr train --data <csv> --out <ovr> [options]
```

#### Options
| Option | Type | Range | Default | Description |
|--------|------|-------|---------|-------------|
| `--trees <n>` | uint32 | 1-10000 | 100 | Trees per head (total = K × trees) |
| `--depth <n>` | uint16 | 1-20 | 6 | Maximum tree depth |
| `--min-leaf <n>` | uint16 | 1-∞ | 8 | Minimum samples per leaf node |
| `--lr <f>` | float | 0.0-1.0 | 0.125 | Learning rate (shrinkage) |
| `--no-parallel` | flag | false | Disable parallel head training |

#### Constraints
- Labels: Integer class indices {0, 1, ..., K-1}
- Minimum K: 2 classes
- Maximum K: No hard limit (memory constrained)
- Seeding: Each head uses `seed XOR class_id` for determinism

#### Training Behavior
- Automatically detects K classes from label distribution
- Trains K binary classifiers (one vs. rest)
- Parallel training enabled by default (disable with `--no-parallel`)
- Shows per-class accuracy breakdown after training

### Predict Action
```
intgrmlc boost-ovr predict --model <ovr> --data <csv> [--out <csv>] [--logits]
```

#### Options
| Option | Description |
|--------|-------------|
| `--out <path>` | Output predictions CSV (optional) |
| `--logits` | Include all K logits in output (default: class only) |

#### Output Format
**Default** (class predictions only):
```
prediction
0
2
1
...
```

**With `--logits`**:
```
prediction,logit_class_0,logit_class_1,logit_class_2
0,127,-45,-89
2,-56,-72,189
1,-34,145,-67
...
```
Logits are Q16.16 fixed-point values. Prediction = argmax(logits).

---

## 4. Module: `forest`

**Purpose**: Random forest ensemble classifier
**File Extension**: `.sff`

### Train Action
```
intgrmlc forest train --data <csv> --out <sff> [options]
```

#### Options
| Option | Type | Range | Default | Description |
|--------|------|-------|---------|-------------|
| `--trees <n>` | uint32 | 1-10000 | 100 | Number of trees in forest |
| `--depth <n>` | uint16 | 1-20 | 8 | Maximum tree depth |
| `--min-leaf <n>` | uint16 | 1-∞ | 10 | Minimum samples per leaf |
| `--max-features <n>` | uint16 | 1-n_features | √n | Features sampled per split |
| `--subsample <f>` | float | 0.0-1.0 | 0.8 | Row sampling ratio (bootstrap) |

#### Constraints
- `--max-features`: Defaults to ⌊√(n_features)⌋
- Labels: Binary {0,1} or multiclass (label scaling)

### Predict Action
```
intgrmlc forest predict --model <sff> --data <csv> [--out <csv>]
```

#### Output Format
CSV with integer predictions (class labels).

---

## 5. Module: `linear`

**Purpose**: Linear regression or logistic regression
**Format**: Fixed-point weights with sigmoid LUT (logistic)
**File Extension**: `.slf`

### Train Action
```
intgrmlc linear train --data <csv> --out <slf> [options]
```

#### Options
| Option | Type | Range | Default | Description |
|--------|------|-------|---------|-------------|
| `--task <type>` | string | logistic\|regression | logistic | Task type |
| `--lr <f>` | float | >0.0 | 0.01 | Learning rate (gradient descent) |
| `--l2 <f>` | float | ≥0.0 | 0.0 | L2 regularization coefficient |
| `--epochs <n>` | uint32 | 1-∞ | 100 | Maximum training epochs |
| `--batch <n>` | uint32 | 0-∞ | 32 | Batch size (0=full batch) |

#### Constraints
- `--batch 0`: Full batch gradient descent
- Labels: Binary {0,1} for logistic, continuous for regression

### Predict Action
```
intgrmlc linear predict --model <slf> --data <csv> [--out <csv>]
```

#### Output Format
- Logistic: Probabilities [0-255] (uint8 scaled)
- Regression: Continuous predictions

---

## 6. Module: `bayes`

**Purpose**: Naive Bayes classifier with integer arithmetic
**File Extension**: `.sbn`

### Train Action
```
intgrmlc bayes train --data <csv> --out <sbn> [options]
```

#### Options
| Option | Type | Range | Default | Description |
|--------|------|-------|---------|-------------|
| `--alpha <f>` | float | ≥0.0 | 1.0 | Laplace smoothing parameter |
| `--bins <n>` | uint16 | 1-256 | 256 | Feature binning resolution |

#### Constraints
- Labels: Multiclass integer labels
- Features: Discretized into bins automatically

### Predict Action
```
intgrmlc bayes predict --model <sbn> --data <csv> [--out <csv>]
```

#### Output Format
CSV with integer class predictions.

---

## 7. Module: `reduce`

**Purpose**: Principal Component Analysis (dimensionality reduction)
**File Extension**: `.srf`

### Train Action
```
intgrmlc reduce train --data <csv> --out <srf> [options]
```

#### Options
| Option | Type | Range | Default | Description |
|--------|------|-------|---------|-------------|
| `--components <n>` | uint16 | 1-n_features | 10 | Number of principal components |
| `--whiten` | flag | false | Apply whitening transformation |

#### Constraints
- `--components`: Must be ≤ n_features
- No labels required (unsupervised)

### Transform Action
```
intgrmlc reduce transform --model <srf> --data <csv> --out <csv>
```

#### Required Options
| Option | Description |
|--------|-------------|
| `--model <srf>` | Trained PCA model |
| `--data <csv>` | Input data to transform |
| `--out <csv>` | Output transformed data |

#### Output Format
CSV with n_components columns (reduced dimensionality).

---

## 8. Module: `edge`

**Purpose**: Universal model inference with automatic type detection
**Supported Formats**: `.sbf`, `.sff`, `.slf`, `.sbn`, `.ovr`

### Predict Action
```
intgrmlc edge predict --model <path> --data <csv> [--out <csv>]
```

#### Behavior
- Reads model file header to detect type
- Dispatches to appropriate inference engine
- Output format matches source model type

#### Supported Models
- IntgrBoost (.sbf)
- IntgrBoost OvR (.ovr)
- IntgrForest (.sff)
- IntgrLinear (.slf)
- IntgrBayes (.sbn)

---

## 9. CSV Input Format

### Requirements
- **Delimiter**: Comma (`,`)
- **Encoding**: UTF-8 or ASCII
- **Header**: Optional (use `--no-header` if absent)
- **Missing Values**: Not supported (must be imputed beforehand)
- **Label Column**: Specified via `--label` (default: last column)

### Example (with header)
```csv
feature1,feature2,feature3,label
1.5,2.3,-0.8,0
-1.2,0.5,1.1,1
0.3,-0.9,2.4,0
```

### Example (without header, `--no-header`)
```csv
1.5,2.3,-0.8,0
-1.2,0.5,1.1,1
0.3,-0.9,2.4,0
```

### Label Column Specification
| Value | Behavior |
|-------|----------|
| `-1` | Last column (default) |
| `0, 1, 2, ...` | Zero-indexed column number |
| `"name"` | Column name (requires header) |

---

## 10. Model File Formats

### Binary Formats (All Little-Endian)

| Extension | Module | Header Magic | Version | Size |
|-----------|--------|--------------|---------|------|
| `.sbf` | IntgrBoost | `SNAPMLv1` | 1.1.0+ | Variable |
| `.mcu` | IntgrBoost (MCU) | `INTGMCU1` | 1.2.1+ | Variable |
| `.ovr` | IntgrBoost OvR | `SNAPOVRv` | 1.2.0+ | Variable |
| `.sff` | IntgrForest | `SNAPFRv1` | 1.0.0+ | Variable |
| `.slf` | IntgrLinear | `SNAPLNv1` | 1.0.0+ | Variable |
| `.sbn` | IntgrBayes | `SNAPBYv1` | 1.0.0+ | Variable |
| `.srf` | IntgrReduce | `INTGRDv1` | 1.0.0+ | Variable |

### File Structure (Generic)
```
┌─────────────────────────┐
│ Magic Header (8 bytes)  │
├─────────────────────────┤
│ Format Version          │
├─────────────────────────┤
│ Metadata                │
├─────────────────────────┤
│ Model Parameters        │
├─────────────────────────┤
│ Model Data (trees/etc)  │
└─────────────────────────┘
```

### OvR-Specific Structure
```
┌─────────────────────────┐
│ OvRHeader (40 bytes)    │
├─────────────────────────┤
│ OvRMetadata (56 bytes)  │
├─────────────────────────┤
│ Head 0: SBF Model       │
├─────────────────────────┤
│ Head 1: SBF Model       │
├─────────────────────────┤
│ ...                     │
├─────────────────────────┤
│ Head K-1: SBF Model     │
└─────────────────────────┘
```

### MCU-Specific Structure (v1.2.1+)
```
┌─────────────────────────────────┐
│ MCUHeader (40 bytes)            │
│  - magic: "INTGMCU1"            │
│  - model_type: u16              │
│  - version: u16                 │
│  - num_trees: u32               │
│  - num_features: u32            │
│  - num_classes: u32             │
│  - learning_rate_fp: i32 (Q8.8) │
│  - prescaled: u8                │
│  - reserved: u8[3]              │
│  - bias_fp: i32                 │
│  - checksum: u32 (CRC-32)       │
├─────────────────────────────────┤
│ Tree 0 (variable length)        │
│  - Nodes serialized sequentially│
├─────────────────────────────────┤
│ Tree 1                          │
├─────────────────────────────────┤
│ ...                             │
├─────────────────────────────────┤
│ Tree N-1                        │
└─────────────────────────────────┘
```

**MCU Format Design Goals**:
- Minimal parsing overhead for embedded systems
- Sequential read-only access (no random seeks)
- Optimized for 32-bit microcontrollers
- No dynamic memory allocation required
- See `docs/formats/sbf_mcu.md` for detailed specification

---

## 11. Exit Codes

| Code | Name | Description |
|------|------|-------------|
| 0 | `OK` | Success |
| 1 | `GENERIC_ERROR` | Unspecified error |
| 2 | `ARG_ERROR` | Invalid command-line arguments |
| 3 | `IO_ERROR` | File I/O error (read/write) |
| 4 | `MODEL_ERROR` | Model format/loading error |
| 5 | `RUNTIME_ERROR` | Runtime error during execution |

---

## 12. Error Handling

### Argument Validation
- Missing required options → `ARG_ERROR` (2)
- Invalid option values → `ARG_ERROR` (2)
- Unknown flags → Ignored (warning printed)

### File I/O Errors
- File not found → `IO_ERROR` (3)
- Permission denied → `IO_ERROR` (3)
- Disk full (write) → `IO_ERROR` (3)
- Corrupt model file → `MODEL_ERROR` (4)

### Runtime Errors
- Out of memory → `RUNTIME_ERROR` (5)
- Numerical overflow → `RUNTIME_ERROR` (5)
- Invalid data (NaN/Inf) → `RUNTIME_ERROR` (5)
- Feature count exceeds 65,535 → `RUNTIME_ERROR` (5)

---

## 13. Performance Characteristics

### Quantization
- **Format**: Int8 (256 bins)
- **Range**: `[clip_min, clip_max]`
- **Precision**: `(clip_max - clip_min) / 256`
- **Percentile Clipping**: Optional via `--per-feature-clips`

### Memory Usage (Approximate)
| Module | Model Size | Training RAM |
|--------|------------|--------------|
| boost (200 trees, depth 6) | ~50-100 KB | Dataset size × 2 |
| boost-ovr (K=7, 150 trees/head) | ~8 MB | Dataset size × 2K |
| forest (100 trees, depth 8) | ~200-400 KB | Dataset size × 2 |
| linear | ~10-50 KB | Dataset size × 1.5 |
| bayes | ~20-100 KB | Dataset size × 1.2 |
| reduce (50 components) | ~10-50 KB | Dataset size × 3 |

### Training Time (100k samples, typical hardware)
| Module | Configuration | Time |
|--------|---------------|------|
| boost | 200 trees, depth 6 | ~30s |
| boost-ovr | K=7, 150 trees/head, parallel | ~45s |
| forest | 100 trees, depth 8 | ~40s |
| linear | 100 epochs, batch 32 | ~5s |
| bayes | 256 bins | <1s |
| reduce | 50 components | ~2s |

---

## 14. Determinism Guarantees

### Reproducibility
All modules guarantee bit-exact reproducibility when:
1. Same `--seed` value provided
2. Same input data order
3. Same hyperparameters
4. Same IntgrML version

### Non-Deterministic Factors
- **Parallel OvR** (boost-ovr): Thread scheduling order may vary, but results are deterministic due to per-head seeding
- **File I/O**: Order of flushing to disk (no impact on model)

### Verification
Run same command 3 times:
```bash
intgrmlc boost train --data data.csv --out m1.sbf --seed 42
intgrmlc boost train --data data.csv --out m2.sbf --seed 42
intgrmlc boost train --data data.csv --out m3.sbf --seed 42
md5sum m1.sbf m2.sbf m3.sbf  # All identical
```

---

## 15. Advanced Features

### Per-Feature Clipping (JSON Format)
File: `clips.json`
```json
{
  "feature_0": [-5.0, 5.0],
  "feature_1": [-100.0, 100.0],
  "feature_2": [0.0, 1.0]
}
```
Usage:
```bash
intgrmlc boost train --data data.csv --out model.sbf --per-feature-clips clips.json
```

### Accumulator Formats
| Format | Description | Precision | Version |
|--------|-------------|-----------|---------|
| `q16_16` | 16-bit int + 16-bit frac | ±32768.0, ε=1/65536 | v1.1+ (default) |

### Parallel Training (OvR Only)
- **Default**: Enabled
- **Threads**: K (one per class)
- **Disable**: Use `--no-parallel` for debugging/profiling
- **Speedup**: ~6× for K=7 on 8-core CPU

---

## 16. Troubleshooting

### Common Issues

**"File not found" error**
- Check file path is correct
- Use absolute paths or verify working directory
- On Windows: Use forward slashes or escaped backslashes

**"Invalid label column" error**
- Verify label column exists
- Check `--no-header` if CSV has no header
- Use numeric index if column name has spaces

**"Out of memory" error**
- Reduce `--trees` count
- Use smaller `--depth`
- Process data in smaller batches
- For OvR: Use `--no-parallel` to reduce peak RAM

**"Model file corrupted" error**
- Re-train model (file may be incomplete)
- Check disk space during training
- Verify file wasn't truncated during transfer

**Low accuracy results**
- Check `--clip-min` and `--clip-max` match data range
- Reduce `--precision` for finer quantization
- Increase `--trees` or `--depth`
- For OvR: Ensure labels are 0-indexed {0,1,2,...,K-1}

**"intgr_ml v1 supports up to 65,535 features" error**
- IntgrML v1.0 enforces a maximum of 65,535 features
- For high-dimensional datasets:
  - Use `intgrmlc reduce train` to reduce dimensionality before training
  - Apply feature selection in preprocessing
  - Consider using sklearn's `SelectKBest` or `PCA` before exporting to CSV

---

## 17. Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.2.1 | 2025-11-06 | **CRITICAL**: Fixed learning rate bug in inference. Added MCU export, determinism verification, parity testing. **BREAKING**: Models trained with v1.2.0 must be retrained. |
| 1.2.0 | 2025-11-06 | Added `boost-ovr` module, OvR multiclass support |
| 1.1.0 | 2025-10-XX | Added Q16.16 accumulator (default), improved accuracy |
| 1.0.0 | 2025-XX-XX | Initial release with 6 modules |

---

## 18. Quick Reference Card

### Most Common Commands

**Binary Classification**
```bash
intgrmlc boost train --data train.csv --out model.sbf --trees 300 --depth 6
intgrmlc boost predict --model model.sbf --data test.csv --out preds.csv

# New in v1.2.1: Determinism verification
intgrmlc boost determinism --data train.csv --out model.sbf --runs 3 --trees 300

# New in v1.2.1: MCU export and parity testing
intgrmlc boost export --model model.sbf --out model.mcu
intgrmlc boost parity --model model.sbf --data test.csv --samples 100
```

**Multiclass Classification**
```bash
intgrmlc boost-ovr train --data iris.csv --out model.ovr --trees 200
intgrmlc boost-ovr predict --model model.ovr --data test.csv --out preds.csv
```

**Random Forest**
```bash
intgrmlc forest train --data train.csv --out model.sff --trees 100
intgrmlc forest predict --model model.sff --data test.csv --out preds.csv
```

**Dimensionality Reduction**
```bash
intgrmlc reduce train --data train.csv --out pca.srf --components 50
intgrmlc reduce transform --model pca.srf --data test.csv --out reduced.csv
```

**Universal Inference**
```bash
intgrmlc edge predict --model any_model.{sbf,ovr,sff,slf,sbn} --data test.csv
```

---

**End of CLI Reference Specification**
