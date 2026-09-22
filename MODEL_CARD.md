# Model card: <model name>

Template based on Mitchell et al., "Model Cards for Model Reporting" (FAT* 2019). Replace every
`<placeholder>`. Numbers must be copied from real evaluation or bench output, with the command
that produced them. Delete this file in repos that ship no model.

## Model details

- **Name and version:** `<name>`; registry digest `<sha256 from ModelRegistry.register>`
- **Type:** `<e.g. LightGBM binary classifier, Kneser-Ney n-gram LM>`
- **Owner:** Rouviere German
- **Date:** `<YYYY-MM-DD>`
- **License:** MIT (code). Data licenses: see Training data.
- **Artifact format:** `<JSON / npz (allow_pickle=False) / safetensors / LightGBM text>`; never pickle.
- **Run manifest:** `<path to manifest.json>`: seed, dataset SHA-256, package versions, params.

## Intended use

- **Primary use:** `<what decision or output the model supports>`
- **Intended users:** `<who>`
- **Out-of-scope uses:** `<uses the model was not designed or evaluated for>`

## Training data

- **Source:** `<synthetic generator module and seed, or bundled dataset>`
- **Fingerprint:** `<dataset name>` sha256 `<digest>` (`<rows>` rows)
- **Contract:** `<pandera schema class>`; rows failing the contract are `<rejected / quarantined>`
- **Known gaps:** `<populations, ranges or conditions missing from the data>`

## Evaluation

- **Data:** `<held-out split / separate generator seed / shifted scenarios>`
- **Metrics and why:** `<metric>`: `<reason>`
- **Results:** copy from `<command>`, for example:

  | Metric | Value | Command |
  |---|---|---|
  | `<metric>` | `<value>` | `uv run python bench/<script>.py` |

- **Slices:** `<performance by segment, if relevant>`

## Robustness, drift and monitoring

- **Robustness and adversarial tests:** `<tests/...::test_name>` (`<what they perturb>`)
- **Drift monitors:** PSI threshold `<0.2>`, KS alpha `<0.01>` with minimum statistic `<0.05>` on
  `<features>`; see `src/<package>/ml/drift.py`
- **Performance monitors:** `<metric, window, alert threshold>`
- **Retraining trigger:** `<condition>`

## Ethical considerations

`<risks of harm, affected groups, privacy of training data, mitigations>`

## Caveats and recommendations

`<limitations, conditions under which results do not hold, recommended safeguards>`
