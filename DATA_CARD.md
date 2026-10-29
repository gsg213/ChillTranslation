# Data card — registre (Spanish sources for ES → FR register translation)
 
Hand-written sections describe *how* and *why* the data was built. The section between the
GENERATED markers is rewritten by `./run card` from `data/processed/stats.json` and
`manifest.json`, so the numbers always match the files. Don't edit it by hand.
 
## Purpose
 
Spanish sentences used as inputs for a model that translates into French in four registers
(soutenu, courant, amical, familier) and explains each translation with tags from
`spec/taxonomy.yaml`.
 
- **train / val**: inputs for teacher generation (day 2) and fine-tuning (day 3).
- **test_tatoeba**: held-out sentences of the same kind as train.
- **test_cocoa**: chat-style segments with *human* French formal/informal references; also the
  source-formality leakage test (each item appears with a formal and an informal Spanish source).
- **test_own**: sentences I actually wanted to say; the only test set in my own Spanish variety.
Out of scope: anything beyond Spanish → French; judging French quality from these files alone
(only CoCoA has human French references).
 
## Sources and licenses
 
| Source | What is used | License | Notes |
|---|---|---|---|
| Tatoeba (tatoeba.org) | Spanish sentences; French translations when linked (`fr_ref`) | CC BY 2.0 FR | Attribution: Tatoeba sentence ids are kept in `id` (`tatoeba-<id>`). Export downloaded on: TODO(you) |
| CoCoA-MT (Amazon, github.com/amazon-science/contrastive-controlled-mt) | Spanish and French formal/informal references, joined through their shared English sources | CDLA-Sharing-1.0 | Check the license terms before publishing CoCoA-derived rows. |
| Own sentences | `data/raw/own.jsonl` | Mine | Test-only, never used for training or model selection. |
 
## How it was built (`./run download`, then `./run data`)
 
1. **Tatoeba** (`registre/data/tatoeba.py`)
   - Spanish sentences that have a linked French translation (all Spanish sentences if the
     links file is absent).
   - Kept 3–25 words; deduplicated on a normalized form (lowercase, no punctuation,
     **accents kept** so *tú* ≠ *tu*).
   - Sentences mentioning Tom or Mary capped at 10% (Tatoeba is full of them).
   - Deterministic subsample of 20,000 candidates by hash before feature extraction.
2. **CoCoA-MT** (`registre/data/cocoa.py`)
   - en-es and en-fr files joined on the English text: 306 train and 489 test items.
   - Each item gives **two rows** (Spanish formal source, Spanish informal source) sharing a
     `group_id` and carrying both French references.
3. **Features** (`registre/data/features.py`, spaCy `es_core_news_sm` 3.8.0)
   - `addresses_someone`: 2nd-person morphology on verbs/auxiliaries/pronouns, or an explicit
     usted/ustedes. Shared across the rows of a group (see limitations).
   - `is_question` (starts with ¿ or ends with ?), `n_words`.
4. **Splits** (`registre/data/splits.py`) — grouped by `group_id`, deterministic (SHA-1 buckets):
   - own → test; CoCoA test → test; CoCoA train → val for ~15% of groups, else train;
   - Tatoeba → up to 300 test and 250 val groups, the rest train.
5. **Train sample**: Tatoeba train reduced to 5,500 rows, at least 50% addressing someone.
6. **Leakage check**: MinHash (5-character shingles, 128 permutations) with LSH at Jaccard 0.8;
   train/val rows nearly duplicating any test row are removed. Test rows are never removed.
## Columns
 
| Column | Meaning |
|---|---|
| `id` | Unique row id (`tatoeba-<id>`, `cocoa-<split>-<domain>-<k>-<formality>`, `own-…`) |
| `source`, `source_lang` | Spanish input, `"es"` |
| `origin` | `tatoeba` · `cocoa` · `own` |
| `group_id` | Rows sharing it are versions of the same item and always share a split |
| `split` | `train` · `val` · `test` |
| `fr_ref` | Tatoeba's French translation (register not controlled), if linked |
| `cocoa_domain`, `cocoa_formality` | CoCoA domain; whether `source` is the formal or informal Spanish |
| `fr_formal`, `fr_informal`, `en` | CoCoA French references and English source |
| `context` | Own sentences: who I was talking to (optional) |
| `addresses_someone`, `is_question`, `n_words` | Features above |
 
## Known limitations and noise
 
- **usted without the pronoun is missed.** *Usted* takes 3rd-person verbs, so "¿Puede venir?"
  looks like él/ella. On CoCoA's test set the raw detector fired on 16% of formal Spanish
  sources vs 60% of the informal versions of the *same* sentences. Sharing the flag across each
  group fixes CoCoA; Tatoeba rows have no pair, so `addresses_someone` under-counts usted there.
- **CoCoA's Spanish is Peninsular** (vosotros in informal plurals); my own variety is only in
  `test_own`.
- **Distribution shift**: Tatoeba sentences are short and clean; CoCoA segments are multi-sentence
  chat turns with typos.
- **Tatoeba's French references have arbitrary register** (contributors chose tu or vous freely):
  usable as rough references for meaning, not for register.
- **CoCoA's French informal references are minimal edits** of the formal ones (tu swapped in,
  *ne* kept): good for tu/vous accuracy, a weak reference for the *amical* register.
- Some CoCoA train items have identical formal and informal references.
<!-- BEGIN GENERATED: ./run card -->
### Rows and features by split and origin

| split | origin | rows | addresses someone | questions | median words |
|---|---|---:|---:|---:|---:|
| test | cocoa | 978 | 60% | 33% | 20 |
| test | tatoeba | 300 | 11% | 13% | 6 |
| train | cocoa | 510 | 57% | 36% | 19 |
| train | tatoeba | 5,498 | 36% | 18% | 6 |
| val | cocoa | 102 | 47% | 29% | 19 |
| val | tatoeba | 250 | 13% | 12% | 6 |

### Files (`data/processed/`)

| file | rows | sha256 |
|---|---:|---|
| train.parquet | 6,008 | `356a8fe69cfd7d25…` |
| val.parquet | 352 | `0cafd19e70915825…` |
| test_tatoeba.parquet | 300 | `9aa7f7329070dfba…` |
| test_cocoa.parquet | 978 | `0b5a5e46a61ffefc…` |
| test_own.parquet | 0 | `ef2c00f9ae3b5dd4…` |

Generated on 2026-10-29 from commit `4919932`.
<!-- END GENERATED -->
