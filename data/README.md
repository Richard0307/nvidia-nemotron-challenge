Place training data files in this directory.

Expected files:

- `train.csv`: Kaggle competition training file with `id`, `prompt`, and `answer`
- `train_cot.jsonl`: Stage 2 CoT-augmented data with `prompt`, `answer`, and `reasoning`

The first local dry run can use a tiny subset of `train.csv`.
