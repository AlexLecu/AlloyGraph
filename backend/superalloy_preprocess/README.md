# AlloyGraph Superalloy Data Processing Pipeline

This README describes the workflow for extracting, transforming, and preparing superalloy data for machine learning fine-tuning using the AlloyGraph project scripts.

## 1. Extract Alloy Data from Excel

**Script:** `backend/superalloy_preprocess/extract_excel_sheets.py`

- This script reads all sheets from the annotated Excel file (`CAST_Wrought_Alloys.xlsx`) and extracts alloy composition and mechanical property data.
- The output is a JSONL file: `backend/superalloy_preprocess/output_data/all_alloys.jsonl`.

**How to run:**
```bash
python backend/superalloy_preprocess/extract_excel_sheets.py
```

## 2. Convert Extracted Data for GPT Fine-Tuning

**Script:** `backend/superalloy_preprocess/convert_to_finetune.py`

- This script takes the JSONL file generated in step 1 and converts it into the OpenAI fine-tuning format.
- It automatically splits the data into train, validation, and test sets.
- Output files are created in `backend/superalloy_preprocess/output_data/finetuned_data/`:
  - `finetuned.jsonl` (full dataset)
  - `train.jsonl`
  - `val.jsonl`
  - `test.jsonl`

**How to run:**
```bash
python backend/superalloy_preprocess/convert_to_finetune.py
```

## 3. Run the Notebook for Model Training or Analysis

- Open the relevant Jupyter notebook (e.g., `backend/superalloy_preprocess/fine_tuning_ner_gpt.ipynb`).
- Use the output files from step 2 as input for training, evaluation, or further analysis.

**How to run:**
```bash
jupyter notebook backend/superalloy_preprocess/fine_tuning_ner_gpt.ipynb
```

---

### Summary of Workflow
1. **Extract Excel Data** → `all_alloys.jsonl`
2. **Convert to Fine-Tune Format & Split** → `finetuned_data/*.jsonl`
3. **Train/Evaluate in Notebook**

This pipeline ensures your superalloy data is clean, structured, and ready for machine learning tasks.

