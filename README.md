# Phishing Detection Weekly Workflow

The six Python scripts follow the weekly tasks and the Markdown handoff. Week 1 uses a local copy of the BERT tokenizer so it can tokenize offline. [BERT tokenizer source and revision](https://huggingface.co/google-bert/bert-base-uncased/tree/86b5e0934494bd15c9632b12f734a8a67f723594).

## Set up the workflow

From the repository root, run the following in PowerShell with Python 3.12.

```powershell
cd provided_code_workflow
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Download `Phishing_Email.csv` into this folder. The direct CSV mirror matches the file used for the recorded run. [Original dataset on Kaggle](https://www.kaggle.com/datasets/subhajournal/phishingemails) · [Direct CSV download](https://huggingface.co/datasets/zefang-liu/phishing-email-dataset/resolve/main/Phishing_Email.csv).

Run this preparation step once to create `phishing.csv` with the columns and label names expected by Week 1.

```powershell
@'
import pandas as pd

df = pd.read_csv("Phishing_Email.csv", usecols=["Email Text", "Email Type"])
df = df.rename(columns={"Email Type": "Label"})
df["Label"] = df["Label"].replace({"Phishing Email": "phishing", "Safe Email": "legitimate"})
df.to_csv("phishing.csv", index=False)
'@ | .\.venv\Scripts\python.exe -
```

## Run the workflow

```powershell
$env:PYTHONUTF8 = '1'
.\.venv\Scripts\python.exe week1.py
.\.venv\Scripts\python.exe week2.py
.\.venv\Scripts\python.exe week3.py
.\.venv\Scripts\python.exe week4.py
.\.venv\Scripts\python.exe phishing_detector.py
.\.venv\Scripts\python.exe week6.py
```

In the detector, enter one email per prompt. Type `exit` to save `final_phishing_model.pkl` before starting Week 6.

## Run results

- Week 1 removed 16 rows with missing email text, printed BERT tokens and token IDs for the first email, and saved 18,634 cleaned rows.
- Week 2 split those rows into 14,907 training and 3,727 test emails. It saved both sets as 5,000-feature TF-IDF matrices with their labels and vectorizer in `processed_data.pkl`.
- Week 3 trained and saved the logistic regression baseline. The script reported accuracy 0.96, precision 0.94, recall 0.97, and F1 0.95.
- Week 4 ran 75 grid-search fits. Tuned logistic regression (`C=100`, `max_iter=500`) scored 97.02% on the cleaned test set, linear SVM scored 96.83%, and random forest scored 95.89%. SMOTE balanced the training matrices to 9,113 emails per class. The supplied manual selection saved the SVM, which was fitted before SMOTE.
- Week 5 classified the built-in example and two interactive inputs. Typing `exit` saved the final classifier. Its predictions matched the Week 4 artifact on the test split.
- Week 6 classified all five supplied examples correctly and ran the misclassification check.

## Week 6 error analysis and future work

The five supplied examples produced zero errors. To examine mistakes, the run also evaluated the final SVM on the 3,727 test emails using the same raw-text input as the CLI. Accuracy was 95.28%, with 147 legitimate emails flagged as phishing and 29 phishing emails missed. Correct predictions had a median of 69 features that matched the fitted TF-IDF vocabulary. The median was 37 for false positives and 33 for false negatives. That pattern suggests collecting more examples with limited vocabulary coverage for future evaluation.

Applying the training cleaner to the same test emails yielded 96.83% accuracy and changed 62 predictions. For 393 test rows, a training row had exactly the same cleaned text. Two of those test rows had empty cleaned text. These figures describe this run on the project dataset.

For future work, evaluate consistent cleaning on an independent, deduplicated split. Add representative phishing and legitimate emails with varied language, evaluate sender, link and attachment metadata, and compare a transformer or LSTM on the same test data.
