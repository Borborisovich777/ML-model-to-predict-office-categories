# Intro to AI Office Project

This guide explains how to:

1. Put this project into GitHub
2. Install the required dependencies
3. Launch the notebook
4. Re-run the model pipeline
5. Check that the project worked correctly

## 1. Current Project Layout

The main project files are inside this folder:

```text
intro-to-ai-main/
```

Important files:

- `nagibator.ipynb`: main notebook with preprocessing, training, evaluation, and submission generation
- `office_train.csv`: training dataset
- `office_test.csv`: test dataset
- `clean.csv`: generated cleaned training data
- `submission.csv`: generated prediction file
- `results.md`: saved performance metrics

Recommended approach:

- Create the GitHub repository from the `intro-to-ai-main` folder
- Do not use the parent folder unless you intentionally want to include duplicate CSV/PDF/ZIP files that sit outside the project

## 2. Create a GitHub Repository for This Project

Open Terminal and move into the project folder:

```bash
cd "/Users/nurtore.arynuruly/Projects/ML Office Project/intro-to-ai-main"
```

Initialize Git:

```bash
git init
git add .
git commit -m "Initial commit"
```

Create a new empty repository on GitHub:

- Go to <https://github.com/new>
- Repository name example: `intro-to-ai-office-project`
- Choose `Public` or `Private`
- Do not add a README, `.gitignore`, or license from GitHub if you already committed locally

Connect the local project to GitHub:

```bash
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/intro-to-ai-office-project.git
git push -u origin main
```

If GitHub asks for authentication:

- Use GitHub Desktop, GitHub CLI, or a Personal Access Token
- If `git push` fails because password authentication is disabled, use a token instead of your GitHub password

## 3. Recommended Python Environment

This project is based on a Jupyter notebook and uses these libraries:

- `pandas`
- `numpy`
- `scikit-learn`
- `xgboost`
- `lightgbm`
- `catboost`
- `jupyter`

The notebook metadata shows it was previously run with Python `3.13.5`.

Recommended:

- Use Python `3.11`, `3.12`, or `3.13`
- Avoid relying on a system Python with no virtual environment

Create and activate a virtual environment:

```bash
cd "/Users/nurtore.arynuruly/Projects/ML Office Project/intro-to-ai-main"
python3 -m venv .venv
source .venv/bin/activate
```

Upgrade `pip`:

```bash
python -m pip install --upgrade pip
```

Install dependencies:

```bash
pip install pandas numpy scikit-learn xgboost lightgbm catboost jupyter
```

Optional check:

```bash
python -c "import pandas, numpy, sklearn, xgboost, lightgbm, catboost; print('All dependencies installed')"
```

## 4. How to Launch the Project

This project does not have a web server or CLI app. The main way to run it is the notebook.

Start Jupyter Notebook:

```bash
cd "/Users/nurtore.arynuruly/Projects/ML Office Project/intro-to-ai-main"
source .venv/bin/activate
jupyter notebook
```

Then open:

```text
nagibator.ipynb
```

Run the notebook from top to bottom.

You can also use JupyterLab:

```bash
jupyter lab
```

## 5. How to Run It Without Opening the Notebook UI

If you want to execute the notebook from Terminal:

```bash
cd "/Users/nurtore.arynuruly/Projects/ML Office Project/intro-to-ai-main"
source .venv/bin/activate
jupyter nbconvert --to notebook --execute nagibator.ipynb --output executed-nagibator.ipynb
```

This will:

- run all notebook cells
- create `executed-nagibator.ipynb`
- fail with an error if one of the cells breaks

## 6. What Counts as "Testing" in This Project

There is currently no automated test suite such as:

- `pytest`
- unit tests
- integration tests

For this repository, practical testing means re-running the notebook and verifying that the expected output files are produced and the metrics look correct.

## 7. How to Check That the Project Works

After running the notebook, verify these files exist or were refreshed:

- `clean.csv`
- `submission.csv`
- `results.md`
- `catboost_info/`

Check the saved metrics:

```bash
sed -n '1,80p' results.md
```

Expected values from the current project version:

- Train Accuracy: about `90.42%`
- Validation Accuracy: about `86.37%`

Check that the submission file was created:

```bash
ls -lh submission.csv
```

Preview the prediction file:

```bash
sed -n '1,10p' submission.csv
```

## 8. Typical Run Flow

Use this sequence when someone clones the repository later:

```bash
git clone https://github.com/YOUR_USERNAME/intro-to-ai-office-project.git
cd intro-to-ai-office-project
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install pandas numpy scikit-learn xgboost lightgbm catboost jupyter
jupyter notebook
```

Then open `nagibator.ipynb` and run all cells.

## 9. Common Problems

### `jupyter: command not found`

Install Jupyter inside the virtual environment:

```bash
pip install jupyter
```

### `ModuleNotFoundError`

One or more required packages are missing. Reinstall dependencies:

```bash
pip install pandas numpy scikit-learn xgboost lightgbm catboost jupyter
```

### LightGBM or CatBoost installation problems

If installation fails on your Python version:

- create a fresh virtual environment
- use Python `3.11`, `3.12`, or `3.13`
- upgrade `pip` before installing packages

### Notebook runs but output files do not change

Possible reasons:

- cells were not executed in order
- the notebook stopped on an earlier error
- you ran it from the wrong working directory

Always run the notebook from inside the project folder:

```bash
cd "/Users/nurtore.arynuruly/Projects/ML Office Project/intro-to-ai-main"
```

## 10. Recommended Next Cleanup

Before sharing the repository widely, consider adding:

- `.gitignore`
- `requirements.txt`
- clearer notebook name, for example `office_classification.ipynb`

Suggested `.gitignore` entries:

```gitignore
.venv/
__pycache__/
.DS_Store
.ipynb_checkpoints/
```

## 11. Verified on This Machine

Checked in this workspace:

- the project is not yet a Git repository
- the real project folder is `intro-to-ai-main`
- the notebook depends on `pandas`, `numpy`, `scikit-learn`, `xgboost`, `lightgbm`, `catboost`, and `jupyter`
- `jupyter` is not currently installed in the active Python environment
- most ML dependencies are also not currently installed in the active Python environment

That means the documentation above includes the required setup steps, but full notebook execution still needs the environment to be installed first.
