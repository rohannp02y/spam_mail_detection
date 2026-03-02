# Spam Email Detection

A simple project to detect spam emails using Machine Learning.

## How It Works

1. **train.py** - Trains the model on your data
2. **predict.py** - Uses the trained model to check if emails are spam

## Setup

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Prepare Data
Make sure you have `mail_data.csv` in the same folder with:
- `Message` column (email text)
- `Category` column (spam or ham)

### Step 3: Train the Model
```bash
python train.py
```

This creates 2 files:
- `model.pkl` - The trained model
- `vectorizer.pkl` - The text converter

### Step 4: Make Predictions
```bash
python predict.py
```

## What Each File Does

| File | Purpose |
|------|---------|
| `train.py` | Trains the model on mail_data.csv |
| `predict.py` | Tests the model with sample emails |
| `model.pkl` | Saved trained model |
| `vectorizer.pkl` | Saved text converter |
| `mail_data.csv` | Your dataset |
| `requirements.txt` | Python packages needed |

## To Upload to GitHub

1. Create a GitHub account
2. Create a new repository
3. Copy all these files to a folder
4. Run these commands:
```bash
git init
git add .
git commit -m "Add spam detection project"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git
git push -u origin main
```

Done! Your project is on GitHub! 🎉

## Customization

To test your own email, edit `predict.py`:
```python
test_emails = [
    "Your email text here",
    "Another email to test"
]
```

Then run:
```bash
python predict.py
```
```

---

## **QUICK START STEPS** 

1. **Create a folder** on your computer
```
   my-spam-project/