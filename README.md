# Rock Paper Scissors Classifier

A small image classifier that looks at a photo of a hand and calls the throw: rock, paper, or scissors. The network is built and trained from scratch (no transfer learning) — it's my documentation while taking the [Dicoding](https://www.dicoding.com/) bootcamp.

## What's in here

- **`model.ipynb`** — the training notebook: data prep, augmentation, the CNN itself, training curves, and a sanity check that mimics the app's preprocessing.
- **`app.py`** — the Streamlit app. Upload an image, press Predict, get a verdict with per-class confidence.
- **`model.h5`** — the trained weights, loaded by the app.
- **`Dataset/rockpaperscissors/train`** — all 2,188 images (712 paper, 726 rock, 750 scissors), sorted into class folders.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Upload a JPG or PNG of a hand sign and press **Predict**. The model was trained on photos against a green background — studio-style shots work much better than, say, a hand on your desk.

## Retrain it

Open `model.ipynb` and run it top to bottom. It splits the dataset 60/40 with a fixed seed, trains the CNN (early stopping keeps the best weights), and overwrites `model.h5` for the app.

## A note on the dataset

Earlier versions of this repo kept identical copies of every image in `train/` *and* `val/`, and the notebook split *both* folders again on top of that — training and validation silently overlapped, so the accuracy numbers were too good to be true. The duplicate `val/` folder is gone; the notebook now owns the split and prints an overlap check so you can see it's zero.
