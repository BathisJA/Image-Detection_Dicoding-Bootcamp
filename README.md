# Rock Paper Scissors Classifier

A small image classifier that looks at a photo of a hand and calls the throw: rock, paper, or scissors. It uses **transfer learning** — MobileNetV2 pretrained on ImageNet, fine-tuned on the hand dataset — so it generalizes to hands photographed outside the original dataset, not just the green-background studio shots the dataset was recorded in. It's my documentation while taking the [Dicoding](https://www.dicoding.com/) bootcamp.

## What's in here

- **`model.ipynb`** — the training notebook: data prep, the MobileNetV2 model, two-phase training (frozen head, then fine-tuning), training curves, and a sanity check that mimics the app's preprocessing.
- **`app.py`** — the Streamlit app. Upload an image, press Predict, get a verdict with per-class confidence.
- **`model.h5`** — the trained weights, loaded by the app.
- **`Dataset/rockpaperscissors/train`** — all 2,188 images (712 paper, 726 rock, 750 scissors), sorted into class folders.
- **`static/theme.css`** — the app's design, kept separate from the app logic.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Python 3.13 is required (TensorFlow doesn't ship wheels for 3.14 yet). On [Streamlit Community Cloud](https://share.streamlit.io), pick **Python 3.13** in the app's Advanced settings when deploying.

Upload a JPG or PNG of a hand sign and press **Predict**. The model was trained on photos against a green background — studio-style shots work much better than, say, a hand on your desk.

## Retrain it

Open `model.ipynb` and run it top to bottom. It splits the dataset 60/40 with a fixed seed, fine-tunes MobileNetV2 in two phases (frozen head, then the top third of the base), and overwrites `model.h5` for the app.

## A note on the dataset and the model

Earlier versions of this repo kept identical copies of every image in `train/` *and* `val/`, and the notebook split *both* folders again on top of that — training and validation silently overlapped, so the accuracy numbers were too good to be true. The duplicate `val/` folder is gone; the notebook now owns the split and prints an overlap check so you can see it's zero.

The first model was a small CNN trained from scratch. It hit ~99% on held-out dataset photos but misclassified hands photographed in the real world — with only ~1,300 training images it had memorized the dataset's green background instead of learning hands. The MobileNetV2 transfer-learning model in this notebook fixed that.
