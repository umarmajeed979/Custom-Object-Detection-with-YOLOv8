# Project Name

One-line description (copy from portfolio doc).

## Stack
Framework | Skills demonstrated

## Setup
```bash
pip install -r requirements.txt
pip install -e .
cp .env.example .env
python run.py
```

## What changes per project
Edit these — this is where a project's actual identity lives:
- `app/config.py` — PROJECT_NAME, DESCRIPTION, MODEL_PATH, thresholds
- `app/core/model.py` — how the model loads (torch/tf/cv2-specific)
- `app/core/predictor.py` — pre/post-processing around inference
- `app/data/preprocessor.py` and `app/data/loader.py` — dataset-specific splitting/loading
- `app/utils/image_utils.py` — TARGET_SIZE and normalization for that model's input
- `frontend/app.py` — the upload/results UI for that project's inputs and outputs
- `requirements.txt` — add torch / tensorflow / opencv-python / mediapipe etc. as needed
- `.env`, `README.md` — project-specific values and docs

Sometimes edit:
- `app/api/schemas.py` — only if the prediction shape genuinely differs (e.g. segmentation mask vs. classification label)
- `app/utils/validators.py` — only if a project accepts video/audio instead of images
- add a project-specific lookup file to `app/data/` (see `disease_info.py` as an example) only if the project needs one

Leave alone — identical across every project, don't touch:
- `app/__init__.py`, `app/utils/logger.py`, `app/api/routes.py`,
  `app/api/middleware.py`, `app/services/storage.py`, `run.py`,
  `wsgi.py`, `Dockerfile`, `pyproject.toml`, `tests/conftest.py`

If you ever find yourself editing something in the "leave alone" list,
that's a sign to fix it in the template repo itself so every future
project inherits the fix — not to patch it in one project only.

## Dataset
Source: <link>
Download to `data/raw/` then run `python scripts/prepare_data.py`

## Training
Trained on Google Colab (GPU) — see `scripts/train_model.py`.

## API
`POST /api/v1/predict` — multipart image upload -> prediction JSON

## Frontend
`streamlit run frontend/app.py`
