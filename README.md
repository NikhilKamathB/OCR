# OCR
This an end-to-end pipeline for the task of optical character recognition. Following this, we will be having a web application delivering the service. The models in this pipeline are trained on [ICDAR](https://rrc.cvc.uab.es/?ch=13&com=introduction) dataset and custom/private invoices.

## Objective
Design an OCR from scratch to extract useful fields from an Invoice.

## Prerequisites
- Docker / Microk8s installed.
- GCP account: enable cloud vision API and Document AI API and setup appropriate service accounts.
- MLFlow: hosted on Docker / Microk8s.

## Folder Structure
```
OCR
|- data
    |- processed (all information after processing any data - information that would be supplied to the model goes here)
        |- information-extraction
    |- raw (this folder holds raw information)
        |- test (holds the test set images and thier ground truth)
        |- train (holds the train and validation split along with their ground truth)
    |- logs (any records such as training logs, graphs, etc goes here)
    |- mlflow
        |- docker (contains scripts to run mlflow in docker)
            |- secrets (contians appropritate service accounts, envirnoment variables, etc)
        |- kubernetes (contains scripts to run mlflow in kubernetes - using kustomize)
            |- base (foundational scripts to deploy mlflow in kubernetes)
                |- local (local system)
                |- stage (remote server - changes in DB connections)
            |- overlays (you may override base files here)
                |- local (local system)
                |- stage (remote server - changes in DB connections)
            |- secrets (contanins secret files needed for the deployment)
                |- local (local system)
                |- secret (contains service acconts and other raw secrets)
                |- stage (remote server - changes in DB connections)
    |- runs (pretrained models, saved/logged models goes here)
        |- models (holds logged models)
        |- pretrained-models (holds downloaded models)
    |- src
        |- information-extraction (OCR pipeline - information extraction from the detected and recognized texts)
        |- pre (this folder containing scripts ought to be run before stepping into the OCR pipeline)
        |- text-detection (OCR pipeline - text detection in scene)
    |- requirements.txt (pip requirements that must be installed prior to running this pipeline)
```

## Tasks
- [X]  Data acquisition.
    - Download relevant datasets.
    - `cd ./src/pre`, run `ocr-annotate.py` (for OCR) and `document-ai-annotate.py` (for Entity recognition) to generate true labels from GCP - Document AI service. This step is optional and we use it to run on our custom dataset.
    - All out data is stored in GCS. Run the following command to download data (not accessible to the public) - `gsutil -m cp -r "gs://data-hangar/ocr/data" .`
- [ ]  Data Annotation/Preparation.
    - [ ]  Text Detection.
    - [ ]  Text Recognition.
    - [ ]  Information Extraction.
- [ ]  Text Detection AI Pipeline.
- [ ]  Text Recognition AI Pipeline.
- [ ]  Information Extraction AI Pipeline.
- [ ]  Integration of Blocks.
- [ ]  Web Application.

## Future Work
* Use better models.
* Improve dataset.
* Extend this application to other related domains.

## GitHub Link
[https://github.com/NikhilKamathB/OCR](https://github.com/NikhilKamathB/OCR)