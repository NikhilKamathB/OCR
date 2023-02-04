from easydict import EasyDict 


config = EasyDict()

# Models
config.CRAFT = '../../runs/downloaded_models/craft_mlt_25k.pth'

# Data
config.CHROME_DRIVER_PATH = '../../drivers/chromedriver_linux64/chromedriver'
config.DATA_DIR_RAW = '../../data/raw'
config.DATA_DIR_INTERIM = '../../data/interim'
config.DATA_DIR_PROCESSED = '../../data/processed'
config.DATA_DIR_OUTPUT = '../../data/output'
config.WEBSCRAPE_OUTPUT = '../../data/raw/webscrape'
config.RAW_2021_06_14_ocr_kyc_pdfs = '../../data/raw/2021-06-14_ocr_kyc-pdfs'
config.RAW_mjsynth = '../../data/raw/mjsynth_sample/mjsynth_sample'
config.INTERIM_2021_06_14_ocr_kyc_pdfs_manual = '../../data/interim/2021-06-14_ocr_kyc-pdfs_manual'
config.INTERIM_2021_06_14_ocr_kyc_pdfs_annotations = '../../data/interim/2021-06-14_ocr_kyc-pdfs_annotations'
config.INTERIM_2022_01_06_combined_orc_kyc = '../../data/interim/2022-01-06_combined_orc_kyc'
config.INTERIM_2021_06_14_ocr_kyc_pdfs_annotations = '../../data/interim/2021-06-14_ocr_kyc-pdfs_annotations'
config.INTERIM_2021_06_14_ocr_kyc_pdfs_annotations_csv = '../../data/interim/2021-06-14_ocr_kyc-pdfs_annotations/text_detection_annots.csv'
config.INTERIM_PDF_TO_IMAGES_2021_06_14_ocr_kyc_pdfs_images = '../../data/interim/2021-06-14_ocr_kyc-pdfs-images'
config.INTERIM_2021_08_10_text_recognition = '../../data/interim/2021-08-10_text_recognition'
config.INTERIM_2021_08_17_text_recognition_mjsynth_images = '../../data/interim/2021-08-17_text_recognition_mjsynth_images'
config.PROCESSED_2021_06_14_ocr_kyc_pdfs = '../../data/processed/2021-06-14_ocr_kyc-pdfs'
config.INTERIM_2021_08_10_text_recognition_ner_csv = '../../data/interim/2021-08-10_text_recognition/text_recognition_annots.csv'