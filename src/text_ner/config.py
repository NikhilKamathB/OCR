from easydict import EasyDict 


config = EasyDict()

# Data
config.DATA_DIR_RAW = '../../data/raw'
config.DATA_DIR_INTERIM = '../../data/interim'
config.DATA_DIR_PROCESSED = '../../data/processed'
config.DATA_DIR_OUTPUT = '../../data/output'
config.INTERIM_2021_06_14_ocr_kyc_pdfs_annotations_csv = '../../data/interim/2021-06-14_ocr_kyc-pdfs_annotations/text_detection_annots.csv'
config.INTERIM_2021_06_14_ocr_kyc_pdfs_manual_dl = '../../data/interim/2021-06-14_ocr_kyc-pdfs_manual-dl'

# Key fields.
config.SIM_KEY_FIELDS = ['nama', 'alamat', 'tempat', 'tgl', 'lahir', 'tgl.lahir', 'tinggi', 'pekerjaan', 'sim', 'no. sim', 'berlaku', 's/d', 'berlaku s/d']
config.SIM_KEY_FIELDS_LINES = {
    'nama' : {'lines': 1}, 
    'alamat' : {'lines': 5}, 
    'tempat' : {'lines': 1},
    'tgl' : {'lines': 1}, 
    'lahir' : {'lines': 1}, 
    'tgl lahir' : {'lines': 1}, 
    'tinggi' : {'lines': 1}, 
    'pekerjaan' : {'lines': 1}, 
    'sim' : {'lines': 1}, 
    'no sim' : {'lines': 1}, 
    'berlaku' : {'lines': 1}, 
    's/d' : {'lines': 1}, 
    'berlaku s/d' : {'lines': 1}
    }
config.KTP_KEY_FIELDS = ['nik', 'nama', 'tempat', 'tgl', 'lahir', 'tempat/tgl lahir', 'jenis', 'kelamin', 'jenis kelamin',  'alamat', 'rt', 'rw', 'rt/rw', 'kel', 'desa', 'kel/desa', 'kecamatan', 'agama', 'status', 'perkawinan', 'status perkawinan', 'pekerjaan', 'kewarganegaraan', 'berlaku', 'hingga', 'berlaku hingga']
config.STNK_SAMSAT_KEY_FIELDS = ['nopol', 
                                 'nama',
                                 'alamat',
                                 'rangka', 'mesin', 'no. rangka / mesin',
                                 'merek', 'type', 'merek / type',
                                 'warna', 'kendaraan', 'warna kendaraan',
                                 'bhn', 'bakar', 'cylinder', 'bhn bakar / cylinder',
                                 'nilai', 'jual', 'nilai jual',
                                 'pokok', 'pkb pokok*)',
                                 'denda', 'pkb denda*)',
                                 'cetak', 'stnk', 'pnbp cetak stnk',
                                 'nrkb', 'pilihan', 'pnbp nrkb pilihan',
                                 'status',
                                 'kendaraan', 'ke', 'kendaraan ke',
                                 'nik',
                                 'no bpkb',
                                 'model', 'pembuatan', 'model / pembuatan',
                                 'tnkb', 'warna tnkb',
                                 'masa', 'berlaku', 'stnk', 'masa berlaku stnk',
                                 'jatuh', 'tempo', 'jatuh tempo pajak',
                                 'swdkllj*)',
                                 'swdkllj denda*)',
                                 'pnbp plat(tnkb)',
                                 'total'
                                 ]
config.STNK_SAMSAT_KEY_FIELDS_DOC_AI = ['nopol', 
                                 'nama',
                                 'alamat',
                                 'no. rangka / mesin',
                                 'merek / type',
                                 'warna kendaraan',
                                 'bhn bakar / cylinder',
                                 'nilai jual',
                                 'pkb pokok*)',
                                 'pkb denda*)',
                                 'pnbp cetak stnk',
                                 'pnbp nrkb pilihan',
                                 'status',
                                 'kendaraan ke',
                                 'nik',
                                 'no bpkb',
                                 'model / pembuatan',
                                 'warna tnkb',
                                 'masa berlaku stnk',
                                 'jatuh tempo pajak',
                                 'swdkllj*)',
                                 'swdkllj denda*)',
                                 'pnbp plat(tnkb)',
                                 'total'
                                 ]                            
config.STNK_KEY_FIELDS = ['no.', 
                        'nomor', 'registrasi', 'nomor registrasi',
                        'nama', 'pemilik', 'nama pemilik',
                        'alamat',
                        'merk',
                        'type',
                        'jenis',
                        'model',
                        'tahun', 'pembuatan', 'tahun pembuatan',
                        'isi', 'silinder', 'isi silinder', 
                        'nomor', 'rangka', 'nik', 'vin', 'rangka/nik/vin', 'nomor rangka/nik/vin', 
                        'nomor', 'mesin', 'nomor mesin',
                        'warna', 
                        'bahan', 'bakar', 'bahan bakar',
                        'warna', 'tnkb', 'warna tnkb',
                        'tahun', 'registrasi', 'tahun registrasi',
                        'nomor', 'bpkb', 'nomor bpkb',
                        'kode', 'lokasi', 'kode lokasi',
                        'urut', 'no urut pendaftaran',
                        'berlaku', 'sampai', 'berlaku_sampai']

# DOC thresholds
config.KEY_FILED_REGION_THRESHOLD_X = 0.60
config.VALUE_FILED_REGION_THRESHOLD_PIXEL_X = 25
config.VALUE_FILED_EXTREME_PIXEL_X = 200
config.ANGLE_THRESHOLD = 3
config.ANGLE_THRESHOLD_TOLERANCE_MIN = -45
config.ANGLE_THRESHOLD_TOLERANCE_MAX = 3
config.DISTANCE_THRESHOLD = 200
config.FORM_PIXEL_TOLERANCE = 7

config.NONE_TOKEN = '<NONE>'
config.HORIZONTAL_DIST_BETWEEN_WORDS = 25
config.VERTICAL_DIST_BETWEEN_WORDS = 3
config.VERTICAL_DIST_BETWEEN_WORD_AND_KEY_LINES_THRESHOLD = 5
config.KTP_FORM_FIELD_THRESHOLD = 0.45
config.SIM_FORM_FIELD_THRESHOLD = 0.75
config.STNK_FORM_FIELD_THRESHOLD = 0.7
config.STNK_SAMSAT_FORM_FIELD_THRESHOLD = 0.8

# GCS
config.GCS_BUCKET = {
    'ktp': "gs://som-datasets/interim/2022-01-06_combined_orc_kyc/KTP/",
    'sim': "gs://som-datasets/interim/2022-01-06_combined_orc_kyc/SIM-OLD/",
    'stnk': "gs://som-datasets/interim/2022-01-06_combined_orc_kyc/STNK/"
}

# SPECIAL TOKENS
config.SPECIAL_TOKEN = ['tempat_tgl_lahir', 'berlaku', 'hingga', 'berlaku_sim', 'tempat', 'lahir', 'status']

# DUPLICATE TOKENS
config.DUPLICATE_TOKEN = [('nomor', 2)]

# KEY_THREHOLDS
config.STNK_KEY_THRESHOLD = {
        'no': 50,
        'nomor_registrasi': 0,
        'nama_pemilik': 0,
        'alamat': 0,
        'type': 0,
        'jenis': 0,
        'model': 0,
        'tahun_pembuatan': 0,
        'isi_silinder': 0,
        'nomor_ragka_nik_vin': 0,
        'nomor_mesin2': 0,
        'warna2': 0,
        'bahan_bakar2': 0,
        'warna_tnkb2': 0,
        'tahun_registrasi': 0,
        'nomor_bpkb': 0,
        'kode_lokasi': 0,
        'no_urut_pendaftaran': 0,
        'berlaku_sampai':0
        }

# MIME_TYPE
config.MIME_TYPES = {
    'jpeg': 'jpeg',
    'jpg': 'jpeg',
    'png': 'png',
    'webp': 'webp'
}

# DF
config.DF_COLS = {
    'ktp': [
        'agama',
        'alamat',
        'berlaku_hingga',
        'jenis_kelamin',
        'kecamatan',
        'kel_desa',
        'kewarganegaraan',
        'nama',
        'nik',
        'pekerjaan',
        'rt_rw',
        'status_perkawinan',
        'tempat',
        'tgl_lahir'
        ],
    'sim': [
        'nama',
        'tgl_lahir',
        'jenis_kelamin',
        'tempat',
        'alamat',
        'kel_desa',
        'kecamatan',
        'agama',
        'pekerjaan',
        'kewarganegaraan',
        'berlaku_hingga',
        'no_sim',
        'tinggi'
        ],
    'stnk': [
        'no',
        'nomor_registrasi',
        'nama_pemilik',
        'alamat',
        'type',
        'jenis',
        'model',
        'tahun_pembuatan',
        'isi_silinder',
        'nomor_ragka_nik_vin',
        'nomor_mesin2',
        'warna2',
        'bahan_bakar2',
        'warna_tnkb2',
        'tahun_registrasi',
        'nomor_bpkb',
        'kode_lokasi',
        'no_urut_pendaftaran',
        'berlaku_sampai'
        ]
}

#################################################################################################

config.TOLERANCE = 4

config.STNK_X_Y_RATIO = 3
config.STNK_SYNTH_RES = [1500, 1800]
config.STNK_SYNTH = {
    # [key_coords, value_coords, key_font, value_font, field_Type, max_value_length, start_with, num_of_words_per_line, num_of_lines]
    'no.': [[0.44491, 0.17509], [0.49621, 0.17509], [0.5, 0.75, 0.85], [0.5, 0.75, 0.85], "number", 10, ":    ", 1, 1],
    'nomor registrasi': [[0.0139225, 0.328519], [0.19103, 0.328519], [0.4, 0.45, 0.5], [0.5, 0.75, 0.85], "text", 10, ":    ", 1, 1],
    'nama pemilik': [[0.0139225, 0.37906], [0.19103, 0.37906], [0.4, 0.45, 0.5], [0.5, 0.75, 0.85], "text", 10, ":    ", 3, 1],
    'alamat': [[0.0139225, 0.436823], [0.19103, 0.436823], [0.4, 0.45, 0.5], [0.5, 0.75, 0.85], "text", 10, ":    ", 5, 2],
    'type': [[0.0139225, 0.60288], [0.19103, 0.60288], [0.35, 0.4, 0.45, 0.5], [0.5, 0.75, 0.85], "text", 10, ":    ", 1, 1],
    'jenis': [[0.0139225, 0.660649], [0.19103, 0.660649], [0.35, 0.4, 0.45, 0.5], [0.5, 0.75, 0.85], "text", 10, ":    ", 1, 1],
    'model': [[0.0139225, 0.716606], [0.19103, 0.716606], [0.35, 0.4, 0.45, 0.5], [0.5, 0.75, 0.85], "text", 10, ":    ", 1, 1],
    'tahun pembuatan': [[0.0139225, 0.77075], [0.19103, 0.77075], [0.35, 0.4, 0.45, 0.5], [0.5, 0.75, 0.85], "text", 10, ":    ", 1, 1],
    'isi silinder': [[0.0139225, 0.830324], [0.19103, 0.830324], [0.35, 0.4, 0.45, 0.5], [0.5, 0.75, 0.85], "text", 10, ":    ", 1, 1],
    'nomor ragka/nik/vin': [[0.0139225, 0.884476], [0.19103, 0.884476], [0.35, 0.4, 0.45, 0.5], [0.5, 0.75, 0.85], "text", 10, ":    ", 1, 1],
    'nomor mesin': [[0.0139225, 0.938628], [0.19103, 0.938628], [0.35, 0.4, 0.45, 0.5], [0.5, 0.75, 0.85], "text", 10, ":    ", 1, 1],
    'warna': [[0.372881, 0.5415162], [0.51193, 0.5415162], [0.35, 0.4, 0.45, 0.5], [0.5, 0.75, 0.85], "text", 10, ":    ", 1, 1],
    'bahan bakar': [[0.372881, 0.60288], [0.51193, 0.60288], [0.35, 0.4, 0.45, 0.5], [0.5, 0.75, 0.85], "text", 10, ":    ", 1, 1],
    'warna tnkb': [[0.372881, 0.660649], [0.51193, 0.660649], [0.35, 0.4, 0.45, 0.5], [0.5, 0.75, 0.85], "text", 10, ":    ", 1, 1],
    'tahun registrasi': [[0.372881, 0.716606], [0.51193, 0.716606], [0.35, 0.4, 0.45, 0.5], [0.5, 0.75, 0.85], "text", 10, ":    ", 1, 1],
    'nomor bpkb': [[0.372881, 0.77075], [0.51193, 0.77075], [0.35, 0.4, 0.45, 0.5], [0.5, 0.75, 0.85], "text", 10, ":    ", 1, 1],
    'kode lokasi': [[0.372881, 0.830324], [0.51193, 0.830324], [0.35, 0.4, 0.45, 0.5], [0.5, 0.75, 0.85], "text", 10, ":    ", 1, 1],
    'no urut pendaftaran': [[0.372881, 0.884476], [0.51193, 0.884476], [0.35, 0.4, 0.45, 0.5], [0.5, 0.75, 0.85], "text", 10, ":    ", 1, 1],
    'berlaku sampai': [[0.38619, 0.938628], [0.523005, 0.938628], [0.35, 0.4, 0.45, 0.5], [0.5, 0.75, 0.85], "date", 10, ":    ", 1, 1],
    'surat tanda nomor kendaraan bermotor': [[0.093695, 0.232558], None, [0.75, 0.85, 0.95], None, None, None, None, None, None],
}

config.KTP_X_Y_RATIO = 1.5
config.KTP_SYNTH_RES = [650, 750]
config.KTP_IMAGE = {
    'top_left': [0.7152777, 0.232662],
    'bottom_right': [0.954166, 0.7158836]
}
config.KTP_SYNTH = {
    # [key_coords, value_coords, key_font, value_font, field_Type, max_value_length, start_with, num_of_words_per_line, num_of_lines]
    'NIK': [[0.034722, 0.24832], [0.1972222, 0.24832], [0.75, 0.85], [0.75, 0.85], "number", 17, ":   ", 1, 1],
    'Nama': [[0.0319444, 0.328859], [0.2486111, 0.328859], [0.5, 0.55], [0.4, 0.45, 0.5], "text", 7, ": ", 3, 1],
    'Tempat/Tgl Lahir': [[0.0319444, 0.38255], [0.2486111, 0.38255], [0.5, 0.55], [0.4, 0.45, 0.5], "text", 10, ": ", 3, 1],
    'Jenis Kelamin': [[0.0319444, 0.422818], [0.2486111, 0.422818], [0.5, 0.55], [0.4, 0.45, 0.5], "text", 10, ": ", 1, 1],
    'Alamat': [[0.0319444, 0.474272], [0.2486111, 0.474272], [0.5, 0.55], [0.4, 0.45, 0.5], "text", 7, ": ", 4, 2],
    'RT/RW': [[0.07966, 0.574944], [0.2486111, 0.574944], [0.5, 0.55], [0.4, 0.45, 0.5], "text", 10, ": ", 1, 1],
    'Kel/Desa': [[0.07966, 0.624161], [0.2486111, 0.624161], [0.5, 0.55], [0.4, 0.45, 0.5], "text", 10, ": ", 1, 1],
    'Kecamatan': [[0.07966, 0.668903], [0.2486111, 0.668903], [0.5, 0.55], [0.4, 0.45, 0.5], "text", 10, ": ", 1, 1],
    'Agama': [[0.025, 0.720357], [0.2486111, 0.720357], [0.5, 0.55], [0.4, 0.45, 0.5], "text", 10, ": ", 1, 1],
    'Status Perkawinan': [[0.025, 0.7651006], [0.2486111, 0.7651006], [0.5], [0.4, 0.45, 0.5], "text", 10, ": ", 1, 1],
    'Pekerjaan': [[0.025, 0.82774], [0.2486111, 0.82774], [0.5, 0.55], [0.4, 0.45, 0.5], "text", 10, ": ", 1, 1],
    'Kewarganegaraan': [[0.025, 0.8724832], [0.2486111, 0.8724832], [0.5], [0.4, 0.45, 0.5], "text", 10, ": ", 1, 1],
    'Berlaku Hingga': [[0.025, 0.923937], [0.2486111, 0.923937], [0.5, 0.55], [0.4, 0.45, 0.5], "date", 10, ": ", 1, 1],
    'Name of Candidate': [[0.7625, 0.77181], None, [0.4, 0.45], None, None, None, None, None, None],
    '30-02-1947': [[0.775, 0.80984], None, [0.4, 0.45], None, None, None, None, None, None],
    'SIGN': [[0.7625, 0.894859], None, [0.6], None, None, None, None, None, None],
    'PROVINSI SUMATERA UTARA': [[0.25277777, 0.100671], None, [0.75], None, None, None, None, None, None],
    'KABUPATEN DELI SERDANG': [[0.25972222, 0.165548], None, [0.75], None, None, None, None, None, None],
}

config.SIM_X_Y_RATIO = 1.7
config.SIM_SYNTH_RES = [400, 500]
config.SIM_IMAGE = {
    'top_left': [0.052132, 0.3967611],
    'bottom_right': [0.303317, 0.923076]
}
config.SIM_SYNTH = {
    # [key_coords, value_coords, key_font, value_font, field_Type, max_value_length, start_with, num_of_words_per_line, num_of_lines]
    'Nama': [[0.104265, 0.327935], [0.201421, 0.327935], [0.5, 0.55], [0.4, 0.45, 0.5], "text", 7, ": ", 3, 1],
    'Alamat': [[0.2085308, 0.3805668], [0.3246445, 0.3805668], [0.5, 0.55], [0.4, 0.45, 0.5], "text", 7, ": ", 4, 2],
    'Tempat &': [[0.331753, 0.510121], [0.4952606, 0.510121], [0.5, 0.55], [0.4, 0.45, 0.5], "text", 10, ": ", 3, 1],
    'Tgl.Lahir': [[0.331753, 0.554655], [0.4952606, 0.554655], [0.5, 0.55], [0.4, 0.45, 0.5], "text", 10, ": ", 1, 1],
    'Tinggi': [[0.331753, 0.6072814], [0.4952606, 0.6072814], [0.5, 0.55], [0.4, 0.45, 0.5], "text", 10, ": ", 1, 1],
    'Pekerjaan': [[0.331753, 0.6477732], [0.4952606, 0.6477732], [0.5, 0.55], [0.4, 0.45, 0.5], "text", 10, ": ", 1, 1],
    'No. SIM': [[0.331753, 0.70004048], [0.4952606, 0.70004048], [0.5, 0.55], [0.4, 0.45, 0.5], "text", 10, ": ", 1, 1],
    'Berlaku s/d': [[0.331753, 0.7287449], [0.4952606, 0.7287449], [0.5, 0.55], [0.4, 0.45, 0.5], "date", 10, ": ", 1, 1],
}