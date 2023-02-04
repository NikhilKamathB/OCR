import os
import pandas as pd
from strsimpy.levenshtein import Levenshtein
from pathlib import Path 
import argparse
import sys 
import mlflow 

cols = {'ktp': ['agama',
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
        'tgl_lahir'],
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
        'berlaku_sd',
        'urut',
        'kohir',
        'nik',
        'bbnkb_pokok',
        'pkb_pokok',
        'swdkllj_pokok',
        'biaya_adm_stnk_pokok',
        'biaya_adm_tnkb_pokok',
        'jumlah_pokok',
        'bbnkb_sanski_administratif',
        'pkb_sanski_administratif',
        'swdkllj_sanski_administratif',
        'biaya_adm_stnk_sanski_administratif',
        'biaya_adm_tnkb_sanski_administratif',
        'jumlah_sanski_administratif',
        'bbnkb_jumlah',
        'pkb_jumlah',
        'swdkllj_jumlah',
        'biaya_adm_stnk_jumlah',
        'biaya_adm_tnkb_jumlah',
        'jumlah_jumlah',
        'ditetapkan_tanggal',
        'penaksir_pajak',
        'merk_type',
        'jenis_model',
        'tahun_pembuatan_perakitan',
        'warna_kb',
        'isi_silinder_hp',
        'nomor_rangka_nik',
        'nomor_mesin',
        'no_bpkb',
        'bahan_bakar',
        'warna_tnkb',
        'kepemilikan_ke',
        'no_registrasilama',
        'kode_njkb',
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
        ]}

def scores(anno_file, inf_file, file_type):

    dann = pd.read_csv(anno_file)
    dann.fillna('', inplace=True)
    dinf = pd.read_csv(inf_file)
    dinf.fillna('', inplace=True)

    #hackfix
    if 'nik' in dinf:
        dinf['nik'] = dinf['nik'].astype(str)
    if 'no_sim' in dinf:
        dinf['no_sim'] = dinf['no_sim'].astype(str)
    if 'biaya_adm_stnk_sanski_administratif' in dinf:
        dinf['biaya_adm_stnk_sanski_administratif'] =  dinf['biaya_adm_stnk_sanski_administratif'].astype(str)
        dann['biaya_adm_stnk_sanski_administratif'] =  dann['biaya_adm_stnk_sanski_administratif'].astype(str)

    # import pdb; pdb.set_trace()
    df = pd.merge(dann, dinf, on='image', suffixes=("_annotated", ""))
    
        
    fields = cols[file_type]
    
    ldist = Levenshtein()
    
    def apply_levenshtein(x):
        for field in fields:
            fieldstr = str(x[field]).lower().strip()
            annotstr = str(x[f"{field}_annotated"]).lower().strip()
            x[f"{field}_dist"] = int(ldist.distance(fieldstr, annotstr))

            # zero division error corrections. if both annotated and source are blank. we consider it as 0% error.
            if x[f"{field}_dist"] == 0:
                x[f"{field}_err"] = 0
            else:
                x[f"{field}_err"] = min(x[f"{field}_dist"] * 100 / len(annotstr) if len(annotstr) > 0 else 100, 100)
        return x

    df = df.apply(apply_levenshtein, axis=1)

    return df


if __name__=="__main__":
    if not os.getenv("MLFLOW_TRACKING_URI"):
        print("missing MLFLOW_TRACKING_URI")
        sys.exit(1)

    parser = argparse.ArgumentParser()
    parser.add_argument("file_type")
    parser.add_argument("csv_annotated")
    parser.add_argument("csv_inference")
    parser.add_argument("--run_name")
    parser.add_argument("--experiment_name")
    
    args = parser.parse_args()


    mlflow.set_experiment(experiment_name=args.experiment_name)
    with mlflow.start_run(run_name=args.run_name):

        df = scores(args.csv_annotated, args.csv_inference, args.file_type)
        print(df.shape)

        metrics = df[[f + "_err" for f in cols[args.file_type]]].mean()
        mlflow.log_metrics(metrics)
        mlflow.log_metric("total_error", metrics.mean())

        for k,v in metrics.iteritems():
            print(f"{k}\t{v:.2f}")        

        print(f"total\t{metrics.mean():.2f}")

