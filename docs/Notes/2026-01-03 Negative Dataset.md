# Dataset zusammenstellen -> wav und flac nicht im gitignore
data/raw/demand -> https://www.kaggle.com/datasets/aanhari/demand-dataset
data/raw/librispeech -> https://www.openslr.org/12/

data/neg_wavs -> werden erstellt von make_neg_dataset.py


# False Alarms testen
# python app-offline.py   --mode offline_neg  --neg_dir data\neg_wavs  --model_path .\models\hey_mycroft_v0.1.onnx  --threshold 0.2  --hop_ms 160  --capture_seconds 0  

python app-offline.py --mode offline --audio_dir data\neg_wavs --model_path .\models\hey_mycroft_v0.1.onnx --threshold 0.2 --hop_ms 160 --capture_seconds 0

python app-offline.py --mode offline --audio_dir data\neg_wavs --model_path .\models\how_do_you_wanna_do_this.onnx --threshold 0.2 --hop_ms 160 --capture_seconds 0


# sweep über verschieden threshholds

foreach ($t in 0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55) {  python app-offline.py   --mode offline  --audio_dir data\neg_wavs  --model_path .\models\hey_mycroft_v0.1.onnx  --threshold $t  --hop_ms 160  --capture_seconds 0  }

# Issues


