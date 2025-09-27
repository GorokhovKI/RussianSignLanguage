.PHONY: extract analyze train plot demo

extract:
	python extract_features.py --mediapipe_json data/slovo_mediapipe.json --out_dir results/features --fixed_length 64

analyze:
	python data_analysis.py --annotations data/annotations.csv --mediapipe_json data/slovo_mediapipe.json --threshold 0.12

train:
	python train_and_save.py --features_dir results/features --annotations data/annotations.csv --models_dir results/models --epochs 60 --batch_size 32

plot:
	python plot_metrics.py --history_csv results/metrics/training_history.csv --out_dir results/plots

demo:
	python demo_realtime.py --model_path results/models/best_model.pth