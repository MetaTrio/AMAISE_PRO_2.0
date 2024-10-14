echo "Retraining"

python3 retrain_ffnn.py -i /home/shared_tools_data/seq2vec/shark_train_5mer.csv -l /mnt/disk-3/home/shared/shark/train/shark_train_labels.csv -m /home/chadmi.20/model_training/model/human_train_kraken/retrain_shark_train/modelInitialFFNN5mer.pth -p /home/chadmi.20/model_training/model/human_train_kraken/modelInitialFFNN5mer.pth -o /home/chadmi.20/model_training/output/human_train_kraken/retrain_shark_train/modelInitialFFNN5mer

echo "Classifcation using retrained model"

python3 mclass_classification_ffnn.py -i /mnt/disk-3/home/shared/shark/test/shark_test.fasta -t fasta -k /home/shared_tools_data/seq2vec/shark_test_5mer.csv -m /home/chadmi.20/model_training/model/human_train_kraken/retrain_shark_train/modelInitialFFNN5mer.pth -o /home/chadmi.20/hosts/shark_test/human_train_kraken/retrain_shark_train/modelInitialFFNN5mer_output

echo "Evaluate results for shark_test"

python3 evaluation_ffnn.py -p /home/chadmi.20/hosts/shark_test/human_train_kraken/retrain_shark_train/modelInitialFFNN5mer_output/predicted_probabilities.csv -t /mnt/disk-3/home/shared/shark/test/shark_test_labels.csv -lf /home/chadmi.20/hosts/shark_test/human_train_kraken/retrain_shark_train/modelInitialFFNN5mer_output/infoEvaluation.log