echo "Training a new model"
python3 xgboostTrain.py  -i /home/shared_tools_data/seq2vec/human_train_kraken_kmer_6.csv -l /mnt/disk-3/home/shared/human/train/human_train_kraken_labels.csv -m /home/venukshi.20/AMAISE_PRO_2.0_XGBoost_trainedModel/xgbclassifier_kmer6/xgbclassifier.pth -o /home/venukshi.20/AMAISE_PRO_2.0_XGBoost_trainedModel/xgbclassifier_kmer6/training/output
python3 xgboostTrain.py  -i /home/shared_tools_data/seq2vec/human_train_kraken_kmer_3.csv -l /mnt/disk-3/home/shared/human/train/human_train_kraken_labels.csv -m /home/venukshi.20/AMAISE_PRO_2.0_XGBoost_trainedModel/xgbclassifier_kmer6/xgbclassifier.pth -o /home/venukshi.20/AMAISE_PRO_2.0_XGBoost_trainedModel/xgbclassifier_kmer6/training/output

echo "Classifcation"
python3 mclass_classification_xgboost.py -i /mnt/disk-3/home/shared/human/test/human_test_new.fasta -t fasta -k  /home/venukshi.20/model_training/seq2vec/human_test_new_kmer_6.csv -m /home/venukshi.20/AMAISE_PRO_2.0_XGBoost_trainedModel/xgbclassifier_kmer6/xgbclassifier.pth -o /home/venukshi.20/AMAISE_PRO_2.0_XGBoost_trainedModel/xgbclassifier_kmer6/classified
# python3 mclass_classification_xgboost.py -i /mnt/disk-3/home/shared/human/test/human_test_new.fasta -t fasta -k  /home/venukshi.20/human_test_new/human_test_new_seq2vec.csv -m /home/venukshi.20/AMAISE_PRO_2.0_XGBoost_trainedModel/xgbclassifier_kmer6/xgbclassifier.pth -o /home/venukshi.20/AMAISE_PRO_2.0_XGBoost_trainedModel/xgbclassifier_kmer6/classified


echo "Evaluate results"
python3 evaluation.py -p /home/venukshi.20/AMAISE_PRO_2.0_XGBoost_trainedModel/xgbclassifier_kmer6/classified/predictions.csv -t /mnt/disk-3/home/shared/human/test/human_test_new_labels_new.csv
