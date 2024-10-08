echo "Generate k-mer frequency vector"

/home/venukshi.20/seq2vec/build/seq2vec -f /mnt/disk-3/home/shared/human/train/human_train_kraken.fasta -o /home/venukshi.20/model_training/seq2vec/human_train_kraken_kmer_3.csv
/home/venukshi.20/seq2vec/build/seq2vec -f /mnt/disk-3/home/shared/human/train/human_train_kraken.fasta -o /home/venukshi.20/model_training/seq2vec/human_train_kraken_kmer_6.csv -k 6
/home/venukshi.20/seq2vec/build/seq2vec -f /mnt/disk-3/home/shared/human/test/human_test_new.fasta -o /home/venukshi.20/model_training/seq2vec/human_test_new_kmer_5.csv -k 5

/home/venukshi.20/seq2vec/build/seq2vec -f /mnt/disk-3/home/shared/human/test/human_test.fasta -o /home/venukshi.20/human_test_mix/human_test_seq2vec.csv

echo "Training a new model"

python3 xgboostTrain.py  -i /home/venukshi.20/model_training/seq2vec/human_train_kraken_kmer_3.csv -l /mnt/disk-3/home/shared/human/train/human_train_kraken_labels.csv -m /home/venukshi.20/AMAISE_PRO_2.0/XGBoost/trainedModel/xgbclassifier/xgbclassifier.pth -o /home/venukshi.20/AMAISE_PRO_2.0/XGBoost/trainedModel/xgbclassifier/training/output

python3 xgboostTrain.py  -i /home/venukshi.20/model_training/seq2vec/human_train_kraken_kmer_5.csv -l /mnt/disk-3/home/shared/human/train/human_train_kraken_labels.csv -m /home/venukshi.20/AMAISE_PRO_2.0_XGBoost_trainedModel/xgbclassifier_kmer5_tune/xgbclassifier.pth -o /home/venukshi.20/AMAISE_PRO_2.0_XGBoost_trainedModel/xgbclassifier_kmer5_tune/training/output


/home/venukshi.20/model_training/seq2vec/human_test_new_kmer_5.csv
echo "Classify human_test_new.fasta using new model"

#change model and output path
python3 mclass_classification_xgboost.py -i /mnt/disk-3/home/shared/human/test/human_test_new.fasta -t fasta -k /home/venukshi.20/human_test_new/human_test_new_seq2vec.csv -m /home/venukshi.20/AMAISE_PRO_2.0/XGBoost/trainedModel/model_epoch100.pth -o /home/venukshi.20/AMAISE_PRO_2.0/XGBoost/trainedModel/model_epoch100/classification_results
python3 mclass_classification_xgboost.py -i /mnt/disk-3/home/shared/human/test/human_test_new.fasta -t fasta -k /home/venukshi.20/human_test_new/human_test_new_seq2vec.csv -m /home/venukshi.20/AMAISE_PRO_2.0/XGBoost/trainedModel/xgbclassifier.pth -o /home/venukshi.20/AMAISE_PRO_2.0/XGBoost/trainedModel/xgbclassifier/classified
python3 mclass_classification_xgboost.py -i /mnt/disk-3/home/shared/human/test/human_test_new.fasta -t fasta -k  /home/venukshi.20/model_training/seq2vec/human_test_new_kmer_5.csv -m /home/venukshi.20/AMAISE_PRO_2.0/XGBoost/trainedModel/xgbclassifier_kmer5/xgbclassifier.pth -o /home/venukshi.20/AMAISE_PRO_2.0/XGBoost/trainedModel/xgbclassifier_kmer5/classified
python3 mclass_classification_xgboost.py -i /mnt/disk-3/home/shared/human/test/human_test_new.fasta -t fasta -k  /home/venukshi.20/model_training/seq2vec/human_test_new_kmer_5.csv -m /home/venukshi.20/AMAISE_PRO_2.0_XGBoost_trainedModel/xgbclassifier_kmer5_tune/xgbclassifier.pth -o /home/venukshi.20/AMAISE_PRO_2.0_XGBoost_trainedModel/xgbclassifier_kmer5_tune/classified


echo "Evaluate results"

#change predicted result
python3 evaluation.py -p /home/venukshi.20/AMAISE_PRO_2.0/XGBoost/trainedModel/model_epoch100/classification_results/predictions.csv -t /mnt/disk-3/home/shared/human/test/human_test_new_labels_new.csv
python3 evaluation.py -p /home/venukshi.20/AMAISE_PRO_2.0/XGBoost/trainedModel/xgbclassifier_kmer5/classified/predictions.csv -t /mnt/disk-3/home/shared/human/test/human_test_new_labels_new.csv
python3 evaluation.py -p /home/venukshi.20/AMAISE_PRO_2.0_XGBoost_trainedModel/xgbclassifier_kmer5_tune/classified/predictions.csv -t /mnt/disk-3/home/shared/human/test/human_test_new_labels_new.csv


#human train mix

/home/venukshi.20/seq2vec/build/seq2vec -f /mnt/disk-3/home/shared/human/train/human_train_final.fasta -o /home/venukshi.20/model_training/seq2vec/human_train_final_kmer_3.csv

python3 xgboostTrain.py  -i /home/venukshi.20/model_training/seq2vec/human_train_final_kmer_3.csv -l /mnt/disk-3/home/shared/human/train/human_train_final_labels.csv  -m /home/venukshi.20/AMAISE_PRO_2.0/XGBoost/trainedModel/model_train_final_epoch_100/model_train_final_epoch_100.pth -o /home/venukshi.20/AMAISE_PRO_2.0/XGBoost/trainedModel/model_train_final_epoch_100/training

python3 mclass_classification_xgboost.py -i /mnt/disk-3/home/shared/human/test/human_test.fasta -t fasta -k /home/venukshi.20/human_test_mix/human_test_seq2vec.csv -m /home/venukshi.20/AMAISE_PRO_2.0/XGBoost/trainedModel/model_train_final_epoch_100/model_train_final_epoch_100.pth -o /home/venukshi.20/AMAISE_PRO_2.0/XGBoost/trainedModel/model_train_final_epoch_100/classified

python3 evaluation.py -p /home/venukshi.20/AMAISE_PRO_2.0/XGBoost/trainedModel/model_train_final_epoch_100/classified/predictions.csv -t /mnt/disk-3/home/shared/human/test/human_test_labels.csv


# Grid search

python3 grid_search.py  -i /home/venukshi.20/model_training/seq2vec/human_train_kraken_kmer_5.csv -l /mnt/disk-3/home/shared/human/train/human_train_kraken_labels.csv -m /home/venukshi.20/AMAISE_PRO_2.0/XGBoost/trainedModel/grid_search_best_model/xgbclassifier.pth -o /home/venukshi.20/AMAISE_PRO_2.0/XGBoost/trainedModel/grid_search_best_model/training/output
