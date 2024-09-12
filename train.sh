echo "Generate k-mer frequency vector"

/home/venukshi.20/seq2vec/build/seq2vec -f /mnt/disk-3/home/shared/human/train/human_train_kraken.fasta -o /home/venukshi.20/model_training/seq2vec/human_train_kraken_kmer_3.csv

echo "Training a new model"

python3 train.py -i /home/venukshi.20/model_training/seq2vec/human_train_kraken_kmer_3.csv -l /mnt/disk-3/home/shared/human/train/human_train_kraken_labels.csv -m /home/venukshi.20/model_training/model/model.pth -o  /home/venukshi.20/model_training/output/output

echo "Classify human_test_new.fasta using new model"

python3 mclass_classification.py -i /mnt/disk-3/home/shared/human/test/human_test_new.fasta -t fasta -k /home/venukshi.20/human_test_new/human_test_new_seq2vec.csv -m /home/venukshi.20/model_training/model/model1.pth -o /home/venukshi.20/human_test_new/model1_amaise_output

echo "Evaluate results"

python3 evaluation_with_roc.py -p /home/venukshi.20/human_test_new/model1_amaise_output/predicted_probabilities.csv -t /mnt/disk-3/home/shared/human/test/human_test_new_labels_new.csv