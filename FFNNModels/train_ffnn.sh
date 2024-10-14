echo "Training a new model"

python3 ffTrain.py -i /home/shared_tools_data/seq2vec/AMAISE_dataset_human_host_5mer.csv -l /home/shared_tools_data/AMAISE_dataset/training_dataset/human_host/amaise_train_true_labels.csv -m /home/chadmi.20/model_training/model/AMAISE_train_dataset/human_host/modelInitialFFNN_5mer.pth -o /home/chadmi.20/model_training/output/AMAISE_train_dataset/human_host/modelInitialFFNN_5mer

echo "Classifcation"

python3 mclass_classification_ffnn.py -i /mnt/disk-3/home/shared/human/test/human_test_new.fasta -t fasta -k /home/shared_tools_data/seq2vec/human_test_new_kmer_5.csv -m /home/chadmi.20/model_training/model/AMAISE_train_dataset/human_host/modelInitialFFNN_5mer.pth -o /home/chadmi.20/human_test_new/AMAISE_train_dataset/human_host/modelInitialFFNN5mer_output

echo "Evaluate results"

python3 evaluation_ffnn.py -p /home/chadmi.20/human_test_new/AMAISE_train_dataset/human_host/modelInitialFFNN5mer_output/predictions.csv -t /mnt/disk-3/home/shared/human/test/human_test_new_labels_new.csv -lf /home/chadmi.20/human_test_new/AMAISE_train_dataset/human_host/modelInitialFFNN5mer_output/infoEvaluation.log