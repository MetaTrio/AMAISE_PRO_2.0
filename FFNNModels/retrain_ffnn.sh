echo "classify AMAISE_dataset/nanopore/human_50"
python3 mclass_classification_ffnn.py -i /home/shared_tools_data/AMAISE_dataset/nanopore/human_50/nanopore_human_50.fasta -t fasta -k /home/shared_tools_data/seq2vec/AMAISE_dataset_nanopore_human_50_5mer.csv -m /home/chadmi.20/model_training/model/AMAISE_train_dataset/human_host/modelInitialFFNN_5mer.pth -o /home/chadmi.20/compare_AMAISE/AMAISE_train_dataset_human_host/test/nanopore/human_50

echo "evaluate human_50"
python3 evaluation_ffnn.py -p /home/chadmi.20/compare_AMAISE/AMAISE_train_dataset_human_host/test/nanopore/human_50/predictions.csv -t /home/shared_tools_data/AMAISE_dataset/nanopore/human_50/amaise_human_50_test_true_labels.csv -lf /home/chadmi.20/compare_AMAISE/AMAISE_train_dataset_human_host/test/nanopore/human_50/infoEvaluation.log

