echo "Training a binary model with AMAISE_human_host_7000_12000_length_range_with_all_virus_2mer_3mer UNBALANCED DATASET"
python3 binary_train_imbalanced_dataset.py -i /home/shared_tools_data/seq2vec/AMAISE_human_host_7000_12000_length_range_with_all_virus_2mer_3mer.csv -l /home/shared_tools_data/AMAISE_dataset/training_dataset/human_host_7000_12000_length_range_with_all_virus_ref/amaise_train_7k_12k_with_all_virus_true_labels_binary.csv -m /home/chadmi.20/model_training/model/AMAISE_train_dataset/human_host_7000_1200_length_range_with_all_virus/binaryFFNN_modified_4_2mer_3mer.pth -o /home/chadmi.20/model_training/output/AMAISE_train_dataset/human_host_7000_1200_length_range_with_all_virus/binaryFFNN_modified_4_2mer_3mer

python3 binary_classification.py -i /home/shared_tools_data/AMAISE_dataset/test_datasets/human_percentages_1000bp_grater/human_99/human_99_nanopore_test_len_grater_1000.fasta -t fasta -k /home/shared_tools_data/seq2vec/AMAISE_test_human_99_nanopore_test_len_greater_1000_2mer_3mer.csv -m /home/chadmi.20/model_training/model/AMAISE_train_dataset/human_host_7000_1200_length_range_with_all_virus/binaryFFNN_modified_4_2mer_3mer.pth -o /home/chadmi.20/compare_AMAISE/AMAISE_train_dataset_human_host_7000_1200_length_range_with_all_virus/AMAISE_test_human_percentages_1000bp_greater/human_99_Binary_FFNN_modified_4_2mer_3mer


python3 binary_evaluation.py -p /home/chadmi.20/compare_AMAISE/AMAISE_train_dataset_human_host_7000_1200_length_range_with_all_virus/AMAISE_test_human_percentages_1000bp_greater/human_99_Binary_FFNN_modified_4_2mer_3mer/predictions.csv -t /home/shared_tools_data/AMAISE_dataset/test_datasets/human_percentages_1000bp_grater/human_99/human_99_nanopore_test_true_labels_length_grater_1000_binary.csv -lf /home/chadmi.20/compare_AMAISE/AMAISE_train_dataset_human_host_7000_1200_length_range_with_all_virus/AMAISE_test_human_percentages_1000bp_greater/human_99_Binary_FFNN_modified_4_2mer_3mer/infoEvaluation.log


python3 binary_classification.py -i /home/shared_tools_data/AMAISE_dataset/test_datasets/human_percentages_1000bp_grater/human_1/human_1_nanopore_test_len_grater_1000.fasta -t fasta -k /home/shared_tools_data/seq2vec/AMAISE_test_human_1_nanopore_test_len_greater_1000_2mer_3mer.csv -m /home/chadmi.20/model_training/model/AMAISE_train_dataset/human_host_7000_1200_length_range_with_all_virus/binaryFFNN_modified_4_2mer_3mer.pth -o /home/chadmi.20/compare_AMAISE/AMAISE_train_dataset_human_host_7000_1200_length_range_with_all_virus/AMAISE_test_human_percentages_1000bp_greater/human_1_Binary_FFNN_modified_4_2mer_3mer


python3 binary_evaluation.py -p /home/chadmi.20/compare_AMAISE/AMAISE_train_dataset_human_host_7000_1200_length_range_with_all_virus/AMAISE_test_human_percentages_1000bp_greater/human_1_Binary_FFNN_modified_4_2mer_3mer/predictions.csv -t /home/shared_tools_data/AMAISE_dataset/test_datasets/human_percentages_1000bp_grater/human_1/human_1_nanopore_test_true_labels_length_grater_1000_binary.csv -lf /home/chadmi.20/compare_AMAISE/AMAISE_train_dataset_human_host_7000_1200_length_range_with_all_virus/AMAISE_test_human_percentages_1000bp_greater/human_1_Binary_FFNN_modified_4_2mer_3mer/infoEvaluation.log
