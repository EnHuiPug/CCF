python extract_feature.py --maxlen 512 \
              --source_data ./data/unlabeled_data.csv \
              --task unlabeled
              --finetune True
              --pretrain_model /home/david/pretrain_model/google_bert/chinese_L-12_H-768_A-12
              