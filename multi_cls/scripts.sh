python run.py --train True \
              --predict True \
              --maxlen 512 \
              --epochs 10 \
              --batch_size 4 \
              --train_data ./data/labeled_data.csv \
              --supply_train_data ./data/labeled_supply_data.csv \
              --add_supply_data True \
              --test_data ./data/test_data.csv \
              --pretrain_model /home/david/pretrain_model/google_bert/chinese_L-12_H-768_A-12 \
              --task category \
              --train_data_split_ratio 0.2 
              