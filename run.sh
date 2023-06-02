# train VGG16Trans model on FSC
python train.py --tag final --device 0 --scheduler step --step 70 \
--max-epoch 200 --batch-size 8 --lr 1e-4 --val-start 1 --val-epoch 1 \
--num-workers 4 --weight-decay 1e-4 --save-dir './final_checkpoint'
