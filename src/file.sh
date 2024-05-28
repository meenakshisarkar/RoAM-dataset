# cd ../models/bair_robot_pushing_small_lfvg_fs_image_w=64_K=2_T=10_batch_size=60_alpha=1.0_beta=0.001_lr=0.0001_z-dim=64_hidden-dim_pred=256_no_epochs=300_beta10.99/Generator
pwd
# scp -r ../../../../tensorflow_datasets/kth gcdsl_tbl@10.40.73.73:/home/gcdsl_tbl/tensorflow_datasets/
cd ../../../catkin_ws/RoAM_dataset/processed/
# cd ../../../data/kth
pwd
scp -r tfrecord gcdsl@10.40.19.40:/home/gcdsl/meenakshi_python2.7/VANET/data/RoAM/train/
# scp -r processed_64 gcdsl@10.40.19.40:/home/gcdsl/meenakshi_python2.7/VANET/data/KTH/
# scp -r ../../../../tensorflow_datasets/kth_test gcdsl_tbl@10.40.73.73:/home/gcdsl_tbl/tensorflow_datasets/
# scp -r ../src asemeena@nvidia-dgx.serc.iisc.ac.in:/localscratch/asemeena/kth_fs
