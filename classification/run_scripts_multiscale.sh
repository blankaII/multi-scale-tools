#!/bin/bash

#UPPER REGION 
#features

python Fully_supervised_training_combine_features_multi.py -n 5 -c resnet34 -m 10 5 -f True -b 64 -d 0.2 -e 10 -r upper_region -i /home/niccolo/ExamodePipeline/Multi_Scale_Tools/csv_folder/classification/0/upper_region_partitions/ -o /home/niccolo/ExamodePipeline/Multi_Scale_Tools/model_weights/classification/0/multi_scale_upper_region/combine_features/
python Fully_supervised_training_combine_features_multi.py -n 5 -c resnet34 -m 10 5 -f True -b 64 -d 0.2 -e 10 -r upper_region -i /home/niccolo/ExamodePipeline/Multi_Scale_Tools/csv_folder/classification/1/upper_region_partitions/ -o /home/niccolo/ExamodePipeline/Multi_Scale_Tools/model_weights/classification/1/multi_scale_upper_region/combine_features/
python Fully_supervised_training_combine_features_multi.py -n 5 -c resnet34 -m 10 5 -f True -b 64 -d 0.2 -e 10 -r upper_region -i /home/niccolo/ExamodePipeline/Multi_Scale_Tools/csv_folder/classification/2/upper_region_partitions/ -o /home/niccolo/ExamodePipeline/Multi_Scale_Tools/model_weights/classification/2/multi_scale_upper_region/combine_features/

python Fully_supervised_training_combine_features_multi.py -n 5 -c resnet34 -m 20 10 5 -f True -b 64 -d 0.2 -e 10 -r upper_region -i /home/niccolo/ExamodePipeline/Multi_Scale_Tools/csv_folder/classification/0/upper_region_partitions/ -o /home/niccolo/ExamodePipeline/Multi_Scale_Tools/model_weights/classification/0/multi_scale_upper_region/combine_features/
python Fully_supervised_training_combine_features_multi.py -n 5 -c resnet34 -m 20 10 5 -f True -b 64 -d 0.2 -e 10 -r upper_region -i /home/niccolo/ExamodePipeline/Multi_Scale_Tools/csv_folder/classification/1/upper_region_partitions/ -o /home/niccolo/ExamodePipeline/Multi_Scale_Tools/model_weights/classification/1/multi_scale_upper_region/combine_features/
python Fully_supervised_training_combine_features_multi.py -n 5 -c resnet34 -m 20 10 5 -f True -b 64 -d 0.2 -e 10 -r upper_region -i /home/niccolo/ExamodePipeline/Multi_Scale_Tools/csv_folder/classification/2/upper_region_partitions/ -o /home/niccolo/ExamodePipeline/Multi_Scale_Tools/model_weights/classification/2/multi_scale_upper_region/combine_features/

#probs
python Fully_supervised_training_combine_probs_multi.py -n 5 -c resnet34 -m 10 5 -f True -b 64 -d 0.2 -e 10 -r upper_region -i /home/niccolo/ExamodePipeline/Multi_Scale_Tools/csv_folder/classification/0/upper_region_partitions/ -o /home/niccolo/ExamodePipeline/Multi_Scale_Tools/model_weights/classification/0/multi_scale_upper_region/combine_probs/
python Fully_supervised_training_combine_probs_multi.py -n 5 -c resnet34 -m 10 5 -f True -b 64 -d 0.2 -e 10 -r upper_region -i /home/niccolo/ExamodePipeline/Multi_Scale_Tools/csv_folder/classification/1/upper_region_partitions/ -o /home/niccolo/ExamodePipeline/Multi_Scale_Tools/model_weights/classification/1/multi_scale_upper_region/combine_probs/
python Fully_supervised_training_combine_probs_multi.py -n 5 -c resnet34 -m 10 5 -f True -b 64 -d 0.2 -e 10 -r upper_region -i /home/niccolo/ExamodePipeline/Multi_Scale_Tools/csv_folder/classification/2/upper_region_partitions/ -o /home/niccolo/ExamodePipeline/Multi_Scale_Tools/model_weights/classification/2/multi_scale_upper_region/combine_probs/

python Fully_supervised_training_combine_probs_multi.py -n 5 -c resnet34 -m 20 10 5 -f True -b 64 -d 0.2 -e 10 -r upper_region -i /home/niccolo/ExamodePipeline/Multi_Scale_Tools/csv_folder/classification/0/upper_region_partitions/ -o /home/niccolo/ExamodePipeline/Multi_Scale_Tools/model_weights/classification/0/multi_scale_upper_region/combine_probs/
python Fully_supervised_training_combine_probs_multi.py -n 5 -c resnet34 -m 20 10 5 -f True -b 64 -d 0.2 -e 10 -r upper_region -i /home/niccolo/ExamodePipeline/Multi_Scale_Tools/csv_folder/classification/1/upper_region_partitions/ -o /home/niccolo/ExamodePipeline/Multi_Scale_Tools/model_weights/classification/1/multi_scale_upper_region/combine_probs/
python Fully_supervised_training_combine_probs_multi.py -n 5 -c resnet34 -m 20 10 5 -f True -b 64 -d 0.2 -e 10 -r upper_region -i /home/niccolo/ExamodePipeline/Multi_Scale_Tools/csv_folder/classification/2/upper_region_partitions/ -o /home/niccolo/ExamodePipeline/Multi_Scale_Tools/model_weights/classification/2/multi_scale_upper_region/combine_probs/

#MULTICENTER
#features
python Fully_supervised_training_combine_features_multi.py -n 5 -c resnet34 -m 10 5 -f True -b 64 -d 0.2 -e 10 -r multi_center -i /home/niccolo/ExamodePipeline/Multi_Scale_Tools/csv_folder/classification/0/multicenter_partitions/ -o /home/niccolo/ExamodePipeline/Multi_Scale_Tools/model_weights/classification/0/multi_scale_multi_center/combine_features/
python Fully_supervised_training_combine_features_multi.py -n 5 -c resnet34 -m 10 5 -f True -b 64 -d 0.2 -e 10 -r multi_center -i /home/niccolo/ExamodePipeline/Multi_Scale_Tools/csv_folder/classification/1/multicenter_partitions/ -o /home/niccolo/ExamodePipeline/Multi_Scale_Tools/model_weights/classification/1/multi_scale_multi_center/combine_features/
python Fully_supervised_training_combine_features_multi.py -n 5 -c resnet34 -m 10 5 -f True -b 64 -d 0.2 -e 10 -r multi_center -i /home/niccolo/ExamodePipeline/Multi_Scale_Tools/csv_folder/classification/2/multicenter_partitions/ -o /home/niccolo/ExamodePipeline/Multi_Scale_Tools/model_weights/classification/2/multi_scale_multi_center/combine_features/

python Fully_supervised_training_combine_features_multi.py -n 5 -c resnet34 -m 20 10 5 -f True -b 64 -d 0.2 -e 10 -r multi_center -i /home/niccolo/ExamodePipeline/Multi_Scale_Tools/csv_folder/classification/0/multicenter_partitions/ -o /home/niccolo/ExamodePipeline/Multi_Scale_Tools/model_weights/classification/0/multi_scale_multi_center/combine_features/
python Fully_supervised_training_combine_features_multi.py -n 5 -c resnet34 -m 20 10 5 -f True -b 64 -d 0.2 -e 10 -r multi_center -i /home/niccolo/ExamodePipeline/Multi_Scale_Tools/csv_folder/classification/1/multicenter_partitions/ -o /home/niccolo/ExamodePipeline/Multi_Scale_Tools/model_weights/classification/1/multi_scale_multi_center/combine_features/
python Fully_supervised_training_combine_features_multi.py -n 5 -c resnet34 -m 20 10 5 -f True -b 64 -d 0.2 -e 10 -r multi_center -i /home/niccolo/ExamodePipeline/Multi_Scale_Tools/csv_folder/classification/2/multicenter_partitions/ -o /home/niccolo/ExamodePipeline/Multi_Scale_Tools/model_weights/classification/2/multi_scale_multi_center/combine_features/

#probs
python Fully_supervised_training_combine_probs_multi.py -n 5 -c resnet34 -m 10 5 -f True -b 64 -d 0.2 -e 10 -r multi_center -i /home/niccolo/ExamodePipeline/Multi_Scale_Tools/csv_folder/classification/0/multicenter_partitions/ -o /home/niccolo/ExamodePipeline/Multi_Scale_Tools/model_weights/classification/0/multi_scale_multi_center/combine_probs/
python Fully_supervised_training_combine_probs_multi.py -n 5 -c resnet34 -m 10 5 -f True -b 64 -d 0.2 -e 10 -r multi_center -i /home/niccolo/ExamodePipeline/Multi_Scale_Tools/csv_folder/classification/1/multicenter_partitions/ -o /home/niccolo/ExamodePipeline/Multi_Scale_Tools/model_weights/classification/1/multi_scale_multi_center/combine_probs/
python Fully_supervised_training_combine_probs_multi.py -n 5 -c resnet34 -m 10 5 -f True -b 64 -d 0.2 -e 10 -r multi_center -i /home/niccolo/ExamodePipeline/Multi_Scale_Tools/csv_folder/classification/2/multicenter_partitions/ -o /home/niccolo/ExamodePipeline/Multi_Scale_Tools/model_weights/classification/2/multi_scale_multi_center/combine_probs/

python Fully_supervised_training_combine_probs_multi.py -n 5 -c resnet34 -m 20 10 5 -f True -b 64 -d 0.2 -e 10 -r multi_center -i /home/niccolo/ExamodePipeline/Multi_Scale_Tools/csv_folder/classification/0/multicenter_partitions/ -o /home/niccolo/ExamodePipeline/Multi_Scale_Tools/model_weights/classification/0/multi_scale_multi_center/combine_probs/
python Fully_supervised_training_combine_probs_multi.py -n 5 -c resnet34 -m 20 10 5 -f True -b 64 -d 0.2 -e 10 -r multi_center -i /home/niccolo/ExamodePipeline/Multi_Scale_Tools/csv_folder/classification/1/multicenter_partitions/ -o /home/niccolo/ExamodePipeline/Multi_Scale_Tools/model_weights/classification/1/multi_scale_multi_center/combine_probs/
python Fully_supervised_training_combine_probs_multi.py -n 5 -c resnet34 -m 20 10 5 -f True -b 64 -d 0.2 -e 10 -r multi_center -i /home/niccolo/ExamodePipeline/Multi_Scale_Tools/csv_folder/classification/2/multicenter_partitions/ -o /home/niccolo/ExamodePipeline/Multi_Scale_Tools/model_weights/classification/2/multi_scale_multi_center/combine_probs/
