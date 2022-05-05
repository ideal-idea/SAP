cd data_gen
python ntu_gendata.py --kind frame_trans
python ntu_gendata.py --kind sequence_trans
python ntu120_gendata.py --kind frame_trans
python ntu120_gendata.py --kind sequence_trans
python kinetics_gendata.py


python gen_bone_data.py --dataset ntu --kind frame_trans
python gen_bone_data.py --dataset ntu --kind sequence_trans
python gen_bone_data.py --dataset ntu120 --kind frame_trans
python gen_bone_data.py --dataset ntu120 --kind sequence_trans
python gen_bone_data.py --dataset kinetics --kind neither
