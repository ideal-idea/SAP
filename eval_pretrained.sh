## Generate test scores

# NTU 60 XSub
python main.py --config ./config/nturgbd-cross-subject/test_joint.yaml --work-dir pretrain_eval/ntu60/xsub/joint-fusion --weights pretrained-models/ntu60-xsub-joint-fusion.pt
python main.py --config ./config/nturgbd-cross-subject/test_bone.yaml --work-dir pretrain_eval/ntu60/xsub/bone --weights pretrained-models/ntu60-xsub-bone.pt
python main.py --config config/nturgbd-cross-subject/test_joint_S_trans.yaml --work-dir pretrain_eval/ntu60/xsub/joint-seq --weights pretrained-models/ntu60-xsub-joint-seq.pt
python main.py --config config/nturgbd-cross-subject/test_bone_S_trans.yaml --work-dir pretrain_eval/ntu60/xsub/bone-seq --weights pretrained-models/ntu60-xsub-bone-seq.pt
python main.py --config config/nturgbd-cross-subject/test_joint_msg3d_angle_attention_F_trans_nolocal_att_multi_g_scale20.yaml --work-dir pretrain_eval/ntu60/xsub/angle-att --weights pretrained-models/ntu60-xsub-angle-att.pt
python main.py --config config/nturgbd-cross-subject/test_joint_msg3d_angle_attention_S_trans_nolocal_att_multi_g_scale20.yaml --work-dir pretrain_eval/ntu60/xsub/angle-att-seq --weights pretrained-models/ntu60-xsub-angle-att-seq.pt
python main.py --config config/nturgbd-cross-subject/test_joint_msg3d_angle_attention_F_trans_nolocal_multi_g_scale20.yaml --work-dir pretrain_eval/ntu60/xsub/angle --weights pretrained-models/ntu60-xsub-angle.pt
python main.py --config config/nturgbd-cross-subject/test_joint_msg3d_angle_attention_S_trans_nolocal_multi_g_scale20.yaml --work-dir pretrain_eval/ntu60/xsub/angle-seq --weights pretrained-models/ntu60-xsub-angle-seq.pt
#
# NTU 60 XView
python main.py --config ./config/nturgbd-cross-view/test_joint.yaml --work-dir pretrain_eval/ntu60/xview/joint --weights pretrained-models/ntu60-xview-joint.pt
python main.py --config ./config/nturgbd-cross-view/test_bone.yaml --work-dir pretrain_eval/ntu60/xview/bone --weights pretrained-models/ntu60-xview-bone.pt
python main.py --config config/nturgbd-cross-view/test_bone_S_trans.yaml --work-dir pretrain_eval/ntu60/xview/bone-seq --weights pretrained-models/ntu60-xview-bone-seq.pt
python main.py --config config/nturgbd-cross-view/test_joint_S_trans.yaml --work-dir pretrain_eval/ntu60/xview/joint-seq --weights pretrained-models/ntu60-xview-joint-seq.pt
python main.py --config config/nturgbd-cross-view/test_joint_msg3d_angle_attention_F_trans_nolocal_multi_g_scale20.yaml --work-dir pretrain_eval/ntu60/xview/angle --weights pretrained-models/ntu60-xview-angle.pt
python main.py --config config/nturgbd-cross-view/test_joint_msg3d_angle_attention_S_trans_nolocal_multi_g_scale20.yaml --work-dir pretrain_eval/ntu60/xview/angle-seq --weights pretrained-models/ntu60-xview-angle-seq.pt

# NTU 120 XSub
python main.py --config ./config/nturgbd120-cross-subject/test_joint.yaml --work-dir pretrain_eval/ntu120/xsub/joint --weights pretrained-models/ntu120-xsub-joint.pt
python main.py --config ./config/nturgbd120-cross-subject/test_bone.yaml --work-dir pretrain_eval/ntu120/xsub/bone --weights pretrained-models/ntu120-xsub-bone.pt
python main.py --config config/nturgbd120-cross-subject/test_bone_S_trans.yaml --work-dir pretrain_eval/ntu120/xsub/bone-seq --weights pretrained-models/ntu120-xsub-bone-seq.pt
python main.py --config config/nturgbd120-cross-subject/test_joint_S_trans.yaml --work-dir pretrain_eval/ntu120/xsub/joint-seq --weights pretrained-models/ntu120-xsub-joint-seq.pt
python main.py --config config/nturgbd120-cross-subject/test_joint_msg3d_angle_attention_F_trans_nolocal_multi_g_scale20.yaml --work-dir pretrain_eval/ntu120/xsub/angle --weights pretrained-models/ntu120-xsub-angle.pt
python main.py --config config/nturgbd120-cross-subject/test_joint_msg3d_angle_attention_S_trans_nonlocal_multi_g_scale20.yaml --work-dir pretrain_eval/ntu120/xsub/angle-seq --weights pretrained-models/ntu120-xsub-angle-seq.pt

# NTU 120 XSet
python main.py --config ./config/nturgbd120-cross-setup/test_joint.yaml --work-dir pretrain_eval/ntu120/xset/joint --weights pretrained-models/ntu120-xset-joint.pt
python main.py --config ./config/nturgbd120-cross-setup/test_bone.yaml --work-dir pretrain_eval/ntu120/xset/bone --weights pretrained-models/ntu120-xset-bone.pt
python main.py --config config/nturgbd120-cross-setup/test_bone_S_trans.yaml --work-dir pretrain_eval/ntu120/xset/bone-seq --weights pretrained-models/ntu120-xset-bone-seq.pt
python main.py --config config/nturgbd120-cross-setup/test_joint_S_trans.yaml --work-dir pretrain_eval/ntu120/xset/joint-seq --weights pretrained-models/ntu120-xset-joint-seq.pt
python main.py --config config/nturgbd120-cross-setup/test_joint_msg3d_angle_attention_F_trans_nolocal_multi_g_scale20.yaml --work-dir pretrain_eval/ntu120/xset/angle --weights pretrained-models/ntu120-xset-angle.pt
python main.py --config config/nturgbd120-cross-setup/test_joint_msg3d_angle_attention_S_trans_nolocal_multi_g_scale20.yaml --work-dir pretrain_eval/ntu120/xset/angle-seq --weights pretrained-models/ntu120-xset-angle-seq.pt


# Kinetics Skeleton 400
python main.py --config ./config/kinetics-skeleton/test_joint.yaml --work-dir pretrain_eval/kinetics/joint --weights pretrained-models/kinetics-joint.pt
python main.py --config ./config/kinetics-skeleton/test_bone.yaml --work-dir pretrain_eval/kinetics/bone --weights pretrained-models/kinetics-bone.pt
python main.py --config ./config/kinetics-skeleton/test_joint_msg3d_angle_attention_F_trans_nolocal_multi_g_scale20.yaml --work-dir pretrain_eval/kinetics/angle --weights pretrained-models/kinetics-angle.pt



# Perform all ensembles at once

# NTU 60 XSub
printf "\nNTU RGB+D 60 XSub\n"
python ensemble.py --val-path ./data/ntu/xsub/val_label.pkl \
--one pretrain_eval/ntu60/xsub/joint-fusion \
--two  pretrain_eval/ntu60/xsub/bone \
--three  pretrain_eval/ntu60/xsub/angle-att \
--four  pretrain_eval/ntu60/xsub/joint-seq \
--five  pretrain_eval/ntu60/xsub/bone-seq \
--six  pretrain_eval/ntu60/xsub/angle-att-seq
# NTU 60 XView
printf "\nNTU RGB+D 60 XView\n"
python ensemble.py --val-path ./data/ntu/xview/val_label.pkl \
--one pretrain_eval/ntu60/xview/joint \
--two  pretrain_eval/ntu60/xview/bone \
--three  pretrain_eval/ntu60/xview/angle \
--four  pretrain_eval/ntu60/xview/joint-seq \
--five  pretrain_eval/ntu60/xview/bone-seq \
--six  pretrain_eval/ntu60/xview/angle-seq
# NTU 120 XSub
printf "\nNTU RGB+D 120 XSub\n"
python ensemble.py --val-path ./data/ntu120/xsub/val_label.pkl \
--one pretrain_eval/ntu120/xsub/joint \
--two  pretrain_eval/ntu120/xsub/bone \
--three  pretrain_eval/ntu120/xsub/angle \
--four  pretrain_eval/ntu120/xsub/joint-seq \
--five  pretrain_eval/ntu120/xsub/bone-seq \
--six  pretrain_eval/ntu120/xsub/angle-seq
# NTU 120 XSet
printf "\nNTU RGB+D 120 XSet\n"
python ensemble.py --val-path ./data/ntu120/xset/val_label.pkl \
--one pretrain_eval/ntu120/xset/joint \
--two  pretrain_eval/ntu120/xset/bone \
--three  pretrain_eval/ntu120/xset/angle \
--four  pretrain_eval/ntu120/xset/joint-seq \
--five  pretrain_eval/ntu120/xset/bone-seq \
--six  pretrain_eval/ntu120/xset/angle-seq
# Kinetics Skeleton 400
printf "\nKinetics Skeleton 400\n"
python ensemble_kinetics.py --val-path ./data/kinetics/val_label.pkl --one pretrain_eval/kinetics/joint \
--two pretrain_eval/kinetics/bone \
--three pretrain_eval/kinetics/angle