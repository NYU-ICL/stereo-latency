#!/bin/bash
echo ""
echo "KLDivergence for Pilot Data"
echo ""
python joint_model_ablation_analysis.py --model_label full
python joint_model_ablation_analysis.py --model_label vergence
python joint_model_ablation_analysis.py --model_label saccade
