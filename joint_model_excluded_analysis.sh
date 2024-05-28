#!/bin/bash
python joint_model_excluded_analysis.py
Rscript ./joint_model_excluded_analysis_ks.r | grep "data\|p-value" | sed '$!N;s/\n/ /' | sed 's/data:  d_\(.*\) and.*\(D = .*\), \(p-value = .*\)/\1,\2,\3/' | column -t -s,
echo ""
cat io/stats/joint_model_excluded_analysis_kldiv.csv | awk -F, 'BEGIN{print "UID,KLDiv"}{printf "%s,%.3f\n", $1, $2}' | column -t -s,
