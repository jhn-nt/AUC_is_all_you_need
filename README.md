# AUC_is_all_you_need
Analyzing different ML model comparison metrics

<p align="center">
  <img src="assets/auc_iayn.jpg" width="500" height="400" alt="AUC IS ALL YOU NEED">
</p>


# Current Status 
- AUROC_AUPRC is Matthew's original examination of beta distributions and their relationship to AUROC and AUPRC
- 1_synthetic_experiments is Giovanni's extension of Matthews work

## Analyses

1. Theory Experiments:
    A. Theory:
        - Clean up expressions for AUROC and AUPRC that involve taking expectations over p_+ of various quantities. 
        - Verify the Au(log)ROC expression and explore if it is useful.

    B. Fairness:
        -  Fairness analyses- if two subgroups, and their p_+ and p_- are each uniform distributions over intersecting ranges, in what settings can we comment on the impact of AUROC over AUPRC w.r.t. fairness implications? 
    
2. Synthetic experiments:
    A. Configurations:
        - Specify model configurations 
            - Specify metrics
                - AUROC
                - AUPRC
                - Brier Score
                - Best-F1
                - Best-Accuracy
                - Best-Precision
                - Best-Recall
                - Best-Sensitivity
                - Best-Specificity
                - Total expected deployment cost under a uniform sampling of independent FP, FN, TP, and TN cost/benefit ratios.
            - Specify subgroup analysis
                - N subgroups
                - Prevalences
                - Positive label prevalences per subgroup

    B. Theoretical analyses
        - For each metric and set up above --> Theoretically computed metrics (independent of sizes of dev and test sets)
    
    C. Empirical analyses: 
        - Goal: Empirically computed metrics with variances over varying test and dev set sizes (dev set only used to pick thresholds for threshold dependent metrics).    
    
    D. Cost analyses
        - For each combination of subgroup prevalences S, positive label prevalences per subgroup of R, test dataset size N, dev dataset size M, and model selection under decision rule Q... 
        - Expected true deployment cost of that metric under cost-benefit ratios V for each subgroup.


# Publication
1. Literature review
    A. Need to do mini-literature review of studies stating auprc is better in imbalanced datasets
2. Dashboard
    A. See Figma sketch [here](https://www.figma.com/file/PXUtoZODXU7b0d4MiixTAA/Untitled?type=design&node-id=0-1&mode=design&t=LCzV3rj1WmjzfRZ1-0)
    B. Basic integration of Matthews code into dashboard done
    C. **To rediscuss**
3. Real-world analyses
    A. Potential analyses
        - Run SubPopBench but integrate the extra metrics above + calculate for subgroups
        - Chexclusion

