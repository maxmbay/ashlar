# ScoringBench results for Ashlar's models

Printed by `analyze.py` from the files in `results/` and ScoringBench's published results (ScoringBenchOutput at commit b590a95), with ScoringBench's code at commit 3bf48c9. Every ScoringBench file used matches its SHA-256 sum in `published_files.json`. The tables and numbers are in the output folder: `numbers.json` and one file per table.

## The numbers in Ashlar's README and docs/dressed.md

### The run

The dressed model has results on 495 folds of 99 tables, with at most 2,400 training rows and 600 test rows in a fold. On 12 tables the target has at most 20 distinct values in the training rows of every fold: 1027_ESL, 1028_SWD, 1029_LEV, 1030_ERA, 294_satellite_image, analcatdata_supreme, CookbookReviews, pol, quake, sensory, Student_Performance, wine_quality.

A fit took a median of 32 minutes per fold (31.7). Fitting and predicting took 448 hours summed over the 495 folds (ScoringBench's `train_time`).

### Mean ranks of the 21 untuned models

The dressed model and the 20 published models trained from scratch and untuned, by CRPS on 99 tables. A mean rank is a model's rank on each table, averaged over the tables; 1 is best.

| Model | Mean rank |
|---|---|
| Dressed model (our run) | 1.33 |
| `condensite_pitbst_v0_1_1` | 4.73 |
| `pytabkit_realmlp_td` | 5.80 |
| `tabm_d` | 5.89 |
| `catboost_quantile` | 7.07 |
| `xgb_vector` | 9.62 |
| `ngboost_gaussian` | 10.07 |
| `flexcode_randomforest` | 10.71 |
| `crepes_catboost_difficulty` | 10.83 |
| `forest_diffusion_flow` | 11.53 |
| `cde_MDN` | 11.54 |
| `xgblss_Gaussian` | 12.24 |
| `xgb_vector_quantile` | 12.78 |
| `crepes_xgb_difficulty` | 12.90 |
| `cde_NF` | 12.95 |
| `flexcode_xgboost` | 13.23 |
| `crepes_catboost_difficulty_mondrian` | 13.65 |
| `nflows_rqs` | 15.11 |
| `crepes_xgb_difficulty_mondrian` | 15.58 |
| `surjectors_maf` | 16.66 |
| `pymc_bart` | 16.81 |

Friedman test: chi-square 855.1 with 20 degrees of freedom, p = 2.8e-168. The Nemenyi critical difference at the 5% level, for 21 models on 99 tables, is 3.15. The dressed model's lead over the next model, `condensite_pitbst_v0_1_1`, is 3.39, and all 20 other models are more than one critical difference behind it.

### The dressed model's place on each table

It has the lowest mean CRPS of the 21 on 82 of the 99 tables. Its place on the other 17:

| Place | Tables |
|---|---|
| 2 | 1201_BNG_breastTumor, 344_mv, 564_fried, abalone, Brazilian_houses, kin8nm, puma32H, sensory, Student_Performance, visualizing_soil |
| 3 | Another-Dataset-on-used-Fiat-500-(1538-rows), MIP-2016-regression, QSAR-TID-10980 |
| 4 | Goodreads-Computer-Books, PRODUCTIVITY |
| 6 | 1000-Cameras-Dataset |
| 7 | nyc-taxi-green-dec-2016 |

### What each part adds

Each row ranks one Ashlar model alone among the same 20 published models. Lead: the mean rank of the best of the 20 minus the row's. CRPS ratio: the geometric mean over the tables of the row's mean CRPS divided by the row above's; the last two columns count the tables where it is lower or higher by more than 1%.

| Results file | Model | Mean rank | Place | Best of the 20 | Its mean rank | Lead | Critical difference | CRPS ratio | Tables lower | Tables higher |
|---|---|---|---|---|---|---|---|---|---|---|
| `one_network_100_epochs` | One network, at most 100 epochs | 4.73 | 2 | `condensite_pitbst_v0_1_1` | 4.44 | -0.28 | 3.15 |  |  |  |
| `one_network` | One network | 3.58 | 1 | `condensite_pitbst_v0_1_1` | 4.51 | 0.93 | 3.15 | 0.934 | 46 | 6 |
| `one_network_one_reader` | One network pooled with one label reader | 1.62 | 1 | `condensite_pitbst_v0_1_1` | 4.67 | 3.05 | 3.15 | 0.903 | 62 | 0 |
| `five_pairs_pooled` | Five cross-fitted members, not dressed | 1.38 | 1 | `condensite_pitbst_v0_1_1` | 4.71 | 3.32 | 3.15 | 0.978 | 57 | 1 |
| `dressed` | Dressed model | 1.33 | 1 | `condensite_pitbst_v0_1_1` | 4.73 | 3.39 | 3.15 | 0.987 | 20 | 0 |

Pooled with one label reader, the network's CRPS is 9.7% lower on the geometric mean. The dressing lowers the pooled members' CRPS by a further 1.3%.

### Since the run

Rescored with ScoringBench's code of 13 November (commit 7f36e0b), the undressed pool's CRPS is identical on 480 of the 480 folds rescored. With the code of 14 November (commit 9406d60) it is identical on 7 of the 480; the largest change on a fold is 0.153% higher (Job_Profitability fold 2) and the largest decrease 0.020%, and by table the ratio of mean CRPS runs from 0.99988 to 1.00123 (Job_Profitability).

Without Brazilian_houses, which ScoringBench has since dropped, the dressed model's mean rank on the other 98 tables is 1.33 and `condensite_pitbst_v0_1_1`'s 4.69, and the critical difference is 3.16.

## What each script printed

### analysis/untuned_crps.py

The dressed model among the 21 untuned models trained from scratch, by mean CRPS per table.

```
99 tables, 21 models | test friedman/nemenyi | dressed 1.3333 | next condensite_pitbst_v0_1_1 4.7273 | gap 3.3939 | CD 3.1477 | Friedman p 2.791e-168
beyond one CD: 20 of 20 | first on 82 tables | tables with any tie 0
by hand: Friedman chi2 855.10 (df 20), p 2.791e-168; Iman-Davenport F 74.50, p 2.913e-223
 rank                               model  meanrank
    1                             dressed  1.333333
    2            condensite_pitbst_v0_1_1  4.727273
    3                 pytabkit_realmlp_td  5.797980
    4                              tabm_d  5.888889
    5                   catboost_quantile  7.070707
    6                          xgb_vector  9.616162
    7                    ngboost_gaussian 10.070707
    8               flexcode_randomforest 10.707071
    9          crepes_catboost_difficulty 10.828283
   10               forest_diffusion_flow 11.525253
   11                             cde_MDN 11.535354
   12                     xgblss_Gaussian 12.242424
   13                 xgb_vector_quantile 12.777778
   14               crepes_xgb_difficulty 12.898990
   15                              cde_NF 12.949495
   16                    flexcode_xgboost 13.232323
   17 crepes_catboost_difficulty_mondrian 13.646465
   18                          nflows_rqs 15.111111
   19      crepes_xgb_difficulty_mondrian 15.575758
   20                      surjectors_maf 16.656566
   21                           pymc_bart 16.808081
```

### analysis/check_loader.py

Check that our loader is ScoringBench's, and that the published files reproduce its published leaderboard.

```
1. 51 scores: our matrix equals ScoringBench's on 51; differs on none
2. 63-model CRPS ranking: 99 tables; largest mean-rank gap to the published leaderboard 0.00e+00; same order True; CD 10.6763 against published 10.6763
3. discrete-target flag: 43 discrete tables; tables where files disagree: none
4. {'dressed': {'rows': 495, 'tables': 99, 'tables_with_5_folds': 99, 'missing_crps': 0, 'same_tables_as_published': True}, 'pool': {'rows': 495, 'tables': 99, 'tables_with_5_folds': 99, 'missing_crps': 0, 'same_tables_as_published': True}}
```

### analysis/families.py

A family for each of the 63 published models, with the evidence for each call.

```
{'fine-tuned': 22, 'untuned': 20, 'pretrained': 19, 'tuned': 2}
untuned by evidence == the list of 20: True | listed but not untuned: [] | untuned but not listed: []
evidence strings found in ScoringBench's code at 3bf48c9 for all 63 models
untuned models with a Hugging Face link: none
```

### analysis/full_field.py

The full field of 64 models (63 published plus the dressed model), by mean CRPS per table.

```
99 tables, 64 models | dressed position 40, mean rank 35.8788 | CD 10.8633 | Friedman p 0.000e+00 | friedman/nemenyi
ahead by family: {'fine-tuned': 22, 'pretrained': 17} | behind by family: {'untuned': 20, 'tuned': 2, 'pretrained': 2}
pretrained or fine-tuned models behind the dressed model: [('crepes_tabiclv2', 43.32), ('crepes_tabiclv2_mondrian', 45.53)]
more than one CD ahead: 28; more than one CD behind: 19; within one CD: 16
  pytabkit_realmlp_hpo_cv_8_new: mean rank 35.9798, position 41
  tabm_hpo_cv_8_tabarena: mean rank 39.1111, position 42
 rank                          model  meanrank     family   side  within_cd_of_dressed
   35                tabpfn_realv2_5 27.565657 pretrained  ahead                  True
   36    finetune_tabpfn_realv2_5_ce 27.848485 fine-tuned  ahead                  True
   37 finetune_tabpfn_realv2_5_is_90 29.252525 fine-tuned  ahead                  True
   38                           nori 30.868687 pretrained  ahead                  True
   39                    tabpfn_v2_6 33.181818 pretrained  ahead                  True
   40                        dressed 35.878788    untuned   self                  True
   41  pytabkit_realmlp_hpo_cv_8_new 35.979798      tuned behind                  True
   42         tabm_hpo_cv_8_tabarena 39.111111      tuned behind                  True
   43                crepes_tabiclv2 43.323232 pretrained behind                  True
   44       condensite_pitbst_v0_1_1 44.676768    untuned behind                  True
   45       crepes_tabiclv2_mondrian 45.525253 pretrained behind                  True
```

### analysis/head_to_head.py

The dressed model head to head, by table and by fold, with the geometric mean CRPS ratio.

```
                           opponent  tables_dressed_lower  tables  geo_ratio_tables  folds_dressed_lower  folds  geo_ratio_folds  folds_dressed_lower_with_an_untuned_model  folds_with_an_untuned_model  sign_p_tables  wilcoxon_p_tables
           condensite_pitbst_v0_1_1                    92      99          0.839273                  442    491         0.839799                                        442                          491   5.074903e-20       1.744721e-14
                pytabkit_realmlp_td                    95      99          0.877470                  453    491         0.882175                                        453                          491   1.238883e-23       4.289305e-15
                             tabm_d                    98      99          0.812516                  472    491         0.814589                                        472                          491   3.155444e-28       2.434346e-17
                  catboost_quantile                    99      99          0.691744                  483    491         0.756753                                        483                          491   3.155444e-30       5.697221e-18
                         xgb_vector                    95      99          0.727330                  469    491         0.725826                                        469                          491   1.238883e-23       1.374554e-17
                   ngboost_gaussian                    97      99          0.718121                  478    491         0.723201                                        478                          491   1.562260e-26       7.271412e-18
              flexcode_randomforest                    92      99          0.725182                  460    491         0.724778                                        460                          491   5.074903e-20       1.206589e-16
         crepes_catboost_difficulty                    99      99          0.685114                  488    491         0.689190                                        488                          491   3.155444e-30       5.697221e-18
              forest_diffusion_flow                    97      99          0.776604                  482    491         0.779119                                        482                          491   1.562260e-26       6.841655e-18
                            cde_MDN                    98      99          0.688210                  489    491         0.693672                                        489                          491   3.155444e-28       5.873890e-18
                    xgblss_Gaussian                    99      99          0.650498                  485    491         0.655250                                        485                          491   3.155444e-30       5.697221e-18
                xgb_vector_quantile                    97      99          0.735472                  476    491         0.739689                                        476                          491   1.562260e-26       6.897567e-17
              crepes_xgb_difficulty                    99      99          0.691775                  489    491         0.695514                                        489                          491   3.155444e-30       5.697221e-18
                             cde_NF                    99      99          0.713917                  482    487         0.724802                                        482                          487   3.155444e-30       5.697221e-18
                   flexcode_xgboost                    96      99          0.667823                  474    491         0.666904                                        474                          491   5.105508e-25       3.186416e-17
crepes_catboost_difficulty_mondrian                    99      99          0.691216                  491    491         0.693518                                        491                          491   3.155444e-30       5.697221e-18
                         nflows_rqs                    99      99          0.647997                  487    491         0.651788                                        487                          491   3.155444e-30       5.697221e-18
     crepes_xgb_difficulty_mondrian                    99      99          0.682295                  491    491         0.684557                                        491                          491   3.155444e-30       5.697221e-18
                     surjectors_maf                    99      99          0.633807                  483    484         0.638329                                        483                          484   3.155444e-30       5.697221e-18
                          pymc_bart                    99      99          0.508383                  491    491         0.509781                                        491                          491   3.155444e-30       5.697221e-18
             tabm_hpo_cv_8_tabarena                    72      99          0.977933                  326    491         0.978146                                        324                          489   6.900699e-06       4.824971e-04
      pytabkit_realmlp_hpo_cv_8_new                    50      99          1.027117                  237    491         1.030040                                        235                          489   1.000000e+00       5.459632e-01
                 kumo_tabular_large                     1      99          1.229772                   17    491         1.231672                                         17                          491   3.155444e-28       7.271412e-18
```

### analysis/by_table.py

The dressed model table by table among the 21 untuned models.

```
rank counts (each model's folds): {1: 82, 2: 10, 3: 3, 4: 2, 6: 1, 7: 1}
rank counts (folds all 21 have): {1: 81, 2: 10, 3: 3, 4: 2, 6: 1, 7: 1} | tables left: 98
tables whose rank changes with common folds: []
ratio to best other untuned: min 0.545 p10 0.849 p25 0.936 median 0.972 p75 0.992 p90 1.045 max 2.467
below 0.9: 14, 0.9-0.99: 57, 0.99-1: 11, above 1: 17, above 1.1: 5
                                     dataset  rank_among_21  rank_among_21_common_folds  ratio_to_best_other                                                                                                                                                                 beaten_by
                        1201_BNG_breastTumor              2                         2.0             1.002764                                                                                                                                          condensite_pitbst_v0_1_1 (0.997)
                                      344_mv              2                         2.0             1.397179                                                                                                                                               pytabkit_realmlp_td (0.716)
                                   564_fried              2                         2.0             1.011334                                                                                                                                               pytabkit_realmlp_td (0.989)
                                     abalone              2                         2.0             1.000324                                                                                                                                                           cde_MDN (1.000)
                            Brazilian_houses              2                         2.0             2.466615                                                                                                                                               pytabkit_realmlp_td (0.405)
                                      kin8nm              2                         2.0             1.062595                                                                                                                                               pytabkit_realmlp_td (0.941)
                                     puma32H              2                         2.0             1.064598                                                                                                                                                            tabm_d (0.939)
                                     sensory              2                         2.0             1.014481                                                                                                                                                  ngboost_gaussian (0.986)
                         Student_Performance              2                         2.0             1.044001                                                                                                                                             flexcode_randomforest (0.958)
                            visualizing_soil              2                         2.0             1.418439                                                                                                                                               xgb_vector_quantile (0.705)
Another-Dataset-on-used-Fiat-500-(1538-rows)              3                         3.0             1.032563                                                                                                           flexcode_randomforest (0.968); condensite_pitbst_v0_1_1 (0.970)
                         MIP-2016-regression              3                         3.0             1.063784                                                                                                                      condensite_pitbst_v0_1_1 (0.940); xgb_vector (0.976)
                              QSAR-TID-10980              3                         3.0             1.045894                                                                                                                         xgb_vector (0.956); flexcode_randomforest (0.963)
                    Goodreads-Computer-Books              4                         4.0             1.044410                                                                                 condensite_pitbst_v0_1_1 (0.957); flexcode_randomforest (0.967); ngboost_gaussian (0.988)
                                PRODUCTIVITY              4                         4.0             1.053261                                                                                 condensite_pitbst_v0_1_1 (0.949); flexcode_randomforest (0.975); flexcode_xgboost (0.993)
                        1000-Cameras-Dataset              6                         6.0             1.129125                              condensite_pitbst_v0_1_1 (0.886); flexcode_randomforest (0.918); flexcode_xgboost (0.923); forest_diffusion_flow (0.926); xgb_vector (0.973)
                     nyc-taxi-green-dec-2016              7                         7.0             1.195491 flexcode_randomforest (0.836); condensite_pitbst_v0_1_1 (0.854); flexcode_xgboost (0.871); xgb_vector_quantile (0.933); xgb_vector (0.949); forest_diffusion_flow (0.988)
```

### analysis/pool.py

The dressed model against its own undressed pool (the five members pooled, from the same fits).

```
by table: geo-mean of table ratios 0.9870, median 0.9998; {'lower_by_more_than_1pct': 20, 'within_1pct': 79, 'higher_by_more_than_1pct': 0, 'higher_at_all': 42, 'lower_at_all': 57, 'equal': 0}
by table, geo-mean of fold ratios per table: overall 0.9866; {'lower_by_more_than_1pct': 20, 'within_1pct': 79, 'higher_by_more_than_1pct': 0, 'higher_at_all': 43, 'lower_at_all': 56, 'equal': 0}
by fold: geo-mean 0.9866, median 0.9997; {'lower_by_more_than_1pct': 100, 'within_1pct': 390, 'higher_by_more_than_1pct': 5, 'higher_at_all': 202, 'lower_at_all': 293, 'equal': 0}; largest 1.0266, smallest 0.6110
folds worse by more than 1%: [['Mortgage', 2, 1.0266], ['1000-Cameras-Dataset', 3, 1.0166], ['MIP-2016-regression', 4, 1.015], ['Buzzinsocialmedia_Twitter', 2, 1.0123], ['Energy_Efficiency', 1, 1.0117]]
Wilcoxon signed-rank on log ratios: tables p 1.80e-04, folds p 4.24e-16
tables most improved: [['344_mv', 0.725], ['294_satellite_image', 0.905], ['tecator', 0.919], ['analcatdata_supreme', 0.922], ['Brazilian_houses', 0.925]]
tables highest ratio: [['QSAR-TID-11', 1.0015], ['QSAR-TID-10980', 1.0016], ['house_prices_nominal', 1.0023], ['Mortgage', 1.0034], ['1000-Cameras-Dataset', 1.0051]]
pool in the dressed model's stead (21 models): mean rank 1.3838, position 1, next condensite_pitbst_v0_1_1 4.7071, CD 3.1477
both (22 models): dressed 1.7576, pool 1.9596, CD 3.3163
pool in the 64-model field: position 41, mean rank 36.6465
```

### analysis/rescore_check.py

Rescoring: the undressed pool's saved predictions, scored again by two newer versions of ScoringBench's code.

```
7f36e0b: 480 folds on 96 tables; matched 480; CRPS identical on 480; largest relative difference 0.00e+00
tables not rescored: ['227_cpu_small', 'Brazilian_houses', 'yprop_4_1']
9406d60: 480 folds matched; CRPS identical on 7, higher on 375, lower on 98; largest increase 0.1529% (Job_Profitability fold 2), largest decrease 0.0199% (analcatdata_supreme fold 4); moved by more than 0.1% on 3 folds and by more than 0.01% on 36
9406d60 by table: ratio of mean CRPS from 0.99988 (analcatdata_supreme) to 1.00123 (Job_Profitability), geometric mean 1.00003; above 1.0005: ['Job_Profitability']
7f36e0b, scores unchanged on every fold: mae, rmse, r2, crps, coverage_20, interval_score_20, coverage_40, interval_score_40, coverage_60, interval_score_60, coverage_80, interval_score_80, coverage_90, interval_score_90, coverage_95, interval_score_95, crts_alpha_1.01, crts_alpha_1.2, crts_alpha_1.5, crts_alpha_2.0, wcrps_left, wcrps_right, wcrps_center, energy_score_beta_0.1, energy_score_beta_0.3, energy_score_beta_0.5, energy_score_beta_0.7, energy_score_beta_0.9, energy_score_beta_1.0, energy_score_beta_1.1, energy_score_beta_1.3, energy_score_beta_1.5, energy_score_beta_1.7, energy_score_beta_1.8, energy_score_beta_1.9, pit_ks_stat, pit_ks_pvalue
7f36e0b, scores changed (folds, largest relative change): sharpness 480, 9.37; dispersion 480, 6.7e+03; cde_loss 270, 0.0441; dpd_beta_0.01 270, 0.0422; dpd_beta_0.2 270, 0.0319; dpd_beta_0.5 270, 0.0328; dpd_beta_1.0 270, 0.0441; pseudospherical_alpha_1.5 270, 0.189; pseudospherical_alpha_2.0 270, 0.305; pseudospherical_alpha_3.0 270, 0.0391
9406d60, scores unchanged on every fold: mae, rmse, r2, coverage_20, interval_score_20, coverage_40, interval_score_40, coverage_60, interval_score_60, coverage_80, interval_score_80, coverage_90, interval_score_90, coverage_95, interval_score_95, pit_ks_stat, pit_ks_pvalue
9406d60, scores changed (folds, largest relative change): crps 473, 0.00153; sharpness 480, 17.6; dispersion 480, 5.84e+03; crts_alpha_1.01 473, 0.75; crts_alpha_1.2 473, 0.183; crts_alpha_1.5 473, 0.0243; crts_alpha_2.0 473, 0.00153; wcrps_left 473, 0.00379; wcrps_right 472, 0.0016; wcrps_center 473, 3.84e-06; energy_score_beta_0.1 473, 2.25e-05; energy_score_beta_0.3 473, 4.95e-05; energy_score_beta_0.5 473, 0.00017; energy_score_beta_0.7 473, 0.000661; energy_score_beta_0.9 473, 0.00132; energy_score_beta_1.0 473, 0.00153; energy_score_beta_1.1 473, 0.00203; energy_score_beta_1.3 473, 0.00385; energy_score_beta_1.5 473, 0.0067; energy_score_beta_1.7 473, 0.0151; energy_score_beta_1.8 473, 0.0228; energy_score_beta_1.9 473, 0.0343; cde_loss 270, 0.0441; dpd_beta_0.01 270, 0.0422; dpd_beta_0.2 270, 0.0319; dpd_beta_0.5 270, 0.0328; dpd_beta_1.0 270, 0.0441; pseudospherical_alpha_1.5 270, 0.189; pseudospherical_alpha_2.0 270, 0.305; pseudospherical_alpha_3.0 270, 0.0391
```

### analysis/robustness.py

The CRPS rankings under other readings of the same data.

```
 1 model(s) lack folds [Brazilian_houses 2; Buzzinsocialmedia_Twitter 2; Infrared_Thermography_Temperature 1; Job_Profitability 1; QSAR-TID-10980 4; QSAR-TID-11 4; california 1; isolet 3 4; yprop_4_1 0 2]: surjectors_maf
 1 model(s) lack folds [Job_Profitability 0 2 3 4; QSAR-TID-10980 4; QSAR-TID-11 4; isolet 3 4]: cde_NF
 2 model(s) lack folds [QSAR-TID-10980 3 4; QSAR-TID-11 3 4]: pytabkit_realmlp_hpo_cv_8_new, tabm_hpo_cv_8_tabarena
58 model(s) lack folds [QSAR-TID-10980 4; QSAR-TID-11 4; isolet 3 4]: 58 models
models with all 495 folds: ['finetune_tabiclv2']
{'field': '21 untuned', 'reading': 'as published', 'tables': 99, 'models': 21, 'folds': None, 'dressed_mean_rank': 1.3333, 'dressed_position': 1, 'next_model': 'condensite_pitbst_v0_1_1', 'next_mean_rank': 4.7273, 'cd': 3.1477, 'friedman_p': 0.0, 'test': 'friedman/nemenyi', 'gap_to_next': 3.3939, 'n_beyond_one_cd': 20, 'tables_first': 82}
{'field': '21 untuned', 'reading': 'common folds', 'tables': 98, 'models': 21, 'folds': 480, 'dressed_mean_rank': 1.3367, 'dressed_position': 1, 'next_model': 'condensite_pitbst_v0_1_1', 'next_mean_rank': 4.7143, 'cd': 3.1638, 'friedman_p': 0.0, 'test': 'friedman/nemenyi', 'gap_to_next': 3.3776, 'n_beyond_one_cd': 20, 'tables_first': 81}
{'field': '21 untuned', 'reading': '98 tables', 'tables': 98, 'models': 21, 'folds': None, 'dressed_mean_rank': 1.3265, 'dressed_position': 1, 'next_model': 'condensite_pitbst_v0_1_1', 'next_mean_rank': 4.6939, 'cd': 3.1638, 'friedman_p': 0.0, 'test': 'friedman/nemenyi', 'gap_to_next': 3.3673, 'n_beyond_one_cd': 20, 'tables_first': 82}
{'field': '21 untuned', 'reading': 'both', 'tables': 97, 'models': 21, 'folds': 476, 'dressed_mean_rank': 1.3299, 'dressed_position': 1, 'next_model': 'condensite_pitbst_v0_1_1', 'next_mean_rank': 4.6804, 'cd': 3.18, 'friedman_p': 0.0, 'test': 'friedman/nemenyi', 'gap_to_next': 3.3505, 'n_beyond_one_cd': 20, 'tables_first': 81}
{'field': '64 field', 'reading': 'as published', 'tables': 99, 'models': 64, 'folds': None, 'dressed_mean_rank': 35.8788, 'dressed_position': 40, 'next_model': 'kumo_tabular_large', 'next_mean_rank': 1.8788, 'cd': 10.8633, 'friedman_p': 0.0, 'test': 'friedman/nemenyi', 'all_ahead_pretrained_or_finetuned': True, 'realmlp_tuned_mean_rank': 35.9798, 'tabm_tuned_mean_rank': 39.1111}
{'field': '64 field', 'reading': 'common folds', 'tables': 98, 'models': 64, 'folds': 478, 'dressed_mean_rank': 36.0918, 'dressed_position': 41, 'next_model': 'kumo_tabular_large', 'next_mean_rank': 1.9184, 'cd': 10.9186, 'friedman_p': 0.0, 'test': 'friedman/nemenyi', 'all_ahead_pretrained_or_finetuned': False, 'realmlp_tuned_mean_rank': 35.949, 'tabm_tuned_mean_rank': 39.4286}
{'field': '64 field', 'reading': '98 tables', 'tables': 98, 'models': 64, 'folds': None, 'dressed_mean_rank': 35.7959, 'dressed_position': 40, 'next_model': 'kumo_tabular_large', 'next_mean_rank': 1.8878, 'cd': 10.9186, 'friedman_p': 0.0, 'test': 'friedman/nemenyi', 'all_ahead_pretrained_or_finetuned': True, 'realmlp_tuned_mean_rank': 36.2755, 'tabm_tuned_mean_rank': 39.0816}
{'field': '64 field', 'reading': 'both', 'tables': 97, 'models': 64, 'folds': 474, 'dressed_mean_rank': 36.0103, 'dressed_position': 40, 'next_model': 'kumo_tabular_large', 'next_mean_rank': 1.9278, 'cd': 10.9747, 'friedman_p': 0.0, 'test': 'friedman/nemenyi', 'all_ahead_pretrained_or_finetuned': True, 'realmlp_tuned_mean_rank': 36.2887, 'tabm_tuned_mean_rank': 39.3918}
{'field': '21 untuned', 'reading': 'dressed cut to folds the untuned field has', 'tables': 99, 'models': 21, 'folds': 491, 'dressed_mean_rank': 1.3333, 'dressed_position': 1, 'next_model': 'condensite_pitbst_v0_1_1', 'next_mean_rank': 4.7273, 'cd': 3.1477, 'friedman_p': 0.0, 'test': 'friedman/nemenyi', 'gap_to_next': 3.3939, 'n_beyond_one_cd': 20, 'tables_first': 82}
```

### analysis/all_scores.py

Every score in the files, for the 21 untuned models, by ScoringBench's rules.

```
                    score  tables  models  dressed_mean_rank  dressed_position                 best_other  best_other_mean_rank             test       cd models_dropped
                     crps      99      21           1.333333                 1   condensite_pitbst_v0_1_1              4.727273 friedman/nemenyi 3.147741               
               wcrps_left      99      21           1.303030                 1   condensite_pitbst_v0_1_1              4.787879 friedman/nemenyi 3.147741               
             wcrps_center      99      21           1.424242                 1   condensite_pitbst_v0_1_1              4.737374 friedman/nemenyi 3.147741               
              wcrps_right      99      21           1.303030                 1   condensite_pitbst_v0_1_1              4.777778 friedman/nemenyi 3.147741               
          crts_alpha_1.01      99      21           1.636364                 1   condensite_pitbst_v0_1_1              2.727273 friedman/nemenyi 3.147741               
           crts_alpha_1.2      99      21           1.272727                 1   condensite_pitbst_v0_1_1              4.101010 friedman/nemenyi 3.147741               
           crts_alpha_1.5      99      21           1.313131                 1   condensite_pitbst_v0_1_1              4.545455 friedman/nemenyi 3.147741               
           crts_alpha_2.0      99      21           1.333333                 1   condensite_pitbst_v0_1_1              4.727273 friedman/nemenyi 3.147741               
        interval_score_20      99      21           1.606061                 1   condensite_pitbst_v0_1_1              4.939394 friedman/nemenyi 3.147741               
        interval_score_40      99      21           1.444444                 1   condensite_pitbst_v0_1_1              4.888889 friedman/nemenyi 3.147741               
        interval_score_60      99      21           1.323232                 1   condensite_pitbst_v0_1_1              5.232323 friedman/nemenyi 3.147741               
        interval_score_80      99      21           1.222222                 1        pytabkit_realmlp_td              5.111111 friedman/nemenyi 3.147741               
        interval_score_90      99      21           1.262626                 1        pytabkit_realmlp_td              5.020202 friedman/nemenyi 3.147741               
        interval_score_95      99      21           1.323232                 1        pytabkit_realmlp_td              4.909091 friedman/nemenyi 3.147741               
    energy_score_beta_0.1      99      21           1.373737                 1   condensite_pitbst_v0_1_1              4.151515 friedman/nemenyi 3.147741               
    energy_score_beta_0.3      99      21           1.282828                 1   condensite_pitbst_v0_1_1              4.313131 friedman/nemenyi 3.147741               
    energy_score_beta_0.5      99      21           1.242424                 1   condensite_pitbst_v0_1_1              4.363636 friedman/nemenyi 3.147741               
    energy_score_beta_0.7      99      21           1.262626                 1   condensite_pitbst_v0_1_1              4.424242 friedman/nemenyi 3.147741               
    energy_score_beta_0.9      99      21           1.313131                 1   condensite_pitbst_v0_1_1              4.646465 friedman/nemenyi 3.147741               
    energy_score_beta_1.0      99      21           1.333333                 1   condensite_pitbst_v0_1_1              4.727273 friedman/nemenyi 3.147741               
    energy_score_beta_1.1      99      21           1.353535                 1   condensite_pitbst_v0_1_1              4.828283 friedman/nemenyi 3.147741               
    energy_score_beta_1.3      99      21           1.424242                 1   condensite_pitbst_v0_1_1              5.030303 friedman/nemenyi 3.147741               
    energy_score_beta_1.5      99      21           1.525253                 1   condensite_pitbst_v0_1_1              5.393939 friedman/nemenyi 3.147741               
    energy_score_beta_1.7      99      21           1.717172                 1   condensite_pitbst_v0_1_1              5.626263 friedman/nemenyi 3.147741               
    energy_score_beta_1.8      99      21           1.828283                 1   condensite_pitbst_v0_1_1              5.787879 friedman/nemenyi 3.147741               
    energy_score_beta_1.9      99      21           1.979798                 1          catboost_quantile              5.656566 friedman/nemenyi 3.147741               
              coverage_20      99      21           8.010101                 6 crepes_catboost_difficulty              5.419192 friedman/nemenyi 3.147741               
              coverage_40      99      21           7.429293                 5 crepes_catboost_difficulty              4.560606 friedman/nemenyi 3.147741               
              coverage_60      99      21           7.272727                 5 crepes_catboost_difficulty              3.469697 friedman/nemenyi 3.147741               
              coverage_80      99      21           6.873737                 3      crepes_xgb_difficulty              2.994949 friedman/nemenyi 3.147741               
              coverage_90      99      21           6.111111                 3      crepes_xgb_difficulty              3.085859 friedman/nemenyi 3.147741               
              coverage_95      99      21           5.924242                 3      crepes_xgb_difficulty              3.237374 friedman/nemenyi 3.147741               
                     amce      99      21           6.333333                 3 crepes_catboost_difficulty              2.747475 friedman/nemenyi 3.147741               
              pit_ks_stat      56      21           3.125000                 1 crepes_catboost_difficulty              4.857143 friedman/nemenyi 4.185262               
            pit_ks_pvalue      56      21           3.589286                 1 crepes_catboost_difficulty              4.696429 friedman/nemenyi 4.185262               
                 cde_loss      56      21           1.553571                 1   condensite_pitbst_v0_1_1              3.160714 friedman/nemenyi 4.185262               
            dpd_beta_0.01      56      21           1.464286                 1   condensite_pitbst_v0_1_1              2.375000 friedman/nemenyi 4.185262               
             dpd_beta_0.2      56      21           1.196429                 1   condensite_pitbst_v0_1_1              3.071429   anova/tukeyhsd      NaN               
             dpd_beta_0.5      56      21           1.285714                 1   condensite_pitbst_v0_1_1              3.285714 friedman/nemenyi 4.185262               
             dpd_beta_1.0      56      21           1.553571                 1   condensite_pitbst_v0_1_1              3.160714 friedman/nemenyi 4.185262               
pseudospherical_alpha_1.5      56      21           1.267857                 1   condensite_pitbst_v0_1_1              3.464286 friedman/nemenyi 4.185262               
pseudospherical_alpha_2.0      56      21           1.375000                 1   condensite_pitbst_v0_1_1              3.571429 friedman/nemenyi 4.185262               
pseudospherical_alpha_3.0      56      21           1.482143                 1   condensite_pitbst_v0_1_1              3.696429 friedman/nemenyi 4.185262               
                      mae      99      21           1.747475                 1   condensite_pitbst_v0_1_1              5.101010 friedman/nemenyi 3.147741               
                     rmse      99      21           2.141414                 1          catboost_quantile              5.454545 friedman/nemenyi 3.147741               
                       r2      99      21           2.292929                 1          catboost_quantile              5.353535 friedman/nemenyi 3.147741               
                sharpness      99      21          12.373737                12      forest_diffusion_flow              2.141414 friedman/nemenyi 3.147741               
               dispersion      99      21          13.686869                16           flexcode_xgboost              2.717172 friedman/nemenyi 3.147741               
                 fit_time      99      21          20.979798                21      crepes_xgb_difficulty              1.141414 friedman/nemenyi 3.147741               
             predict_time      99      21          20.909091                21                 xgb_vector              2.434343 friedman/nemenyi 3.147741               
               train_time      99      21          20.989899                21      crepes_xgb_difficulty              1.121212 friedman/nemenyi 3.147741               

not first: [['coverage_20', 6, 'crepes_catboost_difficulty 5.42; crepes_xgb_difficulty 5.43; crepes_xgb_difficulty_mondrian 6.42; crepes_catboost_difficulty_mondrian 6.53; ngboost_gaussian 7.55'], ['coverage_40', 5, 'crepes_catboost_difficulty 4.56; crepes_xgb_difficulty 4.59; crepes_catboost_difficulty_mondrian 5.64; crepes_xgb_difficulty_mondrian 5.83'], ['coverage_60', 5, 'crepes_catboost_difficulty 3.47; crepes_xgb_difficulty 3.72; crepes_xgb_difficulty_mondrian 5.76; crepes_catboost_difficulty_mondrian 5.86'], ['coverage_80', 3, 'crepes_xgb_difficulty 2.99; crepes_catboost_difficulty 3.08'], ['coverage_90', 3, 'crepes_xgb_difficulty 3.09; crepes_catboost_difficulty 3.21'], ['coverage_95', 3, 'crepes_xgb_difficulty 3.24; crepes_catboost_difficulty 3.39'], ['amce', 3, 'crepes_catboost_difficulty 2.75; crepes_xgb_difficulty 2.77'], ['sharpness', 12, 'forest_diffusion_flow 2.14; surjectors_maf 2.79; catboost_quantile 5.02; xgb_vector_quantile 5.04; nflows_rqs 5.73; cde_NF 6.87; xgblss_Gaussian 7.43; ngboost_gaussian 8.64; pytabkit_realmlp_td 8.75; tabm_d 9.14; crepes_catboost_difficulty_mondrian 11.58'], ['dispersion', 16, 'flexcode_xgboost 2.72; flexcode_randomforest 3.13; xgb_vector 5.34; condensite_pitbst_v0_1_1 7.19; crepes_xgb_difficulty_mondrian 8.23; crepes_xgb_difficulty 8.40; cde_MDN 9.16; crepes_catboost_difficulty 9.32; crepes_catboost_difficulty_mondrian 9.65; cde_NF 10.33; nflows_rqs 11.19; xgblss_Gaussian 11.59; xgb_vector_quantile 12.34; surjectors_maf 12.38; catboost_quantile 13.47'], ['fit_time', 21, 'crepes_xgb_difficulty 1.14; crepes_xgb_difficulty_mondrian 2.16; xgblss_Gaussian 3.20; crepes_catboost_difficulty 3.98; crepes_catboost_difficulty_mondrian 4.75; tabm_d 7.13; flexcode_xgboost 7.19; xgb_vector_quantile 7.82; xgb_vector 8.76; cde_MDN 10.72; pytabkit_realmlp_td 10.79; ngboost_gaussian 11.07; surjectors_maf 12.86; cde_NF 14.51; pymc_bart 14.70; condensite_pitbst_v0_1_1 16.52; forest_diffusion_flow 17.10; flexcode_randomforest 18.14; nflows_rqs 18.26; catboost_quantile 19.23'], ['predict_time', 21, 'xgb_vector 2.43; xgb_vector_quantile 2.43; tabm_d 3.83; cde_NF 4.69; cde_MDN 4.83; catboost_quantile 5.81; crepes_catboost_difficulty 6.38; crepes_xgb_difficulty 7.36; crepes_catboost_difficulty_mondrian 8.69; crepes_xgb_difficulty_mondrian 9.84; xgblss_Gaussian 10.07; ngboost_gaussian 12.76; pytabkit_realmlp_td 13.01; flexcode_xgboost 13.16; nflows_rqs 14.71; surjectors_maf 16.33; flexcode_randomforest 16.67; pymc_bart 18.27; forest_diffusion_flow 18.73; condensite_pitbst_v0_1_1 20.09'], ['train_time', 21, 'crepes_xgb_difficulty 1.12; crepes_xgb_difficulty_mondrian 2.18; xgblss_Gaussian 3.24; crepes_catboost_difficulty 3.93; crepes_catboost_difficulty_mondrian 4.75; tabm_d 7.09; flexcode_xgboost 7.25; xgb_vector_quantile 7.80; xgb_vector 8.73; cde_MDN 10.62; pytabkit_realmlp_td 10.90; ngboost_gaussian 11.05; surjectors_maf 12.88; cde_NF 14.36; pymc_bart 14.75; forest_diffusion_flow 16.70; nflows_rqs 17.43; flexcode_randomforest 17.86; condensite_pitbst_v0_1_1 18.39; catboost_quantile 18.98']]
```

### analysis/calibration.py

Calibration. Empirical coverage of the central 20 to 95% intervals against nominal, and the PIT test.

```
nominal                                                    0.20   0.40   0.60   0.80   0.90   0.95
tables                        model                                                               
all                           condensite_pitbst_v0_1_1    0.302  0.539  0.734  0.889  0.952  0.978
                              crepes_catboost_difficulty  0.199  0.399  0.598  0.799  0.897  0.945
                              dressed                     0.285  0.487  0.675  0.852  0.933  0.970
                              pytabkit_realmlp_td         0.209  0.399  0.586  0.773  0.868  0.924
                              tabm_d                      0.240  0.439  0.614  0.788  0.878  0.935
at most 20 values             condensite_pitbst_v0_1_1    0.346  0.568  0.722  0.852  0.922  0.960
                              crepes_catboost_difficulty  0.203  0.391  0.588  0.787  0.895  0.944
                              dressed                     0.706  0.800  0.882  0.952  0.980  0.991
                              pytabkit_realmlp_td         0.285  0.457  0.606  0.748  0.817  0.858
                              tabm_d                      0.331  0.515  0.655  0.815  0.888  0.935
continuous                    condensite_pitbst_v0_1_1    0.294  0.539  0.742  0.898  0.958  0.981
                              crepes_catboost_difficulty  0.199  0.401  0.600  0.800  0.898  0.945
                              dressed                     0.227  0.444  0.646  0.839  0.925  0.966
                              pytabkit_realmlp_td         0.195  0.385  0.575  0.770  0.870  0.930
                              tabm_d                      0.230  0.433  0.614  0.788  0.879  0.935
discrete                      condensite_pitbst_v0_1_1    0.314  0.539  0.724  0.877  0.944  0.974
                              crepes_catboost_difficulty  0.198  0.395  0.596  0.796  0.896  0.945
                              dressed                     0.361  0.543  0.712  0.870  0.942  0.976
                              pytabkit_realmlp_td         0.228  0.418  0.601  0.777  0.865  0.915
                              tabm_d                      0.254  0.446  0.613  0.788  0.878  0.934
discrete, more than 20 values condensite_pitbst_v0_1_1    0.302  0.528  0.724  0.887  0.953  0.980
                              crepes_catboost_difficulty  0.197  0.396  0.598  0.800  0.896  0.945
                              dressed                     0.227  0.444  0.646  0.838  0.928  0.970
                              pytabkit_realmlp_td         0.206  0.403  0.599  0.789  0.884  0.938
                              tabm_d                      0.224  0.419  0.597  0.778  0.874  0.933
more than 20 values           condensite_pitbst_v0_1_1    0.296  0.535  0.735  0.894  0.956  0.981
                              crepes_catboost_difficulty  0.198  0.399  0.600  0.800  0.897  0.945
                              dressed                     0.227  0.444  0.646  0.838  0.926  0.967
                              pytabkit_realmlp_td         0.199  0.391  0.584  0.777  0.875  0.933
                              tabm_d                      0.228  0.428  0.608  0.784  0.877  0.935

mean absolute coverage error, averaged over the six levels:
 tables                         all  at most 20 values  continuous  discrete  discrete, more than 20 values  more than 20 values
model                                                                                                                          
condensite_pitbst_v0_1_1    0.0939             0.1115      0.0937    0.0942                         0.0876               0.0915
crepes_catboost_difficulty  0.0102             0.0136      0.0099    0.0105                         0.0093               0.0097
dressed                     0.0606             0.2434      0.0342    0.0949                         0.0375               0.0354
pytabkit_realmlp_td         0.0478             0.1118      0.0386    0.0597                         0.0395               0.0390
tabm_d                      0.0643             0.0905      0.0633    0.0656                         0.0559               0.0607

PIT KS test on continuous tables:
                               model  folds  tables  mean_ks_stat_over_tables  median_ks_stat_folds  share_folds_p_below_0_05  share_folds_p_below_0_01
                            dressed    280      56                    0.0626                0.0509                    0.2643                    0.1464
         crepes_catboost_difficulty    278      56                    0.0645                0.0550                    0.3309                    0.1763
              crepes_xgb_difficulty    278      56                    0.0669                0.0604                    0.4029                    0.2158
                pytabkit_realmlp_td    278      56                    0.0838                0.0708                    0.5360                    0.3777
                   ngboost_gaussian    278      56                    0.0843                0.0703                    0.5647                    0.4101
crepes_catboost_difficulty_mondrian    278      56                    0.0869                0.0683                    0.6691                    0.4640
     crepes_xgb_difficulty_mondrian    278      56                    0.0891                0.0667                    0.7374                    0.4568
                         xgb_vector    278      56                    0.0892                0.0686                    0.5540                    0.4065
                             tabm_d    278      56                    0.0912                0.0680                    0.5072                    0.3849
                            cde_MDN    278      56                    0.0948                0.0702                    0.5252                    0.3849
           condensite_pitbst_v0_1_1    278      56                    0.1052                0.0808                    0.7878                    0.6403
                          pymc_bart    278      56                    0.1226                0.1040                    0.8273                    0.6511
              flexcode_randomforest    278      56                    0.1270                0.1040                    0.7986                    0.6835
                   flexcode_xgboost    278      56                    0.1345                0.1214                    0.8525                    0.7950
                  catboost_quantile    278      56                    0.1414                0.1319                    0.9784                    0.9532
                    xgblss_Gaussian    278      56                    0.1447                0.1064                    0.9065                    0.7698
                             cde_NF    278      56                    0.1681                0.1422                    0.8561                    0.7770
                xgb_vector_quantile    278      56                    0.1834                0.1759                    1.0000                    0.9892
              forest_diffusion_flow    278      56                    0.2369                0.2399                    0.9604                    0.9568
                     surjectors_maf    274      56                    0.2461                0.2495                    0.9745                    0.9489
                         nflows_rqs    278      56                    0.2519                0.2564                    0.9748                    0.9460
```

### analysis/timing.py

How long the dressed model took to fit, fold by fold, by table size.

```
against results/target_values.csv: {'folds_in_both': 495, 'n_test_agree': 495, 'n_train_agree': 495}
67 tables capped at 2,400 training rows; the rest have 240 to 2394 rows; training rows per fold 192 to 2400
slowest tables by median fold fit time: {'QSAR-TID-11': 332.8, 'QSAR-TID-10980': 324.4, 'isolet': 253.0} columns: {'QSAR-TID-11': 1024, 'QSAR-TID-10980': 1024, 'isolet': 613}
folds over twice their table's other folds: [['house_16H', 0, 245.86, 3.24], ['Student_Performance', 0, 14.91, 2.37], ['IEEE80211aa-GATS', 4, 266.12, 2.16], ['IEEE80211aa-GATS', 2, 256.48, 2.08]]
     size_band  folds  tables  median_fit_minutes  min_fit_minutes  max_fit_minutes
     under 500     30       6                 7.7              4.1             14.9
    500 to 999     90      18                16.0              3.8             41.6
1,000 to 1,999     40       8                20.2              8.8             85.2
   2,400 (cut)    335      67                39.3              9.1            615.9
fit time per fold: median 31.7 min, from 3.8 to 615.9 min; total 378 worker-hours; log-log slope against training rows 0.92
predict time per fold (includes saving the undressed pool's predictions): median 360 s, max 175.5 min
                                model     family  folds  median_fit_seconds  median_train_seconds  max_train_seconds  total_train_hours
                crepes_xgb_difficulty    untuned    491                 0.1                   0.2               10.9                0.0
       crepes_xgb_difficulty_mondrian    untuned    491                 0.2                   0.2               11.1                0.1
                           tabdptv1_3 pretrained    491                 0.0                   0.3               30.1                0.5
                      xgblss_Gaussian    untuned    491                 0.3                   0.3                5.0                0.1
           crepes_catboost_difficulty    untuned    491                 0.4                   0.4               11.4                0.1
                   kumo_tabular_small pretrained    491                 0.3                   0.5                8.8                0.1
  crepes_catboost_difficulty_mondrian    untuned    491                 0.5                   0.5               11.7                0.1
                           causilo_v1 pretrained    491                 0.3                   0.6               25.2                0.3
                      tabpfn_realv2_5 pretrained    491                 0.3                   0.7               10.6                0.2
                             tabiclv2 pretrained    491                 0.4                   0.7               17.4                0.2
                  kumo_tabular_medium pretrained    491                 0.5                   0.8                8.4                0.1
                      crepes_tabiclv2 pretrained    491                 0.6                   0.8               20.4                0.2
             crepes_tabiclv2_mondrian pretrained    491                 0.7                   0.9               21.4                0.2
                     tabpfn_v3_5_fast pretrained    491                 0.6                   0.9               16.6                0.2
                          tabpfn_v2_6 pretrained    491                 0.4                   1.3               16.1                0.3
                            tabpfn_v3 pretrained    491                 0.7                   1.3               11.4                0.2
                        exaonetabular pretrained    491                 0.0                   1.5               87.0                0.6
                   kumo_tabular_large pretrained    491                 0.9                   1.5               17.3                0.3
                                 nori pretrained    491                 0.0                   2.0               40.5                0.5
                     flexcode_xgboost    untuned    491                 2.0                   2.1              121.3                0.6
                               tabm_d    untuned    491                 2.2                   2.2               17.3                0.4
                          tabpfn_v3_5 pretrained    491                 1.9                   2.7               54.8                0.5
                             nori_30m pretrained    491                 0.0                   3.1               38.2                0.7
                  xgb_vector_quantile    untuned    491                 3.2                   3.2               28.1                0.5
                            tabldm_v1 pretrained    491                 2.1                   3.4               56.0                0.6
                           xgb_vector    untuned    491                 3.7                   3.7               24.7                0.6
                     ngboost_gaussian    untuned    491                 6.6                   6.6              346.5                1.8
                  pytabkit_realmlp_td    untuned    491                10.0                  10.1               19.7                1.2
                              cde_MDN    untuned    491                10.1                  10.1               20.6                1.2
                       surjectors_maf    untuned    484                12.2                  12.5               51.6                1.8
          finetune_tabpfn_realv2_5_ce fine-tuned    491                13.3                  13.6              272.5                3.9
                               limix2 pretrained    491                 3.5                  14.3              302.6                3.4
                    finetune_tabiclv2 fine-tuned    495                14.1                  14.4              633.1                4.3
         finetune_tabpfn_realv2_5_cde fine-tuned    491                18.4                  18.7              398.1                5.1
                            pymc_bart    untuned    491                18.5                  20.3               71.8                2.9
    finetune_tabpfn_realv2_5_beta_0.1 fine-tuned    491                21.8                  22.2              497.4                5.3
  finetune_tabpfn_realv2_5_wCRPS_left fine-tuned    491                22.2                  22.5              863.8                5.7
finetune_tabpfn_realv2_5_wCRPS_center fine-tuned    491                23.2                  23.6              564.4                5.7
    finetune_tabpfn_realv2_5_beta_0.3 fine-tuned    491                24.1                  24.6              645.9                5.6
    finetune_tabpfn_realv2_5_beta_0.5 fine-tuned    491                24.6                  25.0              497.7                5.9
 finetune_tabpfn_realv2_5_wCRPS_right fine-tuned    491                24.8                  25.1              650.2                6.3
    finetune_tabpfn_realv2_5_beta_1.9 fine-tuned    491                24.9                  25.2              859.3                6.4
                               cde_NF    untuned    487                25.2                  25.3               43.2                2.9
    finetune_tabpfn_realv2_5_beta_1.5 fine-tuned    491                24.9                  25.3              589.9                6.2
    finetune_tabpfn_realv2_5_beta_1.7 fine-tuned    491                25.0                  25.3              826.6                6.3
       finetune_tabpfn_realv2_5_is_90 fine-tuned    491                25.3                  25.6              626.9                6.4
    finetune_tabpfn_realv2_5_beta_1.8 fine-tuned    491                25.6                  25.9              864.9                6.1
    finetune_tabpfn_realv2_5_beta_0.7 fine-tuned    491                25.6                  26.2              682.5                6.2
    finetune_tabpfn_realv2_5_beta_1.1 fine-tuned    491                25.6                  26.3              859.3                6.3
    finetune_tabpfn_realv2_5_beta_1.3 fine-tuned    491                25.8                  26.3              857.3                6.4
    finetune_tabpfn_realv2_5_beta_0.9 fine-tuned    491                25.8                  26.3              553.6                6.1
        finetune_tabpfn_realv2_5_crps fine-tuned    491                26.6                  27.0              629.4                6.2
        finetune_tabpfn_realv2_5_crls fine-tuned    491                27.5                  28.1              798.3                7.3
         finetune_tabpfn_realv2_5_mse fine-tuned    491                36.2                  36.8             1471.7               11.1
                forest_diffusion_flow    untuned    491                41.5                  44.5             2293.4               13.3
                           nflows_rqs    untuned    491                69.0                  69.3              135.8                8.0
             condensite_pitbst_v0_1_1    untuned    491                36.7                  75.7              501.4               10.5
                flexcode_randomforest    untuned    491                76.7                  77.5             5439.9               29.1
                    catboost_quantile    untuned    491                99.1                  99.2             6787.7               38.7
                               mitra2 pretrained    491                 0.0                 142.7              407.4               17.8
             finetune_mitra2_steps_50 fine-tuned    491                 0.0                 214.3             3544.9               66.4
        pytabkit_realmlp_hpo_cv_8_new      tuned    491              1707.8                1709.1            21114.9              258.4
                              dressed    untuned    495              1901.9                2261.2            37634.4              448.4
               tabm_hpo_cv_8_tabarena      tuned    491              3151.6                3151.7            23679.1              501.2
```

### analysis/devset_tests.py

The tests on the 23 development tables, as docs/dressed.md reports them under "How it was tested".

```
one_network                        geo 0.569 median 0.910 (upper 0.916) better 37 within 7 worse 2 (worse >5%: 0)
five_networks_averaged             geo 0.568 median 0.914 (upper 0.916) better 37 within 7 worse 2 (worse >5%: 0)
one_network_one_reader_pooled      geo 0.920 median 0.955 (upper 0.956) better 38 within 6 worse 2 (worse >5%: 1)
five_members_pooled_not_dressed    geo 0.955 median 0.974 (upper 0.974) better 27 within 17 worse 2 (worse >5%: 2)
  guide's file one_network            geo 0.569 median 0.910 37/7/2
  guide's file five_members_pooled_not_dressed geo 0.955 median 0.974 27/17/2
log score: {'above': 39, 'n': 46, 'mean_nats': 0.6491700046885698, 'median_nats': 0.15835569809232358, 'upper_median_nats': 0.17869128188082373}
most extreme dressed/network by table: {'wave_energy': 0.003, 'naval_propulsion_plant': 0.068, '557_analcatdata_apnea1': 0.311, 'fps_benchmark': 0.427, 'cars_rated7': 0.547}
geo-mean without those three tables: {'one_network': 0.843, 'five_networks_averaged': 0.848}
pass line: {'geo_vs_pool': 0.9550908927938657, 'geo_vs_pool_needs': '<= 1.005', 'geo_vs_smoothed_crps_pool': 0.9159780150506257, 'geo_vs_smoothed_crps_pool_needs': '<= 0.99', 'worse_gt_1pct': 2, 'worse_needs': '<= 4', 'log_above': 39, 'log_above_needs': '>= 24', 'log_mean': 0.6491700046885698, 'passes': True}
looser line: {'geo_vs_pool': 0.9550908927938657, 'needs': '<= 0.97', 'worse_gt_1pct': 2, 'worse_needs': '<= 8', 'worse_gt_5pct': 2, 'worse_gt_5pct_needs': '<= 1', 'passes': False}
fits worse than the pool by more than 5%: [{'table': 'wave_energy', 'fold': 2, 'ratio': 5.5015, 'gate': 0.7, 'rows': 2400}, {'table': '557_analcatdata_apnea1_rated6', 'fold': 2, 'ratio': 1.1007, 'gate': 0.65, 'rows': 380}]
wave_energy: {'fold2_dressed_over_pool': 5.501, 'fold0_dressed_over_pool': 0.894, 'fold1_dressed_over_pool': 0.939, 'fold2_network_over_dressed': 196.102}
557_analcatdata_apnea1_rated6 fold 2: {'dressed_over_pool': 1.100667376397895, 'gate': 0.65, 'rows': 380, 'gate_rows_one_fifth': 76, 'stop_rows': 76}
geo-mean against the pool without wave_energy fold 2: 0.919
fold 0 dressed/pool: {'n': 23, 'geo_mean': 0.9133261474354568, 'median': 0.980917180916384, 'upper_median': 0.980917180916384, 'better_gt_1pct': 13, 'within_1pct': 9, 'worse_gt_1pct': 1, 'worse_gt_5pct': 0}
library against research code: {'port_dressed': {'median': 0.9999181454630378, 'geo_mean': 0.9997085620018437, 'largest_gap': 0.043695918991496496, 'largest_gap_table': 'fps_benchmark'}, 'port_pool': {'median': 1.000315620553928, 'geo_mean': 0.9998470113512026, 'largest_gap': 0.02148820685584396, 'largest_gap_table': '557_analcatdata_apnea1_rated6'}}
library on its own fits: {'n': 23, 'geo_mean': 0.9131996786763072, 'median': 0.9803685487014021, 'upper_median': 0.9803685487014021, 'better_gt_1pct': 13, 'within_1pct': 9, 'worse_gt_1pct': 1, 'worse_gt_5pct': 0} {'above': 22, 'n': 23, 'mean_nats': 0.6846911895429526}
network on the reader vs one network and one reader: {'n': 23, 'geo_mean': 1.1904132121552735, 'median': 1.0302516768520584, 'upper_median': 1.0302516768520584, 'better_gt_1pct': 6, 'within_1pct': 2, 'worse_gt_1pct': 15, 'worse_gt_5pct': 10} | vs five pooled: {'n': 23, 'geo_mean': 1.2248989481195434, 'median': 1.0804357155406816, 'upper_median': 1.0804357155406816, 'better_gt_1pct': 3, 'within_1pct': 3, 'worse_gt_1pct': 17, 'worse_gt_5pct': 12}
smoothed_by_log_score vs pool, fold 0: {'n': 23, 'geo_mean': 1.0203721648437523, 'median': 1.0035336598234743, 'upper_median': 1.0035336598234743, 'better_gt_1pct': 2, 'within_1pct': 14, 'worse_gt_1pct': 7, 'worse_gt_5pct': 3}
smoothed_by_crps vs pool, fold 0: {'n': 23, 'geo_mean': 1.0104739334490278, 'median': 1.001241497227007, 'upper_median': 1.001241497227007, 'better_gt_1pct': 2, 'within_1pct': 18, 'worse_gt_1pct': 3, 'worse_gt_5pct': 1}
corrected_without_gate vs pool, fold 0: {'n': 23, 'geo_mean': 0.9309227104686275, 'median': 0.9778832196816405, 'upper_median': 0.9778832196816405, 'better_gt_1pct': 14, 'within_1pct': 5, 'worse_gt_1pct': 4, 'worse_gt_5pct': 2}
corrector without gate, smallest four: [{'table': '560_bodyfat', 'rows': 201, 'ratio_vs_pool': 0.6517751076372348, 'corrector_epoch': 54}, {'table': '690_visualizing_galaxy', 'rows': 258, 'ratio_vs_pool': 0.9778832196816405, 'corrector_epoch': 1}, {'table': 'nikuradse_1', 'rows': 289, 'ratio_vs_pool': 0.9753380817380265, 'corrector_epoch': 1}, {'table': '519_vinnie', 'rows': 304, 'ratio_vs_pool': 0.9948232390569536, 'corrector_epoch': 1}]
corrector without gate, tables worse by >1%: {'666_rmftsa_ladata': {'rows': 406, 'ratio': 1.0265241313699451, 'corrector_epoch': 2}, 'solar_flare': {'rows': 852, 'ratio': 1.0195897303437773, 'corrector_epoch': 5}, '557_analcatdata_apnea1_rated6': {'rows': 380, 'ratio': 1.416490456277687, 'corrector_epoch': 3}, '557_analcatdata_apnea1': {'rows': 380, 'ratio': 1.090018107979402, 'corrector_epoch': 1}}
times: {'median_min': 15.924166666666665, 'min_min': 2.0783333333333336, 'max_hours': 2.2826111111111107, 'rows_of_longest': 2400, 'longest_table': 'health_insurance'}
```

### analysis/ladder.py

What each part adds on ScoringBench: Ashlar's earlier entries and the dressed model, each ranked alone among the same 20 untuned published models (21 models, 99 tables, CRPS, ScoringBench's autorank), and the CRPS ratio between each step and the one before it.

```
tables outside the 99, dropped before ranking: {'One network, at most 100 epochs': ['3D_Estimation_using_RSSI_of_WLAN_dataset_complete_1_target']}
{'step': 'One network, at most 100 epochs', 'tables': 99, 'folds': 495, 'mean_rank': 4.7273, 'position': 2, 'best_other': 'condensite_pitbst_v0_1_1', 'best_other_mean_rank': 4.4444, 'lead': -0.2828, 'cd': 3.1477}
{'step': 'One network', 'tables': 99, 'folds': 495, 'mean_rank': 3.5758, 'position': 1, 'best_other': 'condensite_pitbst_v0_1_1', 'best_other_mean_rank': 4.5051, 'lead': 0.9293, 'cd': 3.1477, 'ratio_to_previous_geo': 0.9338, 'ratio_to_previous_median': 0.9909, 'lower_by_more_than_1pct': 46, 'within_1pct': 47, 'higher_by_more_than_1pct': 6, 'lower_at_all': 74, 'higher_at_all': 25, 'largest_ratio': 1.0616, 'tables_compared': 99}
{'step': 'One network pooled with one label reader', 'tables': 99, 'folds': 495, 'mean_rank': 1.6162, 'position': 1, 'best_other': 'condensite_pitbst_v0_1_1', 'best_other_mean_rank': 4.6667, 'lead': 3.0505, 'cd': 3.1477, 'ratio_to_previous_geo': 0.9031, 'ratio_to_previous_median': 0.9852, 'lower_by_more_than_1pct': 62, 'within_1pct': 37, 'higher_by_more_than_1pct': 0, 'lower_at_all': 93, 'higher_at_all': 5, 'largest_ratio': 1.0043, 'tables_compared': 99}
{'step': 'Five cross-fitted members, not dressed', 'tables': 99, 'folds': 495, 'mean_rank': 1.3838, 'position': 1, 'best_other': 'condensite_pitbst_v0_1_1', 'best_other_mean_rank': 4.7071, 'lead': 3.3232, 'cd': 3.1477, 'ratio_to_previous_geo': 0.9776, 'ratio_to_previous_median': 0.9869, 'lower_by_more_than_1pct': 57, 'within_1pct': 41, 'higher_by_more_than_1pct': 1, 'lower_at_all': 95, 'higher_at_all': 4, 'largest_ratio': 1.2355, 'tables_compared': 99}
{'step': 'Dressed model', 'tables': 99, 'folds': 495, 'mean_rank': 1.3333, 'position': 1, 'best_other': 'condensite_pitbst_v0_1_1', 'best_other_mean_rank': 4.7273, 'lead': 3.3939, 'cd': 3.1477, 'ratio_to_previous_geo': 0.987, 'ratio_to_previous_median': 0.9998, 'lower_by_more_than_1pct': 20, 'within_1pct': 79, 'higher_by_more_than_1pct': 0, 'lower_at_all': 57, 'higher_at_all': 42, 'largest_ratio': 1.0051, 'tables_compared': 99}
```

### analysis/loading_fix_tables.py

The tables that needed a local loading fix under pandas 3, and how they overlap the 12 few-values tables.

```
                            dataset  among_the_99  few_values
                             boston          True       False
                            sensory          True        True
                       chscase_foot         False       False
                          yprop_4_1          True       False
                analcatdata_supreme          True        True
                   visualizing_soil          True       False
                       black_friday          True       False
Mercedes_Benz_Greener_Manufacturing          True       False
               OnlineNewsPopularity          True       False
      particulate-matter-ukair-2017          True       False
       SGEMM_GPU_kernel_performance          True       False
                          Moneyball          True       False
12 tables with the fix; 11 among the 99; overlap with the 12 few-values tables: ['analcatdata_supreme', 'sensory']
```

### analysis/record.py

The smaller numbers of the development record: the run's folds, the first run's mixture head, the development tables, the design tests on them before the dressed model, the two tables whose saved pool predictions were overwritten, and the quantile levels the wrapper passes to ScoringBench.

```
the run: {'folds': 495, 'distinct_table_folds': 495, 'tables': 99, 'folds_per_table': [5]}
first run, mixture head, among the 20 untuned: {'tables': 99, 'mean_rank': 4.666666666666667, 'position': 2, 'first': 'condensite_pitbst_v0_1_1', 'first_mean_rank': 4.434343434343434, 'best_other': 'condensite_pitbst_v0_1_1', 'best_other_mean_rank': 4.434343434343434, 'cd': 3.1477414603326133}
development tables: {'n': 23, 'by_source': {'OpenML-CTR23': 12, 'PMLB': 7, 'binned copy': 4}, 'distinct_tables': 19, 'ratings_min': 5, 'ratings_max': 10, 'at_most_20_values': 7, 'at_most_20_values_binned': 4, 'results_in_training_length_test': 69, 'folds_in_any_file': [0, 1, 2], 'among_the_99_by_name': []}
training length, longer with cooldown / at most 100 epochs: {'tables': {'n': 23, 'geo_mean': 0.8129937251910794, 'better': 17, 'worse': 0, 'lower_at_all': 20, 'higher_at_all': 3}, 'fits': {'n': 69, 'geo_mean': 0.8122244912014454, 'better': 48, 'worse': 3, 'lower_at_all': 57, 'higher_at_all': 12}, 'cooldown_against_none_geo_over_fits': 0.9873640308982804, 'best_epoch_past_100': 47, 'ran_to_1000_epochs': 1}
reader prototype: {'tables': 6, 'k1_within_10pct_of_network': 4, 'k1_few_values_table': ['cars_rated7'], 'k1_few_values_ratio': 0.7222660440515289, 'k1_passes': True, 'k2_synthetic_only_10pct_better_than_neighbours': 4, 'k2_synthetic_then_real_3pct_better_on_small': 1, 'k2_small_tables': ['560_bodyfat', '557_analcatdata_apnea1', 'cars'], 'k2_synthetic_then_real_over_reader_small': {'560_bodyfat': 0.6121363624219943, '557_analcatdata_apnea1': 1.234338592674069, 'cars': 0.9783714285458431}, 'k2_passes': False}
solved last layer: {'tables': 10, 'geo_mean': 1.0205877648891493, 'worse_by_more_than_half_pct': 5, 'passes': False}
one pair, fold 0: {'n': 23, 'geo_mean': 0.6810747649311979, 'better': 12, 'worse': 2, 'lower_at_all': 16, 'higher_at_all': 3}
rewrite against the prototype, fold 0: {'tables': 23, 'off_by_more_than_1pct': 6, 'within_0_1pct': 10, 'reader_off_by_more_than_1pct': 6, 'pool_off_by_more_than_1pct': 3}
one pair, folds 1 and 2: {'n': 46, 'geo_mean': 0.6180425729953911, 'better': 27, 'worse': 4, 'lower_at_all': 31, 'higher_at_all': 9, 'worse_fits': [{'table': '560_bodyfat', 'fold': 2, 'rows': 202}, {'table': '690_visualizing_galaxy', 'fold': 1, 'rows': 258}, {'table': '666_rmftsa_ladata', 'fold': 1, 'rows': 406}, {'table': '666_rmftsa_ladata', 'fold': 2, 'rows': 406}], 'worse_fits_max_rows': 406}
five pairs pooled: {'against_taken_singly': {'n': 23, 'geo_mean': 0.9566190070507369, 'better': 16, 'worse': 0, 'lower_at_all': 23, 'higher_at_all': 0}, 'against_one_pair': {'n': 23, 'geo_mean': 0.9718460563483933, 'better': 16, 'worse': 3, 'lower_at_all': 19, 'higher_at_all': 4}}
network on the reader: {'network_on_reader': {'against_one_pair': {'n': 23, 'geo_mean': 1.1904132121552735, 'better': 6, 'worse': 15, 'lower_at_all': 6, 'higher_at_all': 17}, 'against_five_pairs_pooled': {'n': 23, 'geo_mean': 1.2248989481195434, 'better': 3, 'worse': 17, 'lower_at_all': 4, 'higher_at_all': 19}}, 'network_on_five_readers': {'against_one_pair': {'n': 23, 'geo_mean': 1.1708316862789272, 'better': 6, 'worse': 14, 'lower_at_all': 7, 'higher_at_all': 16}, 'against_five_pairs_pooled': {'n': 23, 'geo_mean': 1.2047501542355392, 'better': 3, 'worse': 18, 'lower_at_all': 4, 'higher_at_all': 19}}}
the guide's figure: {'tables_with_a_dot_at_or_below_0_4': ['557_analcatdata_apnea1', 'fps_benchmark', 'naval_propulsion_plant', 'wave_energy'], 'tables_with_both_dots_from_0_5_to_0_95': 9, 'tables_with_a_dot_from_0_5_to_0_95': 15, 'last_five_tables': ['560_bodyfat', 'health_insurance', '666_rmftsa_ladata', 'solar_flare', 'forest_fires'], 'last_five_with_both_dots_within_2pct': ['health_insurance', '666_rmftsa_ladata', 'solar_flare', 'forest_fires'], 'fits_with_ring_right_of_dot': 38, 'tables_with_ring_right_of_dot_on_both_folds': 17, 'largest_ring': 1.2417599954676384, 'largest_ring_at': ['fps_benchmark', 1]}
tables whose saved pool files were overwritten: {'227_cpu_small': {'shares_labels_with': 'cpu_act', 'same_rows_and_value_counts_every_fold': True, 'folds': 5, 'folds_where_pool_differs_from_partner': 5, 'pool_crps_range': [1.3042268726107238, 1.3874759837840425], 'partner_pool_crps_range': [1.116471962503976, 1.1786841733827527], 'pool_over_own_dressed_range': [0.9999759261751161, 1.0103198348161746]}, 'yprop_4_1': {'shares_labels_with': 'topo_2_1', 'same_rows_and_value_counts_every_fold': True, 'folds': 5, 'folds_where_pool_differs_from_partner': 5, 'pool_crps_range': [0.013302090147930564, 0.015729800235291785], 'partner_pool_crps_range': [0.013275334741726702, 0.015631640119518987], 'pool_over_own_dressed_range': [0.9992059667700177, 1.0003913816420433]}}
quantile levels: {'n': 203, 'lowest': 0.001, 'highest': 0.999, 'increasing': True}
```

### analysis/fig_untuned_ranks.py

Mean rank of the 21 untuned models on 99 tables, with the Nemenyi critical difference marked.

```
wrote figures/untuned_ranks.pdf and figures/untuned_ranks.png
```

### analysis/fig_full_field.py

Mean ranks of all 64 models on 99 tables, one row per family, the dressed model marked.

```
wrote figures/full_field.pdf and figures/full_field.png
```

### analysis/fig_table_ratios.py

By table, the dressed model's CRPS divided by the best other untuned model's, sorted.

```
wrote figures/table_ratios.pdf and figures/table_ratios.png
```

### analysis/fig_coverage.py

Empirical coverage of central intervals minus nominal, averaged over tables, by nominal level.

```
wrote figures/coverage.pdf and figures/coverage.png
```

