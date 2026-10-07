# Completed C1 compositional-transfer evaluation

All five final checkpoints were evaluated with 256 atomic training fit samples per function, seed 42, pooling grid 2, ridge 0.01, and up to 128 audited matched examples per split/program. The 20 split/program selections and all 15 atomic function fit selections were independently revalidated after consolidating the selectors. Checkpoints were only read; no model training ran.

Object-pixel accuracy below uses the prediction/target foreground union. All values are percentages. A = atomic f; B = oracle intermediate; C first = model g; C final = sequential predicted intermediate; D = direct [g,f].

| C1 | test_ood n | A | B | C first | C final | D |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 128 | 93.97 | 95.04 | 99.41 | 93.87 | 19.23 |
| 2 | 128 | 99.76 | 98.06 | 99.82 | 97.98 | 11.07 |
| 3 | 128 | 99.28 | 94.39 | 98.09 | 92.44 | 29.69 |
| 4 | 61 | 99.05 | 36.55 | 99.68 | 36.61 | 12.57 |
| 5 | 128 | 99.79 | 97.20 | 99.82 | 97.14 | 63.41 |

Direct composition exact-grid accuracy was 0% for every selected test_ood group. Exact-grid oracle-intermediate accuracy was 53.91%, 67.19%, 71.88%, 0%, and 64.84% for C1-1 through C1-5. These are support-filtered samples, not the unfiltered benchmark totals.

Final-image frozen-affine defects (normalized by target latent norm):

| C1 | Atomic ID error | Context error | Transfer gap | Full composition error |
|---|---:|---:|---:|---:|
| 1 | 0.2311 | 0.2366 | 0.0055 | 0.2410 |
| 2 | 0.0907 | 0.0941 | 0.0034 | 0.1237 |
| 3 | 0.1954 | 0.1993 | 0.0039 | 0.2758 |
| 4 | 0.1689 | 0.3074 | 0.1385 | 0.3053 |
| 5 | 0.0986 | 0.1134 | 0.0147 | 0.1289 |

C1-1/2/3/5 have strong oracle-intermediate behavior compared with direct execution. Their small image-final transfer gaps support a conditioning/binding/execution hypothesis, but do not identify a causal failing module. Absolute affine errors exceed the conservative default Type B threshold, so the automatic classifier retains inconclusive labels rather than forcing that interpretation.

C1-4 shows a different pattern: strong atomic crop_contours competence but poor oracle-intermediate execution, with final-image transfer gap about 0.1385. Of its 61 valid test_ood cases, 43 have raster-consistent intermediate reinference; on those 43 alone, A is 98.65%, B is 36.60%, C is 36.60%, and D is 12.22%. This persistence supports the transfer-failure hypothesis beyond the hidden-object-identity confound. The provisional taxonomy marks Types A and D for the full group.

Important limitations:

- Every affine source fit is underidentified (rank at most 255, latent dimension 512). Interpret held-out data-relative defects, not parameter-space algebra distances.
- Intermediate reinference agrees with persistent-object oracle targets on 100%, 100%, 94.53%, 70.49%, and 100% of test_ood cases. The stratified table separates this ambiguity.
- Geometric resizing need not commute with oracle operations. Resize/oracle order defects and their support counts are reported; prediction labels follow the existing benchmark resize convention.
- C1-4 admits only 55 val_ood and 61 test_ood examples under the audit, and only 11 val / 8 test synthetic ID contexts. C1-5 val admits 126. Coverage tables expose all exclusion reasons.
- B and D receive different images/programs by design. Their early representation divergence is observational and is not a causal localization. Constant function-only representations cannot yield centered CKA.
- Only five final models, one run each, were evaluated. Within-model examples are not independent model replicates. Pearson/Spearman p-values and bootstrap intervals are descriptive; pooled intervals resample experiment runs. Variance-threshold PCA comparisons can use different ranks, listed in the pooled provenance.
- The optional direct-composition probe is a CHEATING supervised reference, fit on the first half and scored on the second half of each selected split. It cannot establish generalization.

Validation: 73 repository tests pass; lint passes. The minimum two-fit/one-eval smoke passes, as does a two-checkpoint trajectory smoke at steps 19,500 and 78,000. Trajectory smoke samples are too small for scientific conclusions. A stale existing spatial test fixture was repaired to expose the final normalization layer. A NumPy report-serialization bug found during execution was fixed; resume support reused the frozen probes.

See README.md for exact single/all-experiment and checkpoint-trajectory commands, hooks, output schemas, calibration definitions, and interpretation limits.
