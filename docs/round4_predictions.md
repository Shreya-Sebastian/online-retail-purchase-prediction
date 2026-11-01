# Round 4: predictions written before the run

Written on 1 November 2026, before the round-4 code was run on the real data. They are committed together with the code, before notebook 04 is rerun on Databricks. The results will be reported next to them unchanged, whether they hold or not.

Round 4 answers three questions a reviewer is likely to ask about rounds 1 to 3:

- All test months so far fall in June to September 2011. Do the findings hold over a longer stretch, including the Christmas season?
- Every tree model used one random seed (7), and early stopping holds out a random 15% of the training rows. Is the gap over the baseline just this seed?
- Every model used one fixed setting. Would tuning change the conclusions?

## Design

1. **Longer rolling evaluation.** The existing rolling evaluation (each test month trained on all earlier months, the month before held back) is extended from the last 4 cutoffs to the last 10: December 2010 to September 2011. Models: the order-count rule, base trees, extended trees. Nothing else changes.
2. **Seeds.** The base and extended trees are refitted with seeds 7, 11, 23, 42 and 99 on the main split (train to July 2011, validate August, test September), and scored on the test month.
3. **Tuning.** On the main split, each tree model is refitted with the same 20 random settings (learning rate, number of leaves, L2 regularisation, minimum leaf size; drawn once with a fixed seed), and each logistic regression with 7 values of C. The setting with the best validation average precision is kept, as for model choice in rounds 1 to 3, and reported on the test month. The test month is not used to choose anything.

## Predictions

1. **Longer window.**
    1. The extended trees beat the order-count rule on ROC AUC in all 10 test months for bags and for kitchen, and in at least 8 of 10 for any purchase.
    2. For any purchase, the three lowest test-month ROC AUCs of the extended trees are among the December, January and February cutoffs (the label windows that start around and after Christmas).
    3. Averaged over the 10 months, the extended trees' lead over the rule is at least half as large as averaged over June to September only, for every target.
2. **Seeds.**
    1. Across the five seeds, the test ROC AUC of each tree model varies by less than 0.005 (highest minus lowest), for every target.
    2. The trees' lead over the order-count rule keeps its sign for every seed and target.
3. **Tuning.**
    1. The tuned trees' test ROC AUC is within 0.01 of the default trees', for every target.
    2. After tuning, a tree model still has the best validation average precision among the plain models, for every target.
    3. The tuned extended trees' lead over the order-count rule on the test month has a paired-bootstrap interval above zero for kitchen and for bags.
