# Grading answers: F1 > 0.5 vs exact match

**Decision: grade answers with word-overlap F1 > 0.5, for training and evaluation, in every arm.**
Exact match rejects about 5% of correct answers; F1 > 0.5 accepts about 1% of wrong ones. F1 is
also what the Rewarding Doubt paper says it uses. The released training code instead defaults to
exact match, and our PPO runs inherited that, so they learned from noisier labels than the
discrete-exact runs. That mismatch partly explains the calibration gap we reported (see "Effect on
reported results").

## The two graders

Both are verbatim ports of the released `SingleAnswerSetting/util/EvaluationMetrics.py`
(`rewarding_doubt.paper_ppo`). They share its normalization: lowercase, punctuation replaced by
spaces, articles removed, whitespace collapsed.

| | Rule | Used by |
|---|---|---|
| **Exact match** | normalized prediction equals a normalized gold alias | released `Train.py` reward: `QAResult_to_reward` defaults to `Metric.EXACT`; our PPO runs |
| **F1 > 0.5** | max over aliases of token-overlap F1 exceeds 0.5 | the paper's text (Sec. 4); the released `Evaluation.py` notebook settings; our discrete-exact training; every evaluation we report |

Gold aliases are TriviaQA's `normalized_aliases`, the released code's `gt_candidates`.

## How often they disagree

On the base model's answers to our 512 held-out questions (509 well-formed):

- **33 answers (6.5%) are graded differently.** Every disagreement goes the same way: F1 accepts, exact match rejects.
- Accuracy is 63.3% under F1 and 56.8% under exact match.

Regenerate the list with `python scripts/grader_disagreements.py`.

## Every disagreement, judged

Verdict: whether the answer is actually correct, judged by reading the question.

| # | Question | Model's answer | Closest alias | F1 | Verdict |
|---|---|---|---|---:|---|
| 0 | What claimed the life of singer Kathleen Ferrier? | Breast cancer | cancer | 0.67 | correct |
| 1 | What is the Japanese share index called? | Nikkei 225 | nikkei | 0.67 | correct |
| 2 | Which highway was Revisited in a classic 60s album by Bob Dylan? | Highway 61 | 61 | 0.67 | correct |
| 3 | What Michelle Pfeiffer movie got a boost from the Coolio song Gangsta's Paradise? | Dangerous | dangerous minds | 0.67 | ambiguous (truncated title) |
| 4 | What was the Paramount Film Company originally called? | Adolph Zukor's Famous Players Film Company | famous players film company | 0.73 | correct |
| 5 | On what date in 1969 did Neil Armstrong first set foot on the Moon? | July 20, 1969 | july 20 | 0.80 | correct |
| 6 | Which Joan's career revived in Whatever Happened to Baby Jane? | Joan Crawford | crawford | 0.67 | correct |
| 7 | On which date in 1945 did Hitler take cyanide then shoot himself? | April 30, 1945 | 30 april | 0.80 | correct |
| 8 | In which river is the Boulder Dam? | Colorado River | colorado | 0.67 | correct |
| 9 | Which grand slam did Pete Sampras not win in the 20th century? | French Open | french | 0.67 | correct |
| 10 | What percentage of the earth's surface is covered by Europe? | 6.8% | 8 | 0.67 | **wrong** (digit split) |
| 11 | Who first drew Mickey Mouse when Disney first supplied the voice? | Walt Disney and Ub Iwerks | iwerks ub | 0.57 | ambiguous (extra name) |
| 12 | Throughout the 80s and 90s Phil Collins recorded on which record label? | Atlantic Records | atlantic | 0.67 | correct |
| 13 | On which label did Chuck Berry record in the 1950s and 1960s? | Chess Records | chess | 0.67 | correct |
| 14 | Which President of the Philippines was deposed in 1986? | Ferdinand Marcos | marcos | 0.67 | correct |
| 15 | Who did Dr. Crippen murder? | His wife Belle Elmore | his wife | 0.67 | correct |
| 16 | The Black Hills lie between which two rivers? | Belle Fourche River and Cheyenne River | belle fourche and cheyenne | 0.80 | correct |
| 17 | In which city was John Lennon murdered? | New York City | new york | 0.80 | correct |
| 18 | In which battle did Harold II, the last Saxon king, lose his life? | Hastings | hastings battle | 0.67 | correct |
| 19 | What was the former name of the British Green Party? | People Party | people party uk 1973 75 | 0.57 | correct |
| 20 | What raw material is used for making glass? | Silica sand | sand | 0.67 | correct |
| 21 | What are the three primary colours of light? | Red, Green, and Blue | red blue and green | 1.00 | correct |
| 22 | What was advertised with Eva Herzigova using the slogan "hello boys"? | Bra | wonder bra | 0.67 | correct (generic) |
| 23 | What is the most populated city in America? | New York City | new york | 0.80 | correct |
| 24 | The melody for which famous song was written by sisters Patty and Mildred Hill? | Happy Birthday to You | happy birthday | 0.67 | correct |
| 25 | How old was Luigina Giavotti when she won a gymnastics silver medal? | 11 years and 161 days | 11 years and 302 days | 0.80 | **wrong** (shared words) |
| 26 | Which brand of beer does Homer Simpson drink regularly? | Duff Beer | duff | 0.67 | correct |
| 27 | Which animal has the longest gestation period at around 22 months? | African Elephant | elephant | 0.67 | correct |
| 28 | For what did Einstein get the Nobel prize in Physics? | Not the photoelectric effect | photoelectric effect | 0.80 | **wrong** (negation) |
| 29 | What is Robin Williams' character called in Good Morning Vietnam? | Adrian Cronauer | adrian | 0.67 | correct |
| 30 | Whose arch nemesis is the Red Skull? | Captain America | captain america s | 0.80 | correct |
| 31 | Macbeth belonged to which royal house or dynasty? | House of Stuart | house of dunkeld | 0.67 | **wrong** (shared "house of") |
| 32 | Which notorious murderer lived at 10 Rillington Place? | John Reginald Christie | christie john | 0.80 | correct |

**Tally: 27 correct, 4 wrong, 2 ambiguous.**

- **Exact match fails on correct answers that are more specific or differently worded than every alias.**
  Examples: "Breast cancer" vs "cancer"; "Ferdinand Marcos" vs "marcos"; "New York City" vs "new york";
  "July 20, 1969" vs "july 20"; reordered lists ("Red, Green, and Blue").
- **F1 > 0.5 fails through token overlap without meaning.** Examples: a negation ("Not the photoelectric
  effect"); shared function words ("House of Stuart" vs "house of dunkeld"); shared units
  ("11 years and … days"); and the normalizer splitting "6.8" into "6" and "8".

## Estimated error rates

Over the 509 well-formed answers, counting only the disagreements:

| Grader | Mislabels | Rate |
|---|---|---:|
| Exact match | 27–29 correct answers marked wrong | ≈ 5.5% |
| F1 > 0.5 | 4–6 wrong answers marked correct | ≈ 1% |

Exact match's errors are not random. They fall on specific, complete answers, so a model trained on
exact-match rewards is told that some of its best-formed answers are wrong. That pushes it toward
lower confidence on exactly those answers.

## Effect on reported results

Re-grading the saved per-question outputs (end of training, 3 seeds each; ECE / AUROC / Brier on the
sampled confidence) from F1 to exact match:

| Arm | Training label | F1 grading | Exact-match grading |
|---|---|---|---|
| Base model | – | 0.333 / 0.560 / 0.337 | 0.397 / 0.563 / 0.397 |
| PPO + KL | exact match | 0.127 / 0.706 / 0.217 | 0.169 / 0.685 / 0.248 |
| PPO − KL + hinge | exact match | 0.070 / 0.744 / 0.190 | **0.078** / 0.721 / 0.209 |
| Exact + KL | F1 | 0.085 / 0.808 / 0.174 | 0.130 / 0.776 / 0.204 |
| Exact + hinge | F1 | **0.057** / 0.789 / 0.174 | 0.093 / 0.762 / 0.200 |
| Exact + KL + value head | F1 | 0.097 / **0.826** / **0.168** | 0.149 / **0.794** / 0.201 |

- **Each method is best calibrated under the label it was trained on.** Under F1 grading, exact + hinge
  has the lowest ECE. Under exact-match grading, PPO + hinge does. Both differences are within seed
  noise.
- **The ranking advantage does not depend on the grader.** Discrete-exact has higher AUROC than PPO
  under both, and lower Brier.
- **Our 2×2's ECE comparison between PPO and discrete-exact was therefore confounded by the training
  label.** The AUROC, Brier and seed-stability conclusions stand. The hinge-vs-KL effect within each
  method stands too: it holds under both graders.

## What we do

1. **Grade with F1 > 0.5 for training and evaluation in every arm.** The PPO wrapper gains an option to
   use F1 in `Train.py`'s reward (its parser and −30 are unchanged). This matches the paper's stated
   method and our evaluation.
2. **Re-run the PPO arms of the 2×2 with F1 training labels** before drawing ECE conclusions between the
   methods.
3. Keep exact match as a secondary evaluation column. It costs nothing to compute from saved outputs
   and catches F1's false positives.

## Limitations

- One reviewer (the analysis agent) judged the 33 disagreements. Items 3, 11 and 22 are debatable; none
  of them changes the conclusion.
- Only disagreements were examined. Both graders can be wrong together, for example on a wrong answer
  that matches a bad alias, and this check cannot see those cases.
- The sample is the base model's answers on 512 questions. Trained models give the same kinds of
  answers (accuracy is unchanged), so the error modes should transfer, but the rates were not
  re-measured per run.
