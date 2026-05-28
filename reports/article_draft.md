# Detecting "We vs. Them" Language on Social Media: A Multi-Step NLP Pipeline for Othering Detection

**Julian Ray-Constanty**
*Robert Gordon University · julian.rayconstanty@etu.unilim.fr · 28/05/2026*

## Abstract

Online hate speech rarely comes labelled. More often it hides inside rhetorical structures, particularly the construction of a threatened "we" against a dangerous "them." This paper describes a pipeline built to detect and measure that dynamic across 134,459 social media posts, drawn from the UC Berkeley Measuring Hate Speech corpus and a sample of politically active Reddit communities. Rather than training a single end-to-end model, we chain several off-the-shelf tools: pronoun tagging, toxicity scoring via Detoxify, emotion detection via GoEmotions, a handcrafted pattern dictionary, a supervised classifier, and BERTopic topic modeling. The approach is deliberately modular; each step adds columns to the same dataset, so any component can be swapped out independently. The main finding is that pronoun structure is a surprisingly strong signal: posts containing both "we" and "them" markers are flagged as othering at a rate of 22.2%, versus 1.2% for posts with neither. Overall, 5.3% of the corpus contains othering language, and 74.6% of those posts also score above 0.7 on toxicity, well above the corpus average.

**Keywords:** othering detection, hate speech, NLP, social media, toxicity, BERTopic, in-group/out-group language


## 1. Introduction

Social media has given exclusionary rhetoric a reach it never had before. What used to circulate in pamphlets or fringe forums now spreads through algorithmic feeds to millions of people, often within hours of being posted. One of the more persistent features of this content is *othering*, the rhetorical move of framing an out-group as threatening, alien, or subhuman in contrast to a cohesive, innocent "us" [CITE: Tajfel & Turner 1979; van Dijk 1993]. The pattern appears in political speeches, in comment sections, in fringe communities, and increasingly in mainstream platforms. Detecting it automatically is hard, partly because it doesn't always look like hate speech in the traditional sense: a post can be technically civil while still systematically dehumanizing a group.

A lot of work has attacked pieces of this problem. Toxicity classifiers [CITE: Jigsaw; Hanu 2020] can flag insults and threats. Hate speech datasets [CITE: Kennedy et al. 2020] provide labeled training data. Researchers have studied pronoun use and social deixis in online communities [CITE: Del Tredici & Fernández 2019]. What's less common is a single study that pulls all these signals together on the same corpus and asks how they relate to each other. That's what we try to do here.

The pipeline we built runs six steps sequentially. It's not particularly novel in any single component; we're largely combining existing tools. But the combination produces a richer picture than any one tool alone would. The key questions we're trying to answer: how prevalent is othering language in political and hate speech content? Do pronouns predict it? What topics cluster around it? And how toxic, emotionally, and linguistically distinct are othering posts from the rest?


## 2. Related Work

The literature on automated hate speech detection is substantial. Schmidt and Wiegand (2017) [CITE] and Fortuna and Nunes (2018) [CITE] provide thorough overviews, and the field has moved from simple keyword lists to transformer-based classifiers over the past decade. The dataset we use as our primary source, Kennedy et al.'s Measuring Hate Speech corpus [CITE: 2020], is notable for its annotation approach: rather than binary hate/not-hate labels, it collects severity scores from multiple annotators, which gives a more continuous and arguably more realistic picture of how harmful content is perceived.

On the toxicity side, the Jigsaw Perspective API [CITE: 2017] and Detoxify [CITE: Hanu 2020] have become standard tools. Detoxify in particular is lightweight enough to run on a CPU across large corpora, which made it practical for our use case. GoEmotions [CITE: Demszky et al. 2020] takes a different angle: rather than toxicity, it predicts fine-grained emotions from Reddit comments across 27 categories. The connection between emotion and harmful language is documented; anger and disgust tend to co-occur with dehumanizing content [CITE: Salminen 2018], though the relationship is noisy.

The theoretical backbone of our work comes from Social Identity Theory [CITE: Tajfel & Turner 1979] and its application to discourse analysis [CITE: van Dijk 1993]. The core idea is simple: people categorize themselves and others into groups, and out-group members get systematically devalued. In language this shows up as pronoun choices, threat metaphors, and exclusionary formulations. Del Tredici and Fernández (2019) [CITE] operationalize some of these patterns computationally; Davidson et al. (2017) [CITE] study hate speech on Twitter with lexical features. We build on these approaches but focus specifically on the othering structure rather than hate speech as a monolithic category.

For topic modeling, we use BERTopic [CITE: Grootendorst 2022], which combines sentence embeddings with UMAP and HDBSCAN clustering. It handles short, noisy texts better than LDA and produces more interpretable outputs, at the cost of sometimes generating an unwieldy number of topics on large, diverse corpora, as we found out.


## 3. Data

### 3.1 Sources

The dataset combines two publicly available sources. The first is the **UC Berkeley Measuring Hate Speech corpus** [CITE: Kennedy et al. 2020], loaded from HuggingFace (`ucberkeley-dlab/measuring-hate-speech`, train split). It contains social media posts, primarily from Twitter and Reddit, annotated by crowdworkers for hate speech severity across multiple dimensions. We use `annotator_severity` as a post-level score.

The second is the **Pushshift Reddit archive** [CITE: Baumgartner et al. 2020], accessed via HuggingFace (`fddemarco/pushshift-reddit`). We filtered to five subreddits selected for their political salience: r/politics, r/worldnews, r/conspiracy, r/europe, and r/immigration. Post titles and body text were merged into a single field.

### 3.2 Dataset statistics

After removing posts under 20 characters and deduplicating, the final corpus contains **134,459 posts**. Of those, 133,170 come from the hate speech corpus (99.04%) and 1,289 from Reddit (0.96%). Within Reddit: r/politics contributes 782 posts, r/worldnews 407, r/conspiracy 87, and r/europe 13.

The imbalance is stark and shapes every result in this paper. The hate speech corpus was collected specifically to oversample harmful content, which inflates our toxicity and othering averages relative to what you'd see on a random sample of social media. The Reddit posts are almost certainly more representative of typical political discourse, but there simply aren't enough of them to draw strong conclusions at the platform level. We come back to this in the limitations section.

### 3.3 Preprocessing

Each post goes through a cleaning function that lowercases the text, strips URLs, removes @mentions, and normalizes special characters while keeping basic punctuation. The result is stored in a `clean_text` column, which all downstream steps operate on.


## 4. Methods

The pipeline has six steps, each building on the previous one by adding new columns to the dataset. The design is intentionally incremental: if one step produces bad output, it doesn't corrupt the earlier columns.

### 4.1 Pronoun tagging

We tag two groups of plural pronouns. WE markers cover the standard first-person plural forms: *we, us, our, ours, ourselves*. THEM markers cover third-person plural plus several multiword expressions: *they, them, their, theirs, those people, these people, people like them*. For each post we record binary presence flags, raw counts, and a `pronoun_type` category: `we_only`, `them_only`, `both`, or `none`. Matching is case-insensitive on the cleaned text. The THEM list is deliberately broader than the strict grammatical third person, since phrases like "those people" and "people like them" carry a derogatory framing that bare pronouns don't.

### 4.2 Toxicity scoring

We run Detoxify (`original` model) across the full corpus in batches of 32. This gives five scores per post: `toxicity`, `severe_toxicity`, `identity_attack`, `insult`, and `threat`. All are floats between 0 and 1. We use `toxicity` as the primary signal in most downstream analyses, with 0.7 as a high-toxicity cutoff.

### 4.3 Emotion classification

The GoEmotions model (`monologg/bert-base-cased-goemotions-original`) assigns a single top emotion and a confidence score to each post. We use the HuggingFace pipeline API with `top_k=1` and truncate inputs to 512 characters. The truncation affects a minority of posts but is worth noting since it can cut off the most emotionally loaded parts of longer texts.

### 4.4 Rule-based othering detector

The core of the othering detection is a hand-built dictionary of 33 surface patterns across four categories. The first, threat metaphors, captures dehumanization through natural-disaster or infestation imagery: *invasion, flood, swarm, horde, plague, wave of*, and similar. The second, moral exclusion, targets explicit boundary-drawing that denies group membership: *go back, don't belong, not like us, their kind, no place here*. The third, generalization, flags sweeping claims about a group as a whole: *they always, all of them, none of them ever, these people always*. The fourth, threat framing, covers replacement and civilizational-threat narratives: *replace us, great replacement, taking over, undermining our, destroying our way*.

A post's `othering_score` is the number of categories that matched (0 to 3), and `has_othering` is true if at least one category matched. Every matched string is stored in a `matched_patterns` field for inspection. The dictionary is deliberately conservative; we'd rather miss some othering posts than flood the dataset with false positives.

### 4.5 Supervised classifier

To test whether the othering signal can be learned beyond the explicit patterns, we train four classifiers using `has_othering` as the label. We combine two feature representations, TF-IDF (`TfidfVectorizer`, max 10,000 features, fitted on the training split) and sentence embeddings (`all-MiniLM-L6-v2` [CITE: Reimers & Gurevych 2019], encoded in batches of 64), with two model families: `LogisticRegression(max_iter=1000)` and `SVC(kernel="linear", probability=True)`. The data is split 80/20 with `random_state=42`. We evaluate on precision, recall, F1, and confusion matrix, and run error analysis on the best model to understand what it gets wrong.

### 4.6 Topic modeling

BERTopic [CITE: Grootendorst 2022] runs on the full cleaned corpus using `all-MiniLM-L6-v2` embeddings, `min_topic_size=50`, and automatic topic count selection. Embeddings and UMAP projections are cached to avoid recomputing them on reruns. After fitting, we manually review the top words per topic and assign human-readable `topic_name` labels. The topic assignments are then cross-referenced with toxicity, emotion, and othering rates to get a per-topic profile.


## 5. Results

### 5.1 Pronoun distribution

Across all 134,459 posts, 18.0% contain at least one WE marker, 25.2% at least one THEM marker, and 6.7% contain both. Most posts (56.6%) contain neither.

### 5.2 Toxicity

The average toxicity score is **0.595**, and 56.0% of posts score above 0.7. These numbers are high, but they largely reflect the composition of the corpus rather than the state of social media in general. The hate speech dataset was collected to study harmful content, not to be a representative sample.

### 5.3 Emotions

*Neutral* dominates at 35.5%, which is a useful sanity check: even in a hate speech corpus, a lot of posts are just statements or descriptions. After that: *anger* (18.2%), *annoyance* (9.9%), *curiosity* (5.6%), and *admiration* (4.8%). The emotional distribution in othering posts skews more toward anger and disgust relative to the full corpus, though the difference is a matter of degree rather than a categorical shift.

### 5.4 Othering detection

The rule-based detector flagged **7,115 posts (5.3%)** as containing othering content. Table 1 breaks this down by pronoun type.

**Table 1. Othering rate by pronoun type**

| Pronoun type | Othering rate |
|---|---|
| both (we + them) | 22.2% |
| them_only | 12.3% |
| we_only | 6.9% |
| none | 1.2% |

The gradient here is striking. Posts with THEM markers are roughly 10x more likely to contain othering language than posts with neither pronoun type. Adding WE markers roughly doubles the rate again. This suggests that explicit in-group/out-group construction in the pronoun structure is a reliable, cheap surface signal for othering intent.

Of the 7,115 flagged posts, **5,305 (74.6%)** also score above 0.7 on toxicity. The full-corpus rate at that threshold is 56%, so othering posts are meaningfully more toxic than average, but they're not uniformly extreme either. Around a quarter of othering posts have relatively low toxicity scores, which points to the kind of implicit or polite exclusionary language the pattern dictionary is least equipped to catch.

### 5.5 Classifier performance

The supervised classifier predicts an othering rate of **5.29%** on the test set, essentially the same as the rule-based rate of 5.3%. The F1 scores are high across all four model/feature combinations.

*[Insert Table 2: precision/recall/F1 per model x embedding type, from step4 output]*

The close match between predicted and actual rates tells us the classifier has learned the patterns well. What it doesn't tell us is whether it can generalize beyond them. Error analysis reveals two recurring failure modes: false positives on posts that use "invasion" or "flooding" literally (news about natural disasters, military campaigns), and false negatives on posts using coded language or irony that isn't covered by any pattern.

### 5.6 Topic modeling

BERTopic found **1,334 topics**, more than we'd hoped for, but not surprising given the size and diversity of the corpus with `nr_topics="auto"`. The largest clusters are organized around racial slurs and abuse (2,300 posts), abortion debate (1,194), Iran and nuclear geopolitics (1,080), homophobia (961), and antisemitism (944).

*[Insert Table 3: per-topic othering rate, avg toxicity, top emotion, from step5 cross-analysis output]*

The topic structure confirms something intuitive: othering language isn't evenly distributed. It concentrates in specific thematic areas like migration, racial identity, and religious conflict, while being nearly absent from others.

### 5.7 Platform comparison

Table 4 compares the Reddit subreddits on toxicity and othering rate.

**Table 4. Platform comparison by subreddit**

| Subreddit | Posts | Avg toxicity | Othering rate |
|---|---|---|---|
| r/politics | 782 | 0.061 | 0.90% |
| r/worldnews | 407 | 0.047 | 0.49% |
| r/conspiracy | 87 | 0.050 | 0.00% |
| r/europe | 13 | 0.002 | 0.00% |

Reddit posts are dramatically less toxic than the hate speech corpus overall (0.06 vs 0.60 average). r/politics shows the highest othering rate at 0.90%, which aligns with prior work on political polarization online. The zero othering rates for r/conspiracy and r/europe are almost certainly an artifact of small sample sizes rather than a genuine absence of exclusionary content in those communities.

### 5.8 Temporal analysis

We weren't able to run the temporal analysis. The `created_utc` timestamp was present in the raw Reddit data at collection time but didn't survive the full pipeline; it was dropped somewhere between the initial dataset merge and the enriched CSV. Without it, there's no way to plot month-by-month toxicity or othering trends, or to look for spikes that might correlate with political events. This is the most frustrating gap in the results, since temporal dynamics are one of the more interesting aspects of how political rhetoric evolves online.


## 6. Discussion

The pronoun result is the clearest takeaway from this work. A 22.2% othering rate for posts with both "we" and "them" markers, versus 1.2% for posts with neither, is a large effect, and it holds up even though the pattern dictionary is fairly conservative. It suggests that surface-level pronoun structure carries real signal about rhetorical intent, which matters practically: pronoun tagging is cheap, fast, and requires no model inference. In a production content moderation system, it could plausibly serve as a first-pass filter to prioritize posts for more expensive analysis.

The agreement between the rule-based detector and the ML classifier (5.3% vs 5.29%) is worth interpreting carefully. On one reading, it's a good sign: the classifier successfully learned the signal and isn't wildly over- or under-predicting. On another reading, it's a red flag: if the classifier is essentially memorizing the dictionary, its precision and recall figures are measuring pattern coverage, not true othering detection. The error analysis supports the second interpretation; most false negatives involve framing strategies the dictionary simply doesn't cover. Implicit othering, dog-whistle language, sarcasm, and culturally specific exclusion patterns are all systematically missed.

The 74.6% overlap between othering and high toxicity (>0.7) confirms that these two signals are correlated but not redundant. About a quarter of othering posts have low toxicity scores, which corresponds to the "polite" form of exclusionary language: rhetorically sophisticated posts that exclude without insulting. Detoxify isn't designed to catch that, and neither is our rule dictionary in its current form. That gap points toward what a more robust second-generation detector would need.

One thing we didn't fully account for is the domain mismatch between the two data sources. The hate speech corpus was collected and annotated with specific harmful content in mind; the Reddit posts are a relatively small, unfiltered slice of political discussion. Comparing their othering rates directly is a bit like comparing oranges and apples. The 5.3% overall rate is dominated by the hate speech corpus; the Reddit rate of roughly 0.7% (weighted average) is a different animal. Future work would benefit from treating these as separate analyses rather than merging them.


## 7. Conclusion

We built a six-step pipeline to detect and characterize othering language across 134,459 social media posts. The main finding is that pronoun structure is a strong and cheap signal: "them"-type markers predict othering at 10x the baseline rate, and posts with both in-group and out-group markers are flagged at 22.2%. Rule-based detection (5.3%) and a trained classifier (5.29%) agree closely, though error analysis suggests the classifier is largely learning the dictionary rather than generalizing beyond it. Topic modeling reveals concrete thematic clusters around race, religion, and geopolitics where othering content concentrates.

The pipeline is modular and fully reproducible; each step leaves the dataset intact and adds columns rather than transforming the input, so individual components can be replaced without breaking the rest. The main methodological gap is the loss of timestamp data, which blocked the temporal analysis. That, along with the dataset imbalance, is the most important thing to fix in any follow-up.


## 8. Limitations and Future Work

The most concrete technical issue is the loss of timestamps. The `created_utc` field dropped out somewhere in the pipeline and we couldn't recover it from intermediate files. Rebuilding the pipeline with explicit column preservation at every step would unlock temporal analysis, spike detection, and event correlation, all of which were intended to be a significant part of the results.

The dataset composition is the other major constraint. The UC Berkeley corpus is valuable but it was collected to study harmful content, not to be representative of social media in general. All aggregate statistics in this paper (average toxicity, overall othering rate) are dominated by that source. A more balanced collection, with larger Reddit samples across more subreddits and time periods, would produce more interpretable and generalizable results.

The pattern dictionary covers the most visible forms of othering but systematically misses implicit framing, coded language, neologisms, and non-English content. The natural next step is to annotate a sample of subtle othering cases by hand and use it to fine-tune a sequence classifier, rather than relying on surface patterns.

Related to this: training the ML classifier on rule-based labels means its performance ceiling is bounded by the dictionary's recall. The high agreement between the two (5.3% vs 5.29%) is an indication of this ceiling, not of generalization. Training on independent human annotations would give a more honest measure of what the model actually knows.

Finally, 1,334 topics is more than can be meaningfully interpreted. BERTopic with `nr_topics="auto"` on a 134k-post corpus produces a very fine-grained decomposition, much of which turns out to be noise. Future runs should explore coarser groupings, either by setting a fixed topic count or by post-hoc hierarchical merging of related clusters.


## References

Baumgartner, J., Zannettou, S., Keegan, B., Squire, M., & Blackburn, J. (2020). The Pushshift Reddit dataset. *Proceedings of ICWSM 2020*.

Davidson, T., Warmsley, D., Macy, M., & Weber, I. (2017). Automated hate speech detection and the problem of offensive language. *Proceedings of ICWSM 2017*.

Del Tredici, M., & Fernández, R. (2019). You say hate speech, I say offensive language: An analysis of social factors and semantic relations. *Proceedings of RANLP 2019*.

Demszky, D., Movshovitz-Attias, D., Ko, J., Cowen, A., Nemade, G., & Ravi, S. (2020). GoEmotions: A dataset of fine-grained emotions. *Proceedings of ACL 2020*.

Fortuna, P., & Nunes, S. (2018). A survey on automatic detection of hate speech in text. *ACM Computing Surveys, 51*(4), 1-30.

Grootendorst, M. (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure. *arXiv:2203.05794*.

Hanu, L., & Unitary Team. (2020). Detoxify. https://github.com/unitaryai/detoxify

Kennedy, C. J., Bacon, G., Sahn, A., & von Vacano, C. (2020). Constructing interval variables via faceted Rasch measurement and multitask deep learning: A hate speech application. *arXiv:2009.10277*.

Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. *Proceedings of EMNLP 2019*.

Schmidt, A., & Wiegand, M. (2017). A survey on hate speech detection using natural language processing. *Proceedings of the Fifth International Workshop on NLP for Social Media*.

Tajfel, H., & Turner, J. C. (1979). An integrative theory of intergroup conflict. In W. G. Austin & S. Worchel (Eds.), *The Social Psychology of Intergroup Relations* (pp. 33-47). Brooks/Cole.

van Dijk, T. A. (1993). *Elite Discourse and Racism*. Sage.
