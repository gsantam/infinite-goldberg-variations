# Infinite Goldberg Variations

Bach's [*Goldberg Variations*](https://open.spotify.com/album/1aCpHSQE5ghxibsQ5gkBe0?si=8XK24O7ZTUShSqC-5yzbNQ) have fascinated listeners and musicians for a long time:
they have something hypnotic and even a bit obsessive, but they are also
delicate and playful, built around a melody that keeps coming and going. I have
come back to them many times during my life, and they still amaze me, so I
thought that, as a music generation exercise, and in a kind of play similar to
the ones suggested in [*Godel, Escher,
Bach*](https://en.wikipedia.org/wiki/G%C3%B6del,_Escher,_Bach), it would be fun
to try to see how other, maybe infinite, variations would sound.

## Why the Goldberg Variations are interesting

The *Goldberg Variations* are composed of an Aria and 30 different variations.
Each variation lasts between a minute and a few minutes, no more than six, for
a total duration of around an hour. Each one is independent from the others: it
starts from scratch, rather than as a continuation of the previous variation.
At the same time, all of them remain
tied to the main Aria. In particular, they share a very strong structure: the
same number of bars, the same broad harmonic plan, and the same large-scale
shape. But inside those constraints, each variation has its own texture, rhythm,
and character.

You can listen to the Aria here:

https://github.com/user-attachments/assets/fa3557df-cc84-4e95-8cbd-53407603f8d6

## Beyond semantic similarity

They also give us an interesting playground to explore the concept of
similarity. When similarity is defined through an embedding trained in a
text/music contrastive way, or through user-preference/music data, as in
[CLaMP 2](https://arxiv.org/abs/2410.13267), it has a basically semantic
meaning: it can capture period, country, genre, or vibe. But it does not
necessarily include much information about similarity in structure, harmony, or
shared themes. Those similarities are much more granular and instance-based:
two pieces can both sound Baroque while having completely different phrase
structures, bass motion, cadences, or relationships to a starting theme. In the
Goldberg Variations, this distinction matters because the interesting question
is not only whether a generated piece sounds like Bach in general, but whether
it behaves like another variation of this particular Aria.

I am still iterating on clean definitions of this more intrinsic similarity.
For now, it is based on different local and global comparisons of the Aria's
harmony and structure with those of the generated piece. Some of these signals
are global, such as pitch-class histogram similarity after normalizing pieces
to a common key. Others are more local, comparing bass motion, top-voice
behavior, harmonic roots, and bar-level harmony against the Aria with simple
alignment rules. None of this is a complete musical definition, but it gives a
more concrete signal for whether the generated piece behaves like another
variation of the Goldberg Aria, rather than only sounding generically Baroque.

## Modelling

I have used [NotaGen-large](https://arxiv.org/abs/2502.18008) as the
pre-trained base model, since it seems to be the state of the art for symbolic
classical music generation, and Bach and the Baroque period are well represented
in its training set. With that model, I do Aria-conditioned supervised
fine-tuning: the prompt contains the conditioning keywords `%Baroque`, `%Bach,
Johann Sebastian`, and `%Keyboard`, as well as the Aria, while each real
variation is used as the target continuation. This is very consistent with the
concept of a variation.

There is an important caveat here: NotaGen's full pre-training corpus and the
internal part of its fine-tuning data are not released, so I cannot completely
rule out that BWV 988 was already seen by the base model. The public
fine-tuning sources I checked do not seem to contain the Goldberg Variations as
sheet data, but some large MIDI sources do contain them, so I treat this as a
possible contamination risk rather than a settled point.

Since the amount of training data is small, I keep a very low learning rate
(`1e-6`) and use a k-fold-style cross-validation setup, keeping roughly 10-20% of the
variations in the test set and computing token-level log loss on them. The
results are quite consistent across splits: the test loss improves quickly and
then mostly plateaus, while clear overfitting only starts to appear around epoch
10.

![SFT train and eval loss](docs/assets/sft_train_eval_loss.svg)

In order to understand similarity, I monitor both the semantic similarity given
by the CLaMP2 embedding and different measures of local and global similarity
related to harmony and structure. I compare those numbers against the average
for the real Goldberg variations to sanity-check whether each metric is
meaningful, and to estimate how far the generations are from the kind of
similarity Bach used when composing the variations.

| Measure | What it compares | Terminal / Patch | Reward use | Epoch 0 | Epoch 1 | Epoch 8 | GT |
|---|---|---|---|---:|---:|---:|---:|
| `clamp2_aria` | CLaMP2 embedding similarity to the Aria. | sequence diagnostic | Logged only | 0.432 | 0.438 | 0.464 | 0.500 |
| `aria_strict_symbolic_component_global_base_z` | Active strict symbolic aggregate over same-bar root/bass, narrow DTW, root/bass n-grams, and cadences, normalized against base-model samples. | terminal | Active | -0.044 | -0.022 | 0.398 | 1.434 |
| `strict_aligned_root_bass` | Same-bar soft match of inferred harmonic root and bass pitch classes to the Aria. | terminal component | Active component | 0.311 | 0.304 | 0.342 | 0.440 |
| `strict_dtw_combined_narrow` | Narrow-band DTW over bar-level harmony, root, and bass sequences. | terminal component | Active component | 0.768 | 0.768 | 0.781 | 0.804 |
| `strict_root_bass_bigram_weighted_jaccard` | Weighted Jaccard overlap of consecutive root/bass bar-pair tokens with the Aria. | terminal component | Active component | 0.056 | 0.064 | 0.071 | 0.100 |
| `strict_root_bass_fourgram_weighted_jaccard` | Weighted Jaccard overlap of four-bar root/bass progression tokens with the Aria. | terminal component | Active component | 0.000 | 0.001 | 0.003 | 0.011 |
| `strict_cadence_root_bass` | Root/bass match at phrase-ending cadence positions. | terminal component | Active component | 0.402 | 0.396 | 0.522 | 0.783 |
| `aria_chroma_full_hist` | Full-texture pitch-class distribution after key normalization. | terminal diagnostic | Logged only | 0.830 | 0.836 | 0.871 | 0.891 |
| `aria_chroma_bass_hist` | Bass pitch-class distribution after key normalization. | terminal diagnostic | Logged only | 0.739 | 0.731 | 0.779 | 0.824 |
| `aria_chroma_top_hist` | Top-voice pitch-class distribution after key normalization. | terminal diagnostic | Logged only | 0.745 | 0.744 | 0.746 | 0.787 |
| `aria_harmony_dtw_combined` | Older broad DTW mean over harmony, root, and bass sequences. | patch DTW diagnostic | Logged only | 0.772 | 0.776 | 0.788 | 0.799 |
| `aria_harmony_root_dtw` | Older broad DTW over inferred harmonic-root pitch classes. | patch DTW diagnostic | Logged only | 0.787 | 0.792 | 0.807 | 0.812 |
| `aria_harmony_bass_dtw` | Older broad DTW over inferred bass pitch classes. | patch DTW diagnostic | Logged only | 0.774 | 0.778 | 0.787 | 0.803 |
| `aria_harmony_aligned_root` | Older same-bar harmonic-root pitch-class match to the Aria. | patch diagnostic | Logged only | 0.169 | 0.176 | 0.202 | 0.222 |
| `aria_harmony_aligned_bass` | Older same-bar bass pitch-class match to the Aria. | patch diagnostic | Logged only | 0.165 | 0.180 | 0.169 | 0.257 |
| `aria_harmony_top_contour_dtw` | Coarse top-voice up/same/down contour aligned to the Aria. | patch DTW diagnostic | Logged only | 0.869 | 0.865 | 0.868 | 0.863 |

Even with this simple SFT setup, the generated samples move closer to the Aria
on several of these signals, but the picture is mixed across the full
similarity bundle. The active similarity reward is the strict symbolic
Aria-similarity aggregate normalized against base-model samples. I keep chroma
histograms, the older broader harmony DTW and same-bar alignment metrics,
CLaMP2 semantic similarity, chroma DTW, top-voice contour DTW, and density DTW
as diagnostics. Here epoch 0 is the base NotaGen-large model prompted only with
the metadata keywords, without the Aria in the prompt.

![SFT similarity metrics across epochs](docs/assets/sft_similarity_breakdown_all_metrics_with_gt.png)

At this point, around epochs 8-9, the model can already generate some
relatively decent melodies that sound Baroque and Bach-like, and that are also
similar to some of the variations, or at least contain clear echoes of the
theme.
This one is similar to the fifth variation:

https://github.com/user-attachments/assets/1c8b07e3-0f19-4afa-a38d-0aac39d53ce6

For comparison, this is the real fifth variation:

https://github.com/user-attachments/assets/0deb5544-7d33-4a89-ab0f-9d2363995b88

And this one is a bit more dreamy and free:

https://github.com/user-attachments/assets/eb4a3cf9-9998-4410-9401-e3708356828f

Both of them are still very unrefined, with harmonic problems where the initial
counterpoint becomes messy and loses a clear structure, but it is a beginning.

### RL through PPO

Something I have been curious about is whether reinforcement learning can push
the model toward these structural rules more directly. I am now experimenting
with [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347). The
setup is close to the standard RLHF loop: generate continuations from the
current policy, score them with automatic rewards, estimate advantages with a
value function, and update the policy with the clipped PPO objective.

The reward is a weighted sum of structural checks and Aria-similarity checks.
The active similarity side uses the strict symbolic Aria-similarity aggregate
defined above; chroma histograms and the older non-strict harmony DTW terms are
retained as diagnostics. On top of those similarity metrics, I also track
structural rewards for ABC validity, NotaGen line structure, meter consistency,
written musical bars, and repeat-expanded rendered bars. The table below lists
only active structural reward terms. Values are mean raw subreward scores
before applying the listed weight, except for the subtotal row, which is already
a weighted sum. Epoch 0 is the pretrained NotaGen-large model with the same
prompt/evaluation setup; GT is the mean over the real Goldberg variations under
the same scorer.

| Reward | Type | Description | Terminal / Patch | Weight | Epoch 0 | Epoch 1 | Epoch 8 | GT |
|---|---|---|---|---:|---:|---:|---:|---:|
| `completion_reward` | Structural | Target written score-measure count reached. | terminal | 0.250 | 1 | 0.967 | 1 | 0.900 |
| `expanded_completion_reward` | Structural | Target repeat-expanded/rendered score-measure count reached. | terminal | 0.250 | 0.500 | 0.417 | 0.583 | 0.900 |
| `parse_reward` | Structural | Graded ABC syntax/tokenizer/music21 parse quality. | terminal | 0.250 | 0.993 | 0.923 | 0.950 | 1 |
| `syntax_penalty_reward` | Structural | Fast malformed-syntax penalty; negative when triggered. | terminal | 0.250 | 0 | -0.100 | -0.083 | 0 |
| `countdown_reward` | Structural | NotaGen stream countdown `[r:i/j]` progression. | patch | 0.250 | 1 | 0.999 | 1 | 1 |
| `line_closure_reward` | Structural | Generated stream lines close syntactically. | patch | 0.250 | 1 | 0.999 | 1 | 1 |
| `bar_token_reward` | Structural | Stream lines contain bar/repeat tokens. | patch | 0.100 | 1 | 1 | 1 | 1 |
| `note_bearing_line_reward` | Structural | Stream lines contain at least one ABC note in some voice, so rest-only generated lines lose structural credit. | patch | 0.250 | 1 | 0.990 | 0.997 | 1 |
| `meter_alignment_reward` | Structural | Populated voices align with expected meter. | patch | 0.750 | 0.969 | 0.927 | 0.974 | 0.987 |
| `meter_duration_closeness_reward` | Structural | Bar durations are close to expected meter. | patch | 0.750 | 0.996 | 0.964 | 0.997 | 0.994 |
| `bar_meter_consistency_reward` | Structural | Voices inside a bar are mutually meter-consistent. | patch | 0.750 | 0.970 | 0.939 | 0.997 | 0.997 |
| `bar_count_reward` | Structural | Written score measures close to target 32. | patch marginal | 1 | 1 | 0.999 | 1 | 0.902 |
| `expanded_bar_count_reward` | Structural | Repeat-expanded/rendered score measures close to target 64. | patch marginal | 1 | 0.764 | 0.757 | 0.834 | 0.899 |
| `voice_declaration_reward` | Structural | Generated voices are declared in the header. | patch | 1 | 1 | 1 | 1 | 1 |
| `score_voice_reward` | Structural | Generated voices match the `%%score` voice set. | patch | 0.500 | 1 | 1 | 1 | 1 |
| `structural_total_reward` | Subtotal | Weighted structural subtotal. | mixed | 1 | 6.939 | 6.777 | 7.023 | 7.084 |

In the current PPO implementation, rewards and value targets are computed per
NotaGen patch for tractability, while the policy loss is reduced over generated
character tokens. Patch-level advantages are repeated over the generated tokens
inside each patch, prompt tokens are excluded from the loss, and Generalized
Advantage Estimation controls how rewards are propagated through the trajectory.
An exact full-vocabulary KL against the frozen SFT/reference policy can be
logged or used as a penalty.

Some qualitative renders from earlier RL runs are still useful as examples of
what this reward-driven stage is trying to improve, independent of whether the
policy update is done with GRPO or PPO. For example, this fixed render from
step 171 has a clear two-part form, and each part splits again into two
subparts, which is close to the phrase layout of the Aria:

https://github.com/user-attachments/assets/56f300b6-9181-48f5-94d3-d57fa0cb6746

Another good example is this high-reward render from step 54:

https://github.com/user-attachments/assets/87f1db78-1f7f-4797-9659-eb4514aa9d66

A more imperfect one, with some mistakes but where the Aria theme is easier to
identify, is this mid-range render from step 253:

https://github.com/user-attachments/assets/def7095a-93e8-4cf1-8791-d05a1c3d880d

Making NotaGen work with PPO is already a challenge because its decoding is
hierarchical: patch-level generation, token-level generation, replayed
log-probability scoring, and value prediction all have to stay aligned. Long
32-bar continuations can also contain many token events, so this is still a
WIP.

### Next steps

There are many follow-ups that come to mind. Continuing with this setup, maybe
the most interesting one is to build a more structural embedding, something
that takes all these rules into account and can be used as a pure similarity
reward, in the same spirit that [NotaGen](https://arxiv.org/abs/2502.18008)
uses a semantic embedding.

That said, I am not even sure that post-training is the best place to force all
of this. Ideally, the base model should be conditioned on a concrete melody,
not only on generic metadata like author and period. That would make it possible
to learn these structural representations for any piece, instead of adding them
after the fact for this particular setup.
