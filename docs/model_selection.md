# Pretrained Model Selection

## Recommendation

Start with **MuPT-v1-8192-190M**:

https://huggingface.co/m-a-p/MuPT-v1-8192-190M

This is the best first model to play with because it is:

- open source under Apache 2.0;
- used as a baseline in the NotaGen paper;
- a standard Hugging Face `LlamaForCausalLM` model;
- small enough to run locally compared with 500M+ or 1B+ models;
- trained for symbolic music generation;
- outputting ABC-like symbolic music, which we can parse and score with rules.

The main downside is that MuPT uses BPE over SMT/interleaved ABC notation. The
final output is symbolic text, but BPE is not as clean for rule-aware generation
as a music-aware tokenizer or NotaGen's character/bar-stream representation.

## Candidate Comparison

| Model | Format | Size | Integration | Fit for our project |
| --- | --- | ---: | --- | --- |
| `m-a-p/MuPT-v1-8192-190M` | SMT/interleaved ABC + BPE | 190M | Standard HF Transformers | Best first choice |
| `ElectricAlexis/NotaGen` | ABC + bar-stream patching | 110M to 516M | Custom code/weights | Best conceptual match, more setup |
| `skytnt/midi-model-tv2o-medium` | MIDI event tokens | 234M | Custom model code | Good MIDI baseline, weaker for SATB rules |
| `sander-wood/tunesformer` | ABC melody | GPT-2 style | Standard-ish HF / custom scripts | Easy smoke test, but Irish monophonic domain |

## Why Not Start With NotaGen?

NotaGen is closest to the project conceptually: symbolic classical sheet music,
ABC notation, bar-stream patching, fine-tuning, and CLaMP-DPO. However, the
released weights are custom `.pth` checkpoints, the fine-tuned/RL checkpoints are
large, and local inference may require the official repository setup. Their
README says the local demo for NotaGen-X may require around 8GB of GPU memory.

It is worth using later, but MuPT is simpler for the first runnable pipeline.

## Why Not Start With SMART/MET?

SMART's actual base model is not the same as a released model from the paper.
The closest open model from the NotaGen comparison is SkyTNT's MET:

https://huggingface.co/skytnt/midi-model-tv2o-medium

This is useful if we want MIDI-event experiments. For our first rule-reward
project, ABC or explicit note/voice events are better because voice ranges, bar
duration, voice crossing, and counterpoint checks are easier to inspect.

## First Experiment

Use MuPT to generate ABC-like output from a simple prompt, then validate:

1. Does generation run locally?
2. Is the output parseable as ABC after post-processing?
3. Can we compute basic rewards: parse validity and bar duration?
4. Does the model output enough variation for later GRPO candidate ranking?

If MuPT output is too messy for rule rewards, use it only as a baseline and move
to a smaller custom model trained on Bach chorales with a simpler tokenizer.

## Trying NotaGen Directly

The NotaGen authors released the relevant checkpoints on Hugging Face:

- paper fine-tuned model:
  `weights_notagen_pretrain-finetune_p_size_16_p_length_1024_p_layers_c_layers_6_20_h_size_1280_lr_1e-05_batch_1.pth`
- paper post-RL model:
  `weights_notagen_pretrain-finetune-RL3_beta_0.1_lambda_10_p_size_16_p_length_1024_p_layers_20_c_layers_6_h_size_1280_lr_1e-06_batch_1.pth`
- newer demo model:
  `weights_notagenx_p_size_16_p_length_1024_p_layers_20_h_size_1280.pth`

Each of these checkpoints is about 6.19 GB. The official repository says local
NotaGen-X inference may require around 8 GB of GPU memory.

Official ways to try it:

1. Static web demo: https://electricalexis.github.io/notagen-demo/
   This shows generated examples, but is not the main interactive path.
2. Hugging Face Space: https://huggingface.co/spaces/ElectricAlexis/NotaGen
   This is the official interactive Gradio demo for NotaGen-X, but the Space was
   returning a runtime error / 503 when checked.
3. Colab notebook:
   https://colab.research.google.com/drive/1yJA1wG0fiwNeehdQxAUw56i4bTXzoVVv?usp=sharing
   The official README links this as a contributed way to launch a Gradio public
   URL.
4. Local Gradio app:
   clone `https://github.com/ElectricAlexis/NotaGen`, install its Python 3.10
   environment, download one checkpoint into `gradio/`, and run
   `python demo.py` from the `gradio/` folder.

For local experiments, the easiest path is to run their Gradio app first with
`NotaGen-X`. To test the paper's exact fine-tuned or RL checkpoint, change
`INFERENCE_WEIGHTS_PATH` in `gradio/config.py` to the downloaded `.pth` file.
The architecture settings in that config match the released large checkpoints:
patch size 16, patch length 1024, 20 patch-level layers, 6 character-level
layers, hidden size 1280.
