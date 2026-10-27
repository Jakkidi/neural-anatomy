# Third-party material

- `vendor/anatomy/` and `src/ui/kit.css` are from **Anatomy** by Ryan
  (<https://github.com/wheresryan22/anatomy>), MIT licence, copied unchanged
  apart from `kit.css`, which is `iso.css` without its two colour blocks. The
  commit they were taken from is in `vendor/anatomy/COMMIT`; the licence is
  `vendor/anatomy/LICENSE`.
- `assets/gpt2/vocab.json` and `assets/gpt2/merges.txt` are GPT-2's vocabulary
  and merge table, published by OpenAI with GPT-2 under a modified MIT licence
  (<https://github.com/openai/gpt-2>). They are the same files Hugging Face
  serves as `vocab.json` and `merges.txt` for `openai-community/gpt2`.
- The GPT-2 weights are not in this repository. The page, and the desktop app, download them from
  <https://huggingface.co/openai-community/gpt2>; they are OpenAI's, under the
  same licence.
