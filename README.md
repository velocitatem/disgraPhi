# Dysgrafix
A tool to help people with dysgraphia. WE take a model like Qwen3-VL and create a main fork with a LoRA adapter that we initally train on:
https://fki.tic.heia-fr.ch/databases/iam-handwriting-database?

To give it a general ability to use handwritten text and bootstraps the model. (this model we freeze)

Then we create a personalization kit:
A 10–15 minute paper packet the user rewrites and photographs:

8 pangram lines + bigram/number mix + common names/addresses placeholders.
Printed page has QR corners for de‑skew and line guides; app auto‑crops lines.
Each line pairs to ground truth you already know; no manual labeling.

Collect 40–120 line‑level pairs per user for LoRA SFT on the VLM's decoder.


we need to define an app in streamlit and fastapi for inference and data handling. we create two training directories one for the global boostrap and then one for the fintuning on an individuals writing
