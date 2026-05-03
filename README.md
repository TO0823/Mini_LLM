# Mini_LLM: A Character-Level Shakespearean Transformer

A from-scratch implementation of a GPT-style Transformer, built for **practice and exploring the architecture of Large Language Models.** This project follows the "Let's Build GPT" architecture to generate Shakespeare-esque dialogue using PyTorch.

## The Mission
This project was born out of a desire to understand the "magic" behind Transformers. It’s not just about running code; it’s about understanding the communication (Self-Attention) and computation (Feed-Forward) that allow machines to mimic human text.

**Note:** This is a **practice project** created while learning the fundamentals of deep learning and sequence modeling.

## Tech Stack & Specs
* **Framework:** PyTorch
* **Architecture:** Decoder-only Transformer (GPT)
* **Hardware:** Optimized for **AMD Radeon RX 7800 XT** (ROCm/HIP)
* **Parameters:** ~10M
* **Layers:** 6 blocks of Multi-Head Attention & Feed-Forward networks

## Key Features
* **Multi-Head Self-Attention:** 8 heads working in parallel to find context.
* **Positional Embeddings:** Giving the model a sense of "time" and order.
* **Residual Connections & LayerNorm:** Keeping the gradients healthy during deep training.
* **Dropout Layering:** Aggressive regularization to prevent the model from just memorizing the training set.

## Lessons Learned (The Hard Way)
* **AMD on Linux:** Successfully configured ROCm and HIP overrides (`HSA_OVERRIDE_GFX_VERSION=11.0.0`) to run PyTorch on RDNA 3 hardware.
* **Overfitting:** Observed the "U-Curve" of validation loss and implemented Dropout and Weight Decay to force the model to generalize.
* **Matrix Math:** Deep-dived into the dimensions of $Batch \times Time \times Channel$ and why $n\_embd$ must be divisible by $n\_head$.

## Sample Output
> **KING EDWARD IV:**
> Lords of Exeter, father, the tyranny
> Will appeal thee for a dear bed;
> Who being shares this shameless maims say
> And five thus, wiser Tybalt, Japet...

## Setup
If you're running this on an AMD GPU, make sure your environment is set up with:
```bash
export HSA_OVERRIDE_GFX_VERSION=11.0.0
```
run:
```bash
python train.py
```
