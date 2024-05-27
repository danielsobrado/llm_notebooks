# Abliterating LLMs: A Simple Technique to Bypass Refusal

Abliterating LLMs is a process that allows you to bypass the safety constraints of language models without the need for retraining. By identifying and removing a specific "refusal direction" in the model's activation space, you can make the model respond to harmful requests that it would normally refuse. Here's how it works:

## Step 1: Finding the Refusal Direction

First, we need to find the "refusal direction" in the model's activation space. To do this, we:

1. Prepare a set of harmful and harmless instructions.
2. Feed these instructions to the model and record the activations at a specific layer and position.
3. Calculate the average difference between the activations for harmful and harmless instructions.
4. Normalize this difference to get the "refusal direction."

Here's the code to do this:

```
# Prepare harmful and harmless instructions
harmful_toks = tokenize_instructions(harmful_instructions)
harmless_toks = tokenize_instructions(harmless_instructions)

# Run the model and record activations
harmful_acts = run_model(harmful_toks)
harmless_acts = run_model(harmless_toks)

# Calculate the refusal direction
layer = 14
pos = -1
harmful_mean = harmful_acts[layer][:, pos, :].mean(dim=0)
harmless_mean = harmless_acts[layer][:, pos, :].mean(dim=0)
refusal_dir = (harmful_mean - harmless_mean).normalized()
```

## Step 2: Removing the Refusal Direction

Now that we have the refusal direction, we can remove it from the model's activations during inference. This prevents the model from recognizing and refusing harmful requests.

Here's a function that removes the refusal direction from an activation:

```
def remove_refusal_direction(activation, refusal_dir):
    projection = einops.einsum(activation, refusal_dir, 'batch hidden, hidden -> batch')
    return activation - projection[:, None] * refusal_dir
```

We can apply this function to the model's activations at multiple layers using hooks:

```
def apply_hooks(model, refusal_dir):
    def hook(activation, hook):
        return remove_refusal_direction(activation, refusal_dir)
    
    for layer in model.layers:
        layer.register_forward_hook(hook)
```

## Step 3: Generating with the Modified Model

Finally, we can generate responses using the modified model:

```
apply_hooks(model, refusal_dir)
generated_text = model.generate(harmful_instruction)
```

The generated text will now include responses to harmful requests that the original model would have refused.

## Conclusion

By identifying and removing a specific direction in the model's activation space, we can make the model respond to harmful requests without the need for retraining.

This technique highlights the vulnerability of current approaches to making language models safe and aligned. It also opens up possibilities for better understanding how these models work internally.