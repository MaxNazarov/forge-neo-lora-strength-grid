# LoRA Strength Grid for Forge Neo

A script for [Forge Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo) that renders the same prompt with each selected LoRA at one or more strengths and builds a comparison grid.

Based on [Lora Queue Helper](https://github.com/Yinzo/sd-webui-Lora-queue-helper) by **Yinzo**. The original commit history is kept in this repository.

## What it does

```
            LoRA A    LoRA B    LoRA C      ← selected LoRAs
  No LoRA │   img       img       img       ← optional baseline
  0.4     │   img       img       img
  0.7     │   img       img       img       ← one row per strength
  1.0     │   img       img       img
```

- **Several LoRAs, several strengths:** each LoRA is applied on its own, and the grid shows LoRAs across and strengths down. With a single strength you get one row.
- **One LoRA, several strengths:** the strengths run across instead.
- **No strengths entered:** each LoRA is rendered at its preferred weight, as in the original Lora Queue Helper.

## Install

**Extensions** → **Install from URL** → paste `https://github.com/MaxNazarov/forge-neo-lora-strength-grid` → **Install**, then restart the WebUI.

## How to use

1. In txt2img or img2img, open the **Script** menu at the bottom and choose **LoRA Strength Grid**.
2. Select the **folders**, then tick the **LoRAs** to test.
3. Enter **Strengths** (or leave it empty to use each LoRA's preferred weight):

   | Input | Result |
   |---|---|
   | `0.4, 0.7, 1.0` | a list |
   | `0.2-1.0 (+0.2)` | a range with a step: 0.2, 0.4, 0.6, 0.8, 1.0 |
   | `1-0 (-0.25)` | a range going down |
   | `0-1 [5]` | a range split into 5 values: 0, 0.25, 0.5, 0.75, 1 |

   Items can be mixed, e.g. `0, 0.3, 0.5-1 (+0.25)`. The line below the field shows how many generations will run.
4. Click **Generate**.

## Options

| Option | What it does |
|---|---|
| **Add baseline (no LoRA)** | Adds a "No LoRA" row (or column), rendered once, for comparison. |
| **Add activation text** | Adds the LoRA's activation text (from its LoRA settings) after the tag. |
| **Swap axes** | Puts strengths across and LoRAs down, or the reverse for a single LoRA. |
| **Seed** | *Same seed for every image* (default, best for comparisons), *New seed for each LoRA* (same seed across that LoRA's strengths), or *Random seed for every image*. |
| **LoRA tag position** | Adds the tag at the start or the end of the prompt. |
| **Save grid image** | Builds the grid, shows it and saves it to the grids folder. With batch count or batch size above 1, one grid is made per image. |
| **Add LoRA name & strength to file names** | Saves single images as `...-LoRAname-0.7.png`. Turn this off to let the WebUI save images in its usual way (see the note below). |
| **Grid labels** | *Axis headers* (LoRA names and strengths along the edges, like X/Y/Z plot), *Text on each image*, or *None*. |
| **Flat grid and label font** | Row count for the grid used when Strengths is empty, plus the font for *Text on each image*. |

Every image also gets a `LoRA grid: <name> @ <strength>` entry in its generation info.

## Notes

- Tags for the tested LoRAs that are already in your prompt (or hires prompt) are removed first, so a LoRA isn't applied twice. Other LoRA tags are left as they are.
- If you set a hires prompt, the LoRA tag is added to it too.
- **Interrupt** stops the queue. The grid is still made, with black cells for images that weren't generated.
- With *Add LoRA name & strength to file names* on, the script saves the final images itself. That means the WebUI's optional extra files (such as "before highres fix" or "before face restoration" copies) aren't saved. Turn the option off if you need them.
- LoRA tags use the same name Forge uses (the "When adding to prompt, refer to Lora by" setting).
- Supported files: `.safetensors`, `.pt` and `.ckpt` in the folder set by `--lora-dir`. Extra `--lora-dirs` folders aren't listed yet.
- Preferred weight and activation text come from the LoRA's `.json` settings file, which the WebUI writes when you edit a LoRA's settings card.

## License

MIT. See [LICENSE](LICENSE). Original work © 2023 Yinz.
