"""
LoRA Strength Grid for Forge Neo.

Renders the same prompt once per selected LoRA and per strength, then stitches
the results into a LoRA x strength comparison grid.

Based on "Lora Queue Helper" by Yinzo
(https://github.com/Yinzo/sd-webui-Lora-queue-helper), MIT License.
"""
import copy
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import gradio as gr
from PIL import Image, ImageDraw, ImageFont

from modules import images, scripts, sd_models
from modules.paths_internal import roboto_ttf_file
from modules.processing import Processed, fix_seed, process_images
from modules.shared import cmd_opts, opts, state

LOG_PREFIX = "[LoRA Strength Grid]"
LORA_EXTENSIONS = (".safetensors", ".pt", ".ckpt")
MAX_STRENGTHS = 50

SEED_SAME = "Same seed for every image"
SEED_PER_LORA = "New seed for each LoRA"
SEED_RANDOM = "Random seed for every image"
SEED_MODES = [SEED_SAME, SEED_PER_LORA, SEED_RANDOM]

LABELS_AXIS = "Axis headers"
LABELS_TEXT = "Text on each image"
LABELS_NONE = "None"
LABEL_MODES = [LABELS_AXIS, LABELS_TEXT, LABELS_NONE]

POSITION_PREPEND = "Prepend"
POSITION_APPEND = "Append"

BASELINE_LABEL = "No LoRA"
STRENGTHS_HELP = "Empty = each LoRA's preferred weight. Examples: 0.4, 0.7, 1.0  |  0.2-1.0 (+0.2)  |  0-1 [5]"

lora_dir = Path(cmd_opts.lora_dir).resolve()


# --------------------------------------------------------------------------
# LoRA folders and files
# --------------------------------------------------------------------------

def allowed_path(path) -> bool:
    try:
        return Path(path).resolve().is_relative_to(lora_dir)
    except (OSError, ValueError):
        return False


def get_base_path(is_use_custom_path, custom_path) -> Path:
    return lora_dir.joinpath(custom_path) if is_use_custom_path and custom_path else lora_dir


def is_lora_file(filename: str) -> bool:
    return filename.lower().endswith(LORA_EXTENSIONS)


def is_directory_contain_lora(path) -> bool:
    try:
        if allowed_path(path):
            return any(is_lora_file(f) for f in os.listdir(path))
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"{LOG_PREFIX} {e}")
    return False


def get_directories(base_path, include_root=True) -> list:
    directories = ["/"] if include_root else []
    try:
        if allowed_path(base_path):
            for entry in sorted(os.listdir(base_path), key=str.lower):
                full_path = os.path.join(base_path, entry)
                if os.path.isdir(full_path):
                    if is_directory_contain_lora(full_path):
                        directories.append(entry)
                    nested = get_directories(full_path, include_root=False)
                    directories.extend(os.path.join(entry, d) for d in nested)
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"{LOG_PREFIX} {e}")
    return directories


def list_lora_files(base_path, directories) -> list:
    """All LoRA files in the chosen directories, in a stable order, without duplicates."""
    files, seen = [], set()
    base_path = Path(base_path)
    for directory in directories or []:
        directory = base_path if directory == "/" else base_path.joinpath(directory)
        if not allowed_path(directory) or not directory.is_dir():
            continue
        for filename in sorted(os.listdir(directory), key=str.lower):
            full = directory.joinpath(filename)
            if is_lora_file(filename) and full.is_file() and full not in seen:
                seen.add(full)
                files.append(full)
    return files


def list_lora_names(base_path, directories) -> list:
    names = []
    for path in list_lora_files(base_path, directories):
        if path.stem not in names:
            names.append(path.stem)
    return names


def get_lora_name(path: Path) -> str:
    """The name to put inside <lora:...>, resolved the same way Forge's LoRA extension does."""
    try:
        import networks  # Forge's built-in LoRA extension

        entry = networks.available_networks.get(path.stem)
        if entry is not None and Path(entry.filename).resolve() == path.resolve():
            return entry.get_alias()
    except Exception:
        pass

    if opts.lora_preferred_name == "Filename" or path.suffix.lower() != ".safetensors":
        return path.stem
    try:
        metadata = sd_models.read_metadata_from_safetensors(str(path))
    except Exception:
        return path.stem
    name = metadata.get("ss_output_name")
    # Some training tools write Python's None as the string "None".
    if not name or name == "None":
        return path.stem
    return name


@dataclass
class LoraInfo:
    path: Path
    label: str  # filename, shown in the UI and on the grid
    tag_name: str  # name used inside <lora:...>
    preferred_weight: float = 1.0
    activation_text: str = ""

    def tag(self, weight: float, with_activation: bool) -> str:
        text = f"<lora:{self.tag_name}:{format_strength(weight)}>"
        activation = self.activation_text.strip().strip(",").strip()
        if with_activation and activation:
            text += ", " + activation
        return text


def load_lora_info(path: Path) -> LoraInfo:
    """Reads preferred weight and activation text from the LoRA's .json sidecar (written by the WebUI's LoRA settings)."""
    weight, activation = 1.0, ""
    json_path = path.with_suffix(".json")
    if json_path.is_file():
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            activation = str(data.get("activation text", "") or "")
            weight = float(data.get("preferred weight", 1) or 1)
        except Exception as e:
            print(f"{LOG_PREFIX} Could not read {json_path.name}: {e}")
    if weight == 0:  # 0 means "not set" in the WebUI
        weight = 1.0
    return LoraInfo(path=path, label=path.stem, tag_name=get_lora_name(path), preferred_weight=weight, activation_text=activation)


# --------------------------------------------------------------------------
# Strengths and prompts
# --------------------------------------------------------------------------

_NUM = r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))"
RE_RANGE_STEP = re.compile(rf"{_NUM}\s*-\s*{_NUM}\s*\(\s*{_NUM}\s*\)")
RE_RANGE_COUNT = re.compile(rf"{_NUM}\s*-\s*{_NUM}\s*\[\s*(\d+)\s*\]")


def parse_strengths(text) -> list:
    """
    Parses a comma separated list of strengths. Each item is one of:
      0.7              a single value
      0.2-1.0 (+0.2)   a range with a step (end included)
      0-1 [5]          a range split into N evenly spaced values
    Raises ValueError with a readable message on bad input.
    """
    values = []
    for token in (text or "").split(","):
        token = token.strip()
        if not token:
            continue

        if m := RE_RANGE_STEP.fullmatch(token):
            start, end, step = (float(x) for x in m.groups())
            if step == 0 or (end - start) * step < 0:
                raise ValueError(f"'{token}': the step has to move from {start:g} towards {end:g}")
            count = int(math.floor((end - start) / step + 1e-9))
            values.extend(start + i * step for i in range(count + 1))
        elif m := RE_RANGE_COUNT.fullmatch(token):
            start, end, count = float(m.group(1)), float(m.group(2)), int(m.group(3))
            if count < 1:
                raise ValueError(f"'{token}': the count must be at least 1")
            if count == 1:
                values.append(start)
            else:
                values.extend(start + (end - start) * i / (count - 1) for i in range(count))
        else:
            try:
                values.append(float(token))
            except ValueError:
                raise ValueError(f"Can't read '{token}'. Use numbers like 0.5, a range with a step like 0.2-1.0 (+0.2), or a range with a count like 0-1 [5]") from None

        if len(values) > MAX_STRENGTHS:
            raise ValueError(f"Too many strengths (more than {MAX_STRENGTHS})")

    return [round(v, 4) + 0.0 for v in values]  # + 0.0 turns -0.0 into 0.0


def format_strength(value: float) -> str:
    return f"{value:g}"


def strip_lora_tags(prompt: str, names) -> str:
    """Removes <lora:name...> tags for the given names, so the tested LoRA is not applied twice."""
    names = sorted({n for n in names if n}, key=len, reverse=True)
    if not prompt or not names:
        return prompt
    pattern = re.compile(r"<lora:(?:" + "|".join(re.escape(n) for n in names) + r")(?::[^>]*)?>", re.IGNORECASE)
    stripped = pattern.sub("", prompt)
    if stripped == prompt:
        return prompt
    stripped = re.sub(r"[ \t]*,(?:[ \t]*,)+", ",", stripped)  # empty slots left by removed tags
    return stripped.strip().strip(",").strip()


def add_to_prompt(prompt: str, addition: str, position: str) -> str:
    prompt = (prompt or "").strip()
    addition = (addition or "").strip().strip(",").strip()
    if not addition:
        return prompt
    if not prompt.strip(", "):
        return addition
    if position == POSITION_APPEND:
        return f"{prompt.rstrip(', ')}, {addition}"
    return f"{addition}, {prompt.lstrip(', ')}"


# --------------------------------------------------------------------------
# Grid layout
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Cell:
    lora: Optional[LoraInfo]  # None = baseline (no LoRA)
    strength: Optional[float]

    @property
    def key(self):
        return ("baseline",) if self.lora is None else (str(self.lora.path), self.strength)

    @property
    def label(self) -> str:
        if self.lora is None:
            return BASELINE_LABEL
        return f"{self.lora.label} @ {format_strength(self.strength)}"


BASELINE_CELL = Cell(None, None)


def build_jobs(loras, strengths, baseline) -> list:
    """Cells to render, in generation order. Without strengths every LoRA uses its preferred weight."""
    jobs = [BASELINE_CELL] if baseline else []
    if strengths:
        jobs += [Cell(lora, s) for lora in loras for s in strengths]
    else:
        jobs += [Cell(lora, lora.preferred_weight) for lora in loras]
    return jobs


def build_table(loras, strengths, baseline, swap_axes):
    """
    Layout of the strength x LoRA table: (row_labels, col_labels, cells[row][col]).
    By default LoRAs run along X and strengths along Y. With a single LoRA the
    strengths run along X instead. "Swap axes" flips whichever layout applies.
    """
    strength_axis = ([None] if baseline else []) + list(strengths)

    strengths_on_x = len(loras) == 1 and len(strengths) > 1
    if swap_axes:
        strengths_on_x = not strengths_on_x

    def strength_label(s):
        return BASELINE_LABEL if s is None else format_strength(s)

    def cell(lora, s):
        return BASELINE_CELL if s is None else Cell(lora, s)

    if strengths_on_x:
        row_labels = [lora.label for lora in loras]
        col_labels = [strength_label(s) for s in strength_axis]
        cells = [[cell(lora, s) for s in strength_axis] for lora in loras]
    else:
        row_labels = [strength_label(s) for s in strength_axis]
        col_labels = [lora.label for lora in loras]
        cells = [[cell(lora, s) for lora in loras] for s in strength_axis]
    return row_labels, col_labels, cells


def auto_row_count(n_images: int) -> int:
    """Rows for a roughly 4:3 flat grid."""
    return max(1, round(3.0 * math.sqrt(n_images / 12.0)))


def resolve_font_path(font_path) -> Optional[str]:
    """Accepts an absolute path or a path relative to the LoRA directory."""
    if not font_path or not str(font_path).strip():
        return None
    font_path = str(font_path).strip()
    for candidate in (Path(font_path), lora_dir.joinpath(font_path)):
        if candidate.is_file():
            return str(candidate)
    print(f"{LOG_PREFIX} Font file not found: {font_path}. Using the default font.")
    return None


def draw_text_with_stroke(draw, text, position, font, text_color, stroke_color, stroke_width):
    x, y = position
    for dx in range(-stroke_width, stroke_width + 1):
        for dy in range(-stroke_width, stroke_width + 1):
            draw.text((x + dx, y + dy), text, font=font, fill=stroke_color)
    draw.text((x, y), text, font=font, fill=text_color)


def image_grid_with_text(imgs, texts, rows, font_path=None, font_size=20, text_color="#FFFFFF", stroke_color="#000000", stroke_width=2):
    cols = math.ceil(len(imgs) / rows)
    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h), color="black")
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))

    font = None
    if font_path:
        try:
            font = ImageFont.truetype(font_path, font_size)
        except OSError as e:
            print(f"{LOG_PREFIX} Error loading font {font_path}: {e}. Using the default font.")
    if font is None:
        font = ImageFont.truetype(roboto_ttf_file, font_size)

    draw = ImageDraw.Draw(grid)
    for i, text in enumerate(texts):
        x = (i % cols) * w
        y = (i // cols) * h
        draw_text_with_stroke(draw, text, (x + 5, y + 5), font, text_color, stroke_color, stroke_width)
    return grid


def make_grid(imgs, rows, label_mode, cell_texts, row_labels=None, col_labels=None, font=None):
    """
    imgs are in row-major order. Axis headers need row and column labels;
    without them (flat grid) "Axis headers" falls back to text on each image.
    """
    font = font or {}
    if label_mode == LABELS_AXIS and row_labels is not None and col_labels is not None:
        grid = images.image_grid(imgs, rows=rows)
        w = max(img.width for img in imgs)
        h = max(img.height for img in imgs)
        hor_texts = [[images.GridAnnotation(t)] for t in col_labels]
        ver_texts = [[images.GridAnnotation(t)] for t in row_labels]
        return images.draw_grid_annotations(grid, w, h, hor_texts, ver_texts)
    if label_mode == LABELS_NONE:
        return images.image_grid(imgs, rows=rows)
    return image_grid_with_text(imgs, cell_texts, rows=rows, **font)


def describe_job_count(selected_loras, strengths_text, baseline) -> str:
    n_loras = len(selected_loras or [])
    if n_loras == 0:
        return "Select at least one LoRA."
    try:
        strengths = parse_strengths(strengths_text)
    except ValueError as e:
        return f"⚠️ {e}"

    lora_word = "LoRA" if n_loras == 1 else "LoRAs"
    if strengths:
        cells = n_loras * len(strengths)
        detail = f"{n_loras} {lora_word} × {len(strengths)} strength{'s' if len(strengths) != 1 else ''}"
    else:
        cells = n_loras
        detail = f"{n_loras} {lora_word} at preferred weight"
    if baseline:
        cells += 1
        detail += " + 1 baseline"
    return f"**{cells} generation{'s' if cells != 1 else ''}** ({detail}), each × batch count × batch size."


# --------------------------------------------------------------------------
# Script
# --------------------------------------------------------------------------

class Script(scripts.Script):
    def title(self):
        return "LoRA Strength Grid"

    def ui(self, is_img2img):
        def show_dir_textbox(enabled, custom_path):
            dirs = get_directories(get_base_path(enabled, custom_path))
            return gr.update(visible=enabled), gr.update(choices=dirs, value=[])

        def update_dirs(is_use_custom_path, custom_path):
            return gr.update(choices=get_directories(get_base_path(is_use_custom_path, custom_path)), value=[])

        def select_all_dirs(is_use_custom_path, custom_path):
            return gr.update(value=get_directories(get_base_path(is_use_custom_path, custom_path)))

        def update_loras(current_selected, is_use_custom_path, custom_path, directories):
            all_loras = list_lora_names(get_base_path(is_use_custom_path, custom_path), directories)
            visible = len(all_loras) > 0
            kept = [lora for lora in all_loras if lora in (current_selected or [])]
            return gr.update(choices=all_loras, value=kept, visible=visible), gr.update(visible=visible), gr.update(visible=visible)

        def select_all_lora(is_use_custom_path, custom_path, directories):
            return gr.update(value=list_lora_names(get_base_path(is_use_custom_path, custom_path), directories))

        with gr.Column():
            base_dir_checkbox = gr.Checkbox(label="Use custom LoRA path", value=False, elem_id=self.elem_id("base_dir_checkbox"))
            base_dir_textbox = gr.Textbox(label="LoRA directory", placeholder="Path relative to the LoRA directory (set with --lora-dir at startup)", visible=False, elem_id=self.elem_id("base_dir_textbox"))

            all_dirs = get_directories(lora_dir)
            directory_checkboxes = gr.CheckboxGroup(label="Select directory", choices=all_dirs, value=["/"], elem_id=self.elem_id("directory_checkboxes"))
            with gr.Row():
                select_all_dirs_button = gr.Button("All")
                deselect_all_dirs_button = gr.Button("Clear")

            startup_loras = list_lora_names(lora_dir, ["/"])
            lora_checkboxes = gr.CheckboxGroup(label="LoRA", choices=startup_loras, value=[], visible=len(startup_loras) > 0, elem_id=self.elem_id("lora_checkboxes"))
            with gr.Row():
                select_all_lora_button = gr.Button("All", visible=len(startup_loras) > 0)
                deselect_all_lora_button = gr.Button("Clear", visible=len(startup_loras) > 0)

            strengths_textbox = gr.Textbox(label="Strengths", placeholder=STRENGTHS_HELP, info=STRENGTHS_HELP, elem_id=self.elem_id("strengths"))
            job_info = gr.Markdown(describe_job_count([], "", False))

            with gr.Row():
                include_baseline = gr.Checkbox(label="Add baseline (no LoRA)", value=False, elem_id=self.elem_id("include_baseline"))
                include_activation = gr.Checkbox(label="Add activation text", value=True, elem_id=self.elem_id("include_activation"))
                swap_axes = gr.Checkbox(label="Swap axes", value=False, elem_id=self.elem_id("swap_axes"))

            with gr.Row():
                seed_mode = gr.Radio(label="Seed", choices=SEED_MODES, value=SEED_SAME, elem_id=self.elem_id("seed_mode"))
                lora_tags_position = gr.Radio(label="LoRA tag position", choices=[POSITION_PREPEND, POSITION_APPEND], value=POSITION_PREPEND, elem_id=self.elem_id("lora_tag_position"))

            with gr.Row():
                checkbox_save_grid = gr.Checkbox(label="Save grid image", value=True, elem_id=self.elem_id("checkbox_save_grid"))
                checkbox_name_files = gr.Checkbox(label="Add LoRA name & strength to file names", value=True, elem_id=self.elem_id("checkbox_name_files"))
            grid_labels = gr.Radio(label="Grid labels", choices=LABEL_MODES, value=LABELS_AXIS, elem_id=self.elem_id("grid_labels"))

            with gr.Accordion("Flat grid and label font", open=False):
                gr.Markdown("Row settings apply only when **Strengths** is empty. Font settings apply to *Text on each image*.")
                with gr.Row():
                    checkbox_auto_row_number = gr.Checkbox(label="Auto row number", value=True, elem_id=self.elem_id("checkbox_auto_row_number"))
                    grid_row_number = gr.Number(label="Grid row number", value=1, precision=0, interactive=False, elem_id=self.elem_id("grid_row_number"))
                with gr.Row():
                    font_path = gr.Textbox(label="Custom font path", placeholder="Path to a .ttf file (absolute, or relative to the LoRA directory)", elem_id=self.elem_id("font_path"))
                    font_size = gr.Number(label="Font size", value=20, precision=0, elem_id=self.elem_id("font_size"))
                with gr.Row():
                    font_color = gr.Textbox(label="Font color", value="#FFFFFF", elem_id=self.elem_id("font_color"))
                    stroke_color = gr.Textbox(label="Stroke color", value="#000000", elem_id=self.elem_id("stroke_color"))
                    stroke_width = gr.Number(label="Stroke width", value=2, precision=0, elem_id=self.elem_id("stroke_width"))

            base_dir_checkbox.change(fn=show_dir_textbox, inputs=[base_dir_checkbox, base_dir_textbox], outputs=[base_dir_textbox, directory_checkboxes])
            base_dir_textbox.change(fn=update_dirs, inputs=[base_dir_checkbox, base_dir_textbox], outputs=[directory_checkboxes])
            directory_checkboxes.change(fn=update_loras, inputs=[lora_checkboxes, base_dir_checkbox, base_dir_textbox, directory_checkboxes], outputs=[lora_checkboxes, select_all_lora_button, deselect_all_lora_button])
            select_all_dirs_button.click(fn=select_all_dirs, inputs=[base_dir_checkbox, base_dir_textbox], outputs=directory_checkboxes)
            deselect_all_dirs_button.click(fn=lambda: gr.update(value=[]), inputs=None, outputs=directory_checkboxes)
            select_all_lora_button.click(fn=select_all_lora, inputs=[base_dir_checkbox, base_dir_textbox, directory_checkboxes], outputs=lora_checkboxes)
            deselect_all_lora_button.click(fn=lambda: gr.update(value=[]), inputs=None, outputs=lora_checkboxes)
            checkbox_auto_row_number.change(fn=lambda auto: gr.update(interactive=not auto), inputs=[checkbox_auto_row_number], outputs=grid_row_number)

            for component in (lora_checkboxes, strengths_textbox, include_baseline):
                component.change(fn=describe_job_count, inputs=[lora_checkboxes, strengths_textbox, include_baseline], outputs=job_info, show_progress=False)

        return [
            base_dir_checkbox, base_dir_textbox, directory_checkboxes, lora_checkboxes,
            strengths_textbox, include_baseline, include_activation, swap_axes,
            seed_mode, lora_tags_position, checkbox_save_grid, checkbox_name_files, grid_labels,
            checkbox_auto_row_number, grid_row_number, font_path, font_size, font_color, stroke_color, stroke_width,
        ]

    def run(self, p, is_use_custom_path, custom_path, directories, selected_loras,
            strengths_text, include_baseline, include_activation, swap_axes,
            seed_mode, lora_tags_position, is_save_grid, is_name_files, grid_labels,
            is_auto_row_number, row_number, font_path, font_size, font_color, stroke_color, stroke_width):
        if not selected_loras:
            return process_images(p)

        strengths = parse_strengths(strengths_text)  # raises a readable error on bad input

        base_path = get_base_path(is_use_custom_path, custom_path)
        loras = [load_lora_info(path) for path in list_lora_files(base_path, directories) if path.stem in selected_loras]
        if not loras:
            print(f"{LOG_PREFIX} None of the selected LoRAs were found; generating normally.")
            return process_images(p)

        # The tested LoRAs must not also be applied by tags already in the prompt.
        strip_names = {n for lora in loras for n in (lora.label, lora.tag_name)}
        base_prompt = strip_lora_tags(p.prompt, strip_names)
        base_hr_prompt = strip_lora_tags(getattr(p, "hr_prompt", "") or "", strip_names)

        jobs = build_jobs(loras, strengths, include_baseline)

        if seed_mode != SEED_RANDOM:
            fix_seed(p)
        lora_index = {str(lora.path): i for i, lora in enumerate(loras)}
        seed_step = p.batch_size * p.n_iter

        def job_seed(cell):
            if seed_mode == SEED_SAME:
                return p.seed
            if seed_mode == SEED_PER_LORA:
                index = 0 if cell.lora is None else lora_index[str(cell.lora.path)]
                return p.seed + index * seed_step
            return None  # random: keep whatever the user set (-1 = new random seed per generation)

        save_samples_manually = is_name_files and opts.samples_save and not p.do_not_save_samples
        p.do_not_save_grid = True  # the WebUI's own per-job grid is not useful here
        state.job_count = len(jobs) * p.n_iter

        results = {}  # cell key -> list of (image, infotext, prompt, seed)
        for job_no, cell in enumerate(jobs):
            if state.interrupted or state.stopping_generation:
                print(f"{LOG_PREFIX} Interrupted after {job_no} of {len(jobs)} generations.")
                break
            if cell.key in results:
                continue
            state.job = f"{job_no + 1} out of {len(jobs)}: {cell.label}"

            copy_p = copy.copy(p)
            copy_p.extra_generation_params = dict(p.extra_generation_params)
            if cell.lora is None:
                copy_p.prompt = base_prompt
                if base_hr_prompt:
                    copy_p.hr_prompt = base_hr_prompt
            else:
                tag = cell.lora.tag(cell.strength, include_activation)
                copy_p.prompt = add_to_prompt(base_prompt, tag, lora_tags_position)
                if base_hr_prompt:
                    copy_p.hr_prompt = add_to_prompt(base_hr_prompt, tag, lora_tags_position)
            copy_p.extra_generation_params["LoRA grid"] = cell.label

            seed = job_seed(cell)
            if seed is not None:
                copy_p.seed = seed
            if save_samples_manually:
                copy_p.do_not_save_samples = True

            proc = process_images(copy_p)

            start = proc.index_of_first_image
            count = p.batch_size * p.n_iter
            cell_images = proc.images[start:start + count]
            cell_results = []
            for i, img in enumerate(cell_images):
                infotext = proc.infotexts[start + i] if start + i < len(proc.infotexts) else proc.info
                prompt = proc.all_prompts[i] if i < len(proc.all_prompts) else copy_p.prompt
                image_seed = proc.all_seeds[i] if i < len(proc.all_seeds) else proc.seed
                cell_results.append((img, infotext, prompt, image_seed))
            results[cell.key] = cell_results

            if save_samples_manually and cell_results and (opts.save_incomplete_images or not (state.interrupted or state.skipped)):
                suffix = "-" + images.sanitize_filename_part(cell.label.replace(" @ ", "-"), replace_spaces=True)
                for img, infotext, prompt, image_seed in cell_results:
                    images.save_image(img, p.outpath_samples, "", image_seed, prompt, opts.samples_format, info=infotext, p=copy_p, suffix=suffix)

        # Individual images in generation order.
        out_images, out_infotexts, out_prompts, out_seeds = [], [], [], []
        for key in dict.fromkeys(cell.key for cell in jobs):
            for img, infotext, prompt, image_seed in results.get(key, []):
                out_images.append(img)
                out_infotexts.append(infotext)
                out_prompts.append(prompt)
                out_seeds.append(image_seed)

        if not out_images:
            return Processed(p, [], p.seed, "")

        grids = []
        if is_save_grid and len(jobs) > 1:
            grids = self.make_grids(
                results, loras, strengths, include_baseline, swap_axes, jobs, grid_labels,
                is_auto_row_number, row_number,
                font=dict(font_path=resolve_font_path(font_path), font_size=int(font_size or 20), text_color=font_color or "#FFFFFF", stroke_color=stroke_color or "#000000", stroke_width=int(stroke_width or 0)),
            )

        grid_info = "\n".join([
            f"Prompt: {base_prompt or 'Empty'}",
            f"LoRAs: {', '.join(lora.label for lora in loras)}",
            f"Strengths: {', '.join(format_strength(s) for s in strengths) if strengths else 'preferred weight'}",
            f"Seed: {out_seeds[0]}",
        ])
        for grid in grids:
            images.save_image(grid, p.outpath_grids, "lora_grid", seed=out_seeds[0], prompt=base_prompt, extension=opts.grid_format, info=grid_info, grid=True, p=p)

        return Processed(
            p,
            grids + out_images,
            seed=out_seeds[0],
            info=out_infotexts[0],
            all_prompts=[base_prompt] * len(grids) + out_prompts,
            all_seeds=[out_seeds[0]] * len(grids) + out_seeds,
            infotexts=[grid_info] * len(grids) + out_infotexts,
            index_of_first_image=len(grids),
        )

    @staticmethod
    def make_grids(results, loras, strengths, baseline, swap_axes, jobs, label_mode, is_auto_row_number, row_number, font):
        """One grid per image of a cell (batch count × batch size). Missing cells (interrupted) are left black."""
        per_cell = max((len(r) for r in results.values()), default=0)
        reference = next((r[0][0] for r in results.values() if r), None)
        if per_cell == 0 or reference is None:
            return []
        blank = Image.new("RGB", reference.size, "black")

        def image_for(cell, k):
            cell_results = results.get(cell.key, [])
            return cell_results[k][0] if k < len(cell_results) else blank

        grids = []
        for k in range(per_cell):
            if strengths:
                row_labels, col_labels, cells = build_table(loras, strengths, baseline, swap_axes)
                flat_cells = [cell for row in cells for cell in row]
                imgs = [image_for(cell, k) for cell in flat_cells]
                grids.append(make_grid(imgs, len(cells), label_mode, [c.label for c in flat_cells], row_labels, col_labels, font))
            else:
                rows = auto_row_count(len(jobs)) if is_auto_row_number else max(1, int(row_number or 1))
                rows = min(rows, len(jobs))
                imgs = [image_for(cell, k) for cell in jobs]
                texts = [BASELINE_LABEL if c.lora is None else c.lora.label for c in jobs]
                grids.append(make_grid(imgs, rows, label_mode, texts, font=font))
        return grids
