# requirements:
#   gradio>=4.0
#   gradio_image_annotation>=0.4
#   pillow

import gradio as gr
from gradio_image_annotation import image_annotator
from PIL import Image
import numpy as np, uuid, tempfile, os, json, shutil, datetime as dt

THEME = gr.themes.Soft(primary_hue="indigo", neutral_hue="slate")

SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "Please write this sentence in your normal handwriting.",
    "Numbers 0 1 2 3 4 5 6 7 8 9.",
    "Email: name@example.com",
    "Address: 221B Baker Street",
    "I have dysgraphia friendly loops.",
    "Cursive sample goes here.",
    "All caps SAMPLE LINE.",
    "Symbols: @ # $ % & * ( )",
    "Final line for calibration."
]

def _new_session():
    sid = str(uuid.uuid4())[:8]
    root = os.path.join(tempfile.gettempdir(), f"disgraf_{sid}")
    os.makedirs(os.path.join(root, "images"), exist_ok=True)
    os.makedirs(os.path.join(root, "crops"), exist_ok=True)
    return {"sid": sid, "root": root, "i": 0, "items": []}

def _to_pil(img):
    if isinstance(img, np.ndarray):
        return Image.fromarray(img)
    return img  # already PIL


def _instruction_text(state):
    idx = state["i"]
    sentence = SENTENCES[idx]
    total = len(SENTENCES)
    return (
        f"#### Step {idx + 1} of {total}\n"
        "Capture a clear photo of your handwriting and draw a tight bounding box around **just this sentence**:\n\n"
        f"> {sentence}"
    )


def _progress_badge(state):
    total = len(SENTENCES)
    current = state["i"]
    dots = ["[X]" if i < current else ("[-]" if i == current else "[ ]") for i in range(total)]
    return f"**Progress**  {''.join(dots)}  ({current}/{total} saved)"


def _history_table(state):
    rows = [
        [item['index'] + 1, item['text'], item['timestamp']]
        for item in state["items"]
    ]
    return rows


def _gallery_items(state):
    gallery = []
    for item in state["items"]:
        crop_path = os.path.join(state["root"], item["crop"])
        caption = f"#{item['index'] + 1}: {item['text']}"
        gallery.append((crop_path, caption))
    return gallery


def _status(message, level="info"):
    if not message:
        return ""
    icons = {"info": "ℹ️", "success": "✅", "warn": "⚠️", "error": "❌"}
    return message


def start():
    state = _new_session()
    instruction = _instruction_text(state)
    progress = _progress_badge(state)
    return (
        state,
        gr.update(value=None),
        instruction,
        progress,
        gr.update(visible=False, value=None),
        _status("Ready for your first sentence", "info"),
        _history_table(state),
        _gallery_items(state),
    )

def save_and_next(ann, state):
    if not ann or ann.get("image") is None:
        return (
            state,
            gr.update(),
            _instruction_text(state),
            _progress_badge(state),
            gr.update(),
            _status("Upload a photo before saving.", "warn"),
            _history_table(state),
            _gallery_items(state),
        )
    if not ann.get("boxes"):
        return (
            state,
            gr.update(),
            _instruction_text(state),
            _progress_badge(state),
            gr.update(),
            _status("Draw a box around the sentence before saving.", "warn"),
            _history_table(state),
            _gallery_items(state),
        )

    img_pil = _to_pil(ann["image"])
    W, H = img_pil.size
    box = ann["boxes"][0]
    xmin, ymin, xmax, ymax = [int(box[k]) for k in ("xmin","ymin","xmax","ymax")]
    idx = state["i"]; root = state["root"]

    img_path = os.path.join(root, "images", f"{idx:02d}.png")
    crop_path = os.path.join(root, "crops",  f"{idx:02d}.png")
    img_pil.save(img_path)
    img_pil.crop((xmin, ymin, xmax, ymax)).save(crop_path)

    state["items"].append({
        "index": idx,
        "text": SENTENCES[idx],
        "image": f"images/{idx:02d}.png",
        "crop":  f"crops/{idx:02d}.png",
        "bbox_xyxy": [xmin, ymin, xmax, ymax],
        "image_size_wh": [W, H],
        "timestamp": dt.datetime.utcnow().isoformat() + "Z"
    })
    state["i"] += 1

    if state["i"] >= len(SENTENCES):
        with open(os.path.join(root, "annotations.jsonl"), "w") as f:
            for it in state["items"]:
                f.write(json.dumps(it) + "\n")
        zip_path = shutil.make_archive(root, "zip", root)
        return (
            state,
            gr.update(value=None),
            "### Great work!\nAll sentences captured. Download your dataset below.",
            _progress_badge(state),
            gr.update(visible=True, value=zip_path),
            _status("Dataset packaged. You can restart for another user or close the page.", "success"),
            _history_table(state),
            _gallery_items(state),
        )
    else:
        instruction = _instruction_text(state)
        return (
            state,
            gr.update(value=None),
            instruction,
            _progress_badge(state),
            gr.update(visible=False, value=None),
            _status("Sentence saved. Move on to the next one!", "success"),
            _history_table(state),
            _gallery_items(state),
        )


with gr.Blocks(fill_height=True, title="Handwriting Data Collector", theme=THEME) as demo:
    gr.Markdown(
        """
        ## Handwriting Data Collector
        1. Write the prompted sentence on paper.
        2. Snap a photo (upload / webcam / paste).
        3. Draw a tight box around the sentence and press **Save and next**.

        Photos stay on this device; a ZIP with crops & metadata is generated at the end.
        """
    )
    state = gr.State()

    with gr.Row(equal_height=True):
        with gr.Column(scale=1, min_width=280):
            instruction = gr.Markdown()
            progress = gr.Markdown(elem_classes="progress-chip")
            gr.Markdown(
                """
                **Tips for crisp captures**
                - Shoot in good lighting to avoid shadows.
                - Keep the page flat and the camera parallel.
                - Draw the box right up to the handwriting edges.
                """
            )
            status = gr.Markdown()
        with gr.Column(scale=2):
            annotator = image_annotator(
                None,
                label_list=["sentence"],
                use_default_label=True,
                single_box=True,
                disable_edit_boxes=True,
                sources=["upload", "webcam", "clipboard"],
                height=480,
            )

    with gr.Row():
        btn_next = gr.Button("Save and next", variant="primary")
        btn_restart = gr.Button("Restart")

    zip_file = gr.File(label="Download dataset (.zip)", visible=False)

    with gr.Accordion("Captured sentences", open=False):
        gallery = gr.Gallery(label="Sentence crops", show_label=False, columns=5, rows=1, height="auto")
        history = gr.Dataframe(
            headers=["#", "Sentence", "Captured (UTC)"],
            datatype=["number", "str", "str"],
            interactive=False,
            wrap=True,
            col_count=(3, "fixed"),
        )

    btn_restart.click(start, outputs=[state, annotator, instruction, progress, zip_file, status, history, gallery])
    btn_next.click(
        save_and_next,
        inputs=[annotator, state],
        outputs=[state, annotator, instruction, progress, zip_file, status, history, gallery],
    )
    demo.load(start, outputs=[state, annotator, instruction, progress, zip_file, status, history, gallery])

demo.launch()
