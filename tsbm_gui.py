"""Shared Tkinter interface for the TSBM analysis scripts.

The GUI runs calculations in a background thread, captures console output,
shows generated images, and provides a small non-destructive image editor.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable
import contextlib
import io
import os
import shutil
import subprocess
import sys
import threading
import traceback

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

try:
    from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageTk, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:  # pragma: no cover - handled in the GUI
    PIL_AVAILABLE = False


Runner = Callable[[Path, dict[str, Any]], Any]


@dataclass(frozen=True)
class OptionSpec:
    key: str
    label: str
    kind: str = "str"  # str, int, float, bool, choice
    default: Any = ""
    choices: tuple[str, ...] = ()
    help_text: str = ""


@dataclass(frozen=True)
class TaskSpec:
    title: str
    description: str
    runner: Runner
    default_output: str = "outputs"
    options: tuple[OptionSpec, ...] = field(default_factory=tuple)
    run_button_text: str = "Run analysis"
    warning: str = ""


class _GuiWriter(io.TextIOBase):
    """Thread-safe stdout/stderr redirector for a Tk text widget."""

    def __init__(self, append_callback: Callable[[str], None]) -> None:
        super().__init__()
        self._append_callback = append_callback

    def write(self, text: str) -> int:
        if text:
            self._append_callback(text)
        return len(text)

    def flush(self) -> None:
        return None


class ImageEditor(tk.Toplevel):
    """Small, non-destructive editor for generated result images."""

    def __init__(self, master: tk.Misc, image_path: Path) -> None:
        super().__init__(master)
        self.title(f"Edit image - {image_path.name}")
        self.geometry("1040x760")
        self.minsize(860, 620)
        self.image_path = image_path
        self.original = None
        self.base_image = None
        self.working_image = None
        self._preview_photo = None

        if not PIL_AVAILABLE:
            messagebox.showerror(
                "Pillow is required",
                "Image editing requires Pillow. Install it with: pip install pillow",
                parent=self,
            )
            self.destroy()
            return

        try:
            self.original = Image.open(image_path).convert("RGBA")
        except Exception as exc:
            messagebox.showerror("Cannot open image", str(exc), parent=self)
            self.destroy()
            return

        self.base_image = self.original.copy()
        self.working_image = self.original.copy()
        self._build()
        self._refresh_preview()

    def _build(self) -> None:
        outer = ttk.Frame(self, padding=10)
        outer.pack(fill="both", expand=True)

        ttk.Label(
            outer,
            text=(
                "Basic non-destructive editing. Save the edited result as a new file. "
                "Do not use visual edits to change or conceal scientific data."
            ),
            wraplength=980,
        ).pack(fill="x", pady=(0, 8))

        body = ttk.Frame(outer)
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=1)
        body.rowconfigure(0, weight=1)

        preview_frame = ttk.LabelFrame(body, text="Preview", padding=8)
        preview_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(0, weight=1)
        self.preview_label = ttk.Label(preview_frame, anchor="center")
        self.preview_label.grid(row=0, column=0, sticky="nsew")

        controls = ttk.LabelFrame(body, text="Editing controls", padding=10)
        controls.grid(row=0, column=1, sticky="ns")

        transform = ttk.LabelFrame(controls, text="Transform", padding=6)
        transform.pack(fill="x", pady=(0, 8))
        ttk.Button(transform, text="Rotate left", command=lambda: self._transform("left")).grid(row=0, column=0, padx=2, pady=2)
        ttk.Button(transform, text="Rotate right", command=lambda: self._transform("right")).grid(row=0, column=1, padx=2, pady=2)
        ttk.Button(transform, text="Flip horizontal", command=lambda: self._transform("flip_h")).grid(row=1, column=0, padx=2, pady=2)
        ttk.Button(transform, text="Flip vertical", command=lambda: self._transform("flip_v")).grid(row=1, column=1, padx=2, pady=2)
        ttk.Button(transform, text="Grayscale", command=lambda: self._transform("gray")).grid(row=2, column=0, padx=2, pady=2)
        ttk.Button(transform, text="Reset", command=self._reset).grid(row=2, column=1, padx=2, pady=2)

        adjust = ttk.LabelFrame(controls, text="Brightness / contrast / sharpness", padding=6)
        adjust.pack(fill="x", pady=(0, 8))
        self.brightness_var = tk.DoubleVar(value=1.0)
        self.contrast_var = tk.DoubleVar(value=1.0)
        self.sharpness_var = tk.DoubleVar(value=1.0)
        for row, (label, var, low, high) in enumerate(
            (
                ("Brightness", self.brightness_var, 0.4, 1.8),
                ("Contrast", self.contrast_var, 0.4, 1.8),
                ("Sharpness", self.sharpness_var, 0.0, 2.5),
            )
        ):
            ttk.Label(adjust, text=label).grid(row=row * 2, column=0, sticky="w")
            scale = ttk.Scale(adjust, from_=low, to=high, variable=var, orient="horizontal", length=230, command=lambda _v: self._apply_adjustments())
            scale.grid(row=row * 2 + 1, column=0, sticky="ew", pady=(0, 5))

        crop = ttk.LabelFrame(controls, text="Crop margins (%)", padding=6)
        crop.pack(fill="x", pady=(0, 8))
        self.crop_vars: dict[str, tk.StringVar] = {}
        for i, key in enumerate(("Left", "Top", "Right", "Bottom")):
            ttk.Label(crop, text=key).grid(row=i // 2 * 2, column=i % 2, sticky="w", padx=3)
            var = tk.StringVar(value="0")
            self.crop_vars[key] = var
            ttk.Entry(crop, textvariable=var, width=10).grid(row=i // 2 * 2 + 1, column=i % 2, padx=3, pady=(0, 4))
        ttk.Button(crop, text="Apply crop", command=self._apply_crop).grid(row=4, column=0, columnspan=2, pady=4)

        annotate = ttk.LabelFrame(controls, text="Optional note", padding=6)
        annotate.pack(fill="x", pady=(0, 8))
        self.note_var = tk.StringVar()
        ttk.Entry(annotate, textvariable=self.note_var, width=34).pack(fill="x", pady=(0, 4))
        ttk.Button(annotate, text="Add note at bottom", command=self._add_note).pack(fill="x")

        buttons = ttk.Frame(controls)
        buttons.pack(fill="x", pady=(8, 0))
        ttk.Button(buttons, text="Save as...", command=self._save_as).pack(side="left", padx=(0, 4))
        ttk.Button(buttons, text="Close", command=self.destroy).pack(side="right")

    def _transform(self, action: str) -> None:
        assert self.base_image is not None
        if action == "left":
            self.base_image = self.base_image.rotate(90, expand=True)
        elif action == "right":
            self.base_image = self.base_image.rotate(-90, expand=True)
        elif action == "flip_h":
            self.base_image = ImageOps.mirror(self.base_image)
        elif action == "flip_v":
            self.base_image = ImageOps.flip(self.base_image)
        elif action == "gray":
            self.base_image = ImageOps.grayscale(self.base_image).convert("RGBA")
        self._apply_adjustments()

    def _apply_adjustments(self) -> None:
        if self.base_image is None:
            return
        image = self.base_image.copy().convert("RGBA")
        image = ImageEnhance.Brightness(image).enhance(float(self.brightness_var.get()))
        image = ImageEnhance.Contrast(image).enhance(float(self.contrast_var.get()))
        image = ImageEnhance.Sharpness(image).enhance(float(self.sharpness_var.get()))
        self.working_image = image
        self._refresh_preview()

    def _apply_crop(self) -> None:
        if self.base_image is None:
            return
        try:
            values = {key: float(var.get()) for key, var in self.crop_vars.items()}
        except ValueError:
            messagebox.showerror("Invalid crop", "Crop values must be numbers.", parent=self)
            return
        if any(v < 0 or v >= 50 for v in values.values()):
            messagebox.showerror("Invalid crop", "Each crop margin must be between 0 and 49 percent.", parent=self)
            return
        width, height = self.base_image.size
        left = round(width * values["Left"] / 100)
        top = round(height * values["Top"] / 100)
        right = width - round(width * values["Right"] / 100)
        bottom = height - round(height * values["Bottom"] / 100)
        if right <= left or bottom <= top:
            messagebox.showerror("Invalid crop", "The crop margins remove the entire image.", parent=self)
            return
        self.base_image = self.base_image.crop((left, top, right, bottom))
        for var in self.crop_vars.values():
            var.set("0")
        self._apply_adjustments()

    def _add_note(self) -> None:
        if self.working_image is None:
            return
        note = self.note_var.get().strip()
        if not note:
            messagebox.showinfo("No note", "Enter note text first.", parent=self)
            return
        image = self.working_image.convert("RGBA")
        font = ImageFont.load_default()
        draw = ImageDraw.Draw(image)
        bbox = draw.textbbox((0, 0), note, font=font)
        text_h = max(20, bbox[3] - bbox[1] + 12)
        canvas = Image.new("RGBA", (image.width, image.height + text_h), "white")
        canvas.paste(image, (0, 0), image)
        draw = ImageDraw.Draw(canvas)
        draw.text((8, image.height + 5), note, fill="black", font=font)
        self.base_image = canvas
        self.note_var.set("")
        self._apply_adjustments()

    def _reset(self) -> None:
        assert self.original is not None
        self.base_image = self.original.copy()
        self.brightness_var.set(1.0)
        self.contrast_var.set(1.0)
        self.sharpness_var.set(1.0)
        self._apply_adjustments()

    def _refresh_preview(self) -> None:
        if self.working_image is None:
            return
        preview = self.working_image.copy()
        preview.thumbnail((700, 620), Image.Resampling.LANCZOS)
        self._preview_photo = ImageTk.PhotoImage(preview)
        self.preview_label.configure(image=self._preview_photo)

    def _save_as(self) -> None:
        if self.working_image is None:
            return
        initial = self.image_path.with_name(f"{self.image_path.stem}_edited{self.image_path.suffix}")
        target = filedialog.asksaveasfilename(
            parent=self,
            title="Save edited image",
            initialdir=str(initial.parent),
            initialfile=initial.name,
            defaultextension=self.image_path.suffix or ".png",
            filetypes=[
                ("PNG image", "*.png"),
                ("TIFF image", "*.tif *.tiff"),
                ("JPEG image", "*.jpg *.jpeg"),
                ("PDF image", "*.pdf"),
                ("All files", "*.*"),
            ],
        )
        if not target:
            return
        try:
            image = self.working_image
            suffix = Path(target).suffix.lower()
            if suffix in {".jpg", ".jpeg", ".pdf"}:
                image = image.convert("RGB")
            image.save(target, dpi=(300, 300))
            messagebox.showinfo("Saved", f"Edited image saved to:\n{target}", parent=self)
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc), parent=self)


class TaskWindow:
    def __init__(self, master: tk.Misc | None, spec: TaskSpec, standalone: bool = False) -> None:
        self.spec = spec
        self.standalone = standalone
        self.window: tk.Tk | tk.Toplevel
        if standalone:
            self.window = tk.Tk()
        else:
            assert master is not None
            self.window = tk.Toplevel(master)
        self.window.title(spec.title)
        self.window.geometry("1160x820")
        self.window.minsize(950, 680)
        self.option_vars: dict[str, tk.Variable] = {}
        self.image_paths: list[Path] = []
        self._preview_photo = None
        self._running = False
        self._build()

    def _build(self) -> None:
        root = ttk.Frame(self.window, padding=12)
        root.pack(fill="both", expand=True)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(3, weight=1)
        root.rowconfigure(5, weight=2)

        ttk.Label(root, text=self.spec.title, font=("TkDefaultFont", 16, "bold")).grid(row=0, column=0, sticky="w")
        ttk.Label(root, text=self.spec.description, wraplength=1080, justify="left").grid(row=1, column=0, sticky="ew", pady=(5, 7))
        if self.spec.warning:
            ttk.Label(root, text=self.spec.warning, wraplength=1080, foreground="#8a3b12").grid(row=2, column=0, sticky="ew", pady=(0, 7))

        settings = ttk.LabelFrame(root, text="Run settings", padding=8)
        settings.grid(row=3, column=0, sticky="new", pady=(0, 8))
        settings.columnconfigure(1, weight=1)

        ttk.Label(settings, text="Output folder").grid(row=0, column=0, sticky="w", padx=(0, 6))
        default_path = Path.cwd() / self.spec.default_output
        self.output_var = tk.StringVar(value=str(default_path.resolve()))
        ttk.Entry(settings, textvariable=self.output_var).grid(row=0, column=1, sticky="ew")
        ttk.Button(settings, text="Browse...", command=self._browse_output).grid(row=0, column=2, padx=(6, 0))

        option_row = 1
        for option in self.spec.options:
            ttk.Label(settings, text=option.label).grid(row=option_row, column=0, sticky="w", pady=3, padx=(0, 6))
            if option.kind == "bool":
                var = tk.BooleanVar(value=bool(option.default))
                widget = ttk.Checkbutton(settings, variable=var)
            elif option.kind == "choice":
                var = tk.StringVar(value=str(option.default))
                widget = ttk.Combobox(settings, textvariable=var, values=option.choices, state="readonly")
            else:
                var = tk.StringVar(value=str(option.default))
                widget = ttk.Entry(settings, textvariable=var)
            widget.grid(row=option_row, column=1, sticky="ew", pady=3)
            self.option_vars[option.key] = var
            if option.help_text:
                ttk.Label(settings, text=option.help_text, wraplength=320, foreground="#555555").grid(row=option_row, column=2, sticky="w", padx=(8, 0))
            option_row += 1

        action = ttk.Frame(root)
        action.grid(row=4, column=0, sticky="ew", pady=(0, 8))
        self.run_button = ttk.Button(action, text=self.spec.run_button_text, command=self._start_run)
        self.run_button.pack(side="left")
        ttk.Button(action, text="Refresh images", command=self._refresh_images).pack(side="left", padx=5)
        ttk.Button(action, text="Open output folder", command=self._open_output_folder).pack(side="left")
        self.progress = ttk.Progressbar(action, mode="indeterminate", length=250)
        self.progress.pack(side="right", padx=(8, 0))
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(action, textvariable=self.status_var).pack(side="right")

        lower = ttk.Panedwindow(root, orient="vertical")
        lower.grid(row=5, column=0, sticky="nsew")

        log_frame = ttk.LabelFrame(lower, text="Run log", padding=5)
        self.log = ScrolledText(log_frame, height=12, wrap="word", state="disabled")
        self.log.pack(fill="both", expand=True)
        lower.add(log_frame, weight=1)

        gallery_frame = ttk.LabelFrame(lower, text="Generated images", padding=5)
        gallery_frame.columnconfigure(1, weight=1)
        gallery_frame.rowconfigure(0, weight=1)
        list_side = ttk.Frame(gallery_frame)
        list_side.grid(row=0, column=0, sticky="nsw", padx=(0, 8))
        self.image_list = tk.Listbox(list_side, width=42, height=16, exportselection=False)
        self.image_list.pack(fill="both", expand=True)
        self.image_list.bind("<<ListboxSelect>>", lambda _event: self._show_selected_image())
        self.image_list.bind("<Double-1>", lambda _event: self._edit_selected())
        image_buttons = ttk.Frame(list_side)
        image_buttons.pack(fill="x", pady=(5, 0))
        ttk.Button(image_buttons, text="Edit image", command=self._edit_selected).pack(side="left")
        ttk.Button(image_buttons, text="Save copy as...", command=self._save_copy).pack(side="left", padx=4)

        preview_box = ttk.Frame(gallery_frame)
        preview_box.grid(row=0, column=1, sticky="nsew")
        preview_box.columnconfigure(0, weight=1)
        preview_box.rowconfigure(0, weight=1)
        self.preview_label = ttk.Label(preview_box, text="Generated images will appear here.", anchor="center")
        self.preview_label.grid(row=0, column=0, sticky="nsew")
        lower.add(gallery_frame, weight=2)

        self._refresh_images()

    def _browse_output(self) -> None:
        selected = filedialog.askdirectory(parent=self.window, initialdir=self.output_var.get() or str(Path.cwd()))
        if selected:
            self.output_var.set(selected)
            self._refresh_images()

    def _parse_options(self) -> dict[str, Any]:
        parsed: dict[str, Any] = {}
        for option in self.spec.options:
            raw = self.option_vars[option.key].get()
            if option.kind == "int":
                parsed[option.key] = int(str(raw).strip())
            elif option.kind == "float":
                parsed[option.key] = float(str(raw).strip())
            elif option.kind == "bool":
                parsed[option.key] = bool(raw)
            else:
                parsed[option.key] = str(raw)
        return parsed

    def _start_run(self) -> None:
        if self._running:
            return
        try:
            output = Path(self.output_var.get()).expanduser().resolve()
            options = self._parse_options()
        except Exception as exc:
            messagebox.showerror("Invalid setting", str(exc), parent=self.window)
            return
        output.mkdir(parents=True, exist_ok=True)
        self._clear_log()
        self._running = True
        self.run_button.configure(state="disabled")
        self.progress.start(12)
        self.status_var.set("Running...")
        self._append_log(f"Starting: {self.spec.title}\nOutput folder: {output}\n\n")
        thread = threading.Thread(target=self._run_worker, args=(output, options), daemon=True)
        thread.start()

    def _run_worker(self, output: Path, options: dict[str, Any]) -> None:
        writer = _GuiWriter(self._append_log)
        success = True
        error_text = ""
        try:
            with contextlib.redirect_stdout(writer), contextlib.redirect_stderr(writer):
                self.spec.runner(output, options)
        except Exception:
            success = False
            error_text = traceback.format_exc()
            writer.write("\n" + error_text)
        self.window.after(0, lambda: self._finish_run(success, error_text))

    def _finish_run(self, success: bool, error_text: str) -> None:
        self._running = False
        self.run_button.configure(state="normal")
        self.progress.stop()
        self._refresh_images()
        if success:
            self.status_var.set("Complete")
            self._append_log("\nRun completed successfully.\n")
            messagebox.showinfo("Complete", "The analysis finished successfully.", parent=self.window)
        else:
            self.status_var.set("Failed")
            messagebox.showerror("Run failed", error_text[-2000:] or "Unknown error", parent=self.window)

    def _append_log(self, text: str) -> None:
        def append() -> None:
            try:
                self.log.configure(state="normal")
                self.log.insert("end", text)
                self.log.see("end")
                self.log.configure(state="disabled")
            except tk.TclError:
                pass
        try:
            self.window.after(0, append)
        except tk.TclError:
            pass

    def _clear_log(self) -> None:
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    def _refresh_images(self) -> None:
        output = Path(self.output_var.get()).expanduser()
        exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
        if output.exists():
            self.image_paths = sorted(
                (p for p in output.rglob("*") if p.is_file() and p.suffix.lower() in exts),
                key=lambda p: (str(p.parent), p.name.lower()),
            )
        else:
            self.image_paths = []
        self.image_list.delete(0, "end")
        for path in self.image_paths:
            try:
                label = str(path.relative_to(output))
            except ValueError:
                label = path.name
            self.image_list.insert("end", label)
        if self.image_paths:
            self.image_list.selection_set(0)
            self._show_selected_image()
        else:
            self.preview_label.configure(image="", text="No generated images found in the selected output folder.")
            self._preview_photo = None

    def _selected_path(self) -> Path | None:
        selection = self.image_list.curselection()
        if not selection:
            return None
        idx = int(selection[0])
        if idx >= len(self.image_paths):
            return None
        return self.image_paths[idx]

    def _show_selected_image(self) -> None:
        path = self._selected_path()
        if path is None:
            return
        if not PIL_AVAILABLE:
            self.preview_label.configure(text="Install Pillow to preview and edit images: pip install pillow")
            return
        try:
            image = Image.open(path)
            image.thumbnail((760, 420), Image.Resampling.LANCZOS)
            self._preview_photo = ImageTk.PhotoImage(image)
            self.preview_label.configure(image=self._preview_photo, text="")
        except Exception as exc:
            self.preview_label.configure(image="", text=f"Could not preview image:\n{exc}")

    def _edit_selected(self) -> None:
        path = self._selected_path()
        if path is None:
            messagebox.showinfo("Select an image", "Select an image first.", parent=self.window)
            return
        ImageEditor(self.window, path)

    def _save_copy(self) -> None:
        path = self._selected_path()
        if path is None:
            messagebox.showinfo("Select an image", "Select an image first.", parent=self.window)
            return
        target = filedialog.asksaveasfilename(
            parent=self.window,
            title="Save image copy",
            initialfile=path.name,
            defaultextension=path.suffix,
            filetypes=[("Image file", f"*{path.suffix}"), ("All files", "*.*")],
        )
        if not target:
            return
        try:
            shutil.copy2(path, target)
            messagebox.showinfo("Saved", f"Image copied to:\n{target}", parent=self.window)
        except Exception as exc:
            messagebox.showerror("Copy failed", str(exc), parent=self.window)

    def _open_output_folder(self) -> None:
        path = Path(self.output_var.get()).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        try:
            if sys.platform.startswith("win"):
                os.startfile(path)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path)])
        except Exception as exc:
            messagebox.showerror("Cannot open folder", str(exc), parent=self.window)

    def mainloop(self) -> None:
        if self.standalone:
            self.window.mainloop()


def launch_task_gui(spec: TaskSpec) -> None:
    """Launch a standalone GUI for one analysis task."""
    window = TaskWindow(None, spec, standalone=True)
    window.mainloop()


def open_task_window(master: tk.Misc, spec: TaskSpec) -> TaskWindow:
    """Open a task as a child window from the master launcher."""
    return TaskWindow(master, spec, standalone=False)
