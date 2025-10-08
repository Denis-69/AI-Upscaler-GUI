import glob
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
import subprocess
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.request
import tempfile
import zipfile

# Pull third-party availability/helpers from prepare_environment to avoid
# duplicating installation logic and import attempts in multiple modules.
from prepare_environment import (
    prepare_env,
    find_python310,
    pip_install,
    Image,
    ImageTk,
    DND_FILES,
    TkinterDnD,
    cv2,
    venv_pip_install,
    ensure_torch_for_device,
    CONSTRAINTS_FILE,
)

# Choose a resampling constant that's compatible across Pillow versions
try:
    RESAMPLE_LANCZOS = Image.Resampling.LANCZOS  # Pillow >=9
except Exception:
    # Pillow older versions may expose LANCZOS or have BICUBIC under Image
    res_bicubic = getattr(Image, "BICUBIC", None)
    if res_bicubic is None:
        try:
            res_bicubic = Image.Resampling.BICUBIC
        except Exception:
            res_bicubic = 3  # fallback numeric constant for BICUBIC
    RESAMPLE_LANCZOS = getattr(Image, "LANCZOS", res_bicubic)


REAL_ESRGAN_REPO = "https://github.com/xinntao/Real-ESRGAN.git"
REAL_ESRGAN_DIR = os.path.abspath("Real-ESRGAN")
PYTHON_EXEC = None


class ESRGAN_GUI:
    def __init__(self, root):
        self.root = root
        # --- Initialize instance state used by the UI and logic ---
        # threading, paths and image holders
        self.input_path = None
        self.output_path = None
        self.tk_image = None
        self.tk_output = None

        # Variables for zoom/pan and drag
        self.zoom = {"input": 1.0, "output": 1.0}
        self.pan = {"input": [0, 0], "output": [0, 0]}
        self.drag_data = {"x": 0, "y": 0}
        self.active_img_type = "input"

        # Processing state
        self.processing_lock = threading.Lock()
        self.is_processing = False
        self.current_process = None

        # Tk variables used by widgets (strict: require tkinter available)
        self.device_mode = tk.StringVar(value="GPU")
        self.model_name = tk.StringVar(value="RealESRGAN_x4plus")
        self.outscale = tk.StringVar(value="4")
        self.tile = tk.StringVar(value="128")
        self.parallel_jobs = tk.IntVar(value=1)
        self.frame_downscale = tk.IntVar(value=1)
        self.face_enhance = tk.BooleanVar(value=False)
        self.fp32 = tk.BooleanVar(value=True)
        self.output_dir = tk.StringVar()
        # Build UI in a dedicated method
        self.setup_ui()
<<<<<<< HEAD

        # --- Start background environment preparation ---
        # prepare_environment.prepare_env will call the provided setter with the
        # path to venv python when ready. This prevents venv helper methods from
        # raising "PYTHON_EXEC is not set" later.
        def _set_python_exec(python_path: str):
            try:
                global PYTHON_EXEC
                PYTHON_EXEC = python_path
                # reflect readiness in the UI if possible
                try:
                    self.status.config(text=f"Python venv ready: {os.path.basename(python_path)}", fg="green")
                    self.root.update_idletasks()
                except Exception:
                    pass
            except Exception:
                pass

        # Start prepare_env in background (non-blocking)
        # If a venv already exists from a previous run, set PYTHON_EXEC immediately
        try:
            venv_python = os.path.abspath(os.path.join(REAL_ESRGAN_DIR, "venv", "Scripts" if os.name == "nt" else "bin", "python"))
            if os.path.exists(venv_python):
                PYTHON_EXEC = venv_python
                try:
                    self.status.config(text=f"Using existing venv python: {os.path.basename(PYTHON_EXEC)}", fg="green")
                    self.root.update_idletasks()
                except Exception:
                    pass
        except Exception:
            pass

        try:
            threading.Thread(target=prepare_env, args=(self, REAL_ESRGAN_DIR, CONSTRAINTS_FILE, _set_python_exec), daemon=True).start()
        except Exception:
            # If background start fails, leave it for manual call
            pass
=======
        self.auto_set_tile()  # <--- добавьте этот вызов
        # self.prepare_env()
        threading.Thread(target=self.prepare_env, daemon=True).start()
>>>>>>> 497bb3a (Обновлён)

    def setup_ui(self):
        # Main window title and size
        try:
            self.root.title("AI Upscaler (Real-ESRGAN)")
            # Reduce window height by 60 pixels (was 950x860)
            self.root.geometry("950x780")
        except Exception:
            pass

        self.label = tk.Label(self.root, text="Выберите фото/видео для улучшения или перетащите его на область ниже", font=("Arial", 12))
        self.label.pack(pady=5)

        canvas_frame = tk.Frame(self.root)
        canvas_frame.pack()

        self.input_canvas = tk.Canvas(canvas_frame, width=450, height=400, bg="white")
        self.input_canvas.grid(row=0, column=0)
        self.output_canvas = tk.Canvas(canvas_frame, width=450, height=400, bg="white")
        self.output_canvas.grid(row=0, column=1)

        # Try to wire drag'n'drop if available. Use attribute checks so static
        # type checkers (and linters) won't complain about Canvas missing
        # these methods (they are provided by tkinterdnd2 at runtime).
        dtr = getattr(self.input_canvas, "drop_target_register", None)
        dndb = getattr(self.input_canvas, "dnd_bind", None)
        if callable(dtr) and callable(dndb):
            # type: ignore[attr-defined]
            dtr(DND_FILES)  # type: ignore[attr-defined]
            # type: ignore[attr-defined]
            dndb('<<Drop>>', self.on_drop)

        # Bind mouse and drag handlers
        self.input_canvas.bind("<MouseWheel>", lambda e: self.on_mousewheel(e, "input"))
        self.input_canvas.bind("<ButtonPress-1>", lambda e: self.on_drag_start(e, "input"))
        self.input_canvas.bind("<B1-Motion>", lambda e: self.on_drag_move(e, "input"))

        self.output_canvas.bind("<MouseWheel>", lambda e: self.on_mousewheel(e, "output"))
        self.output_canvas.bind("<ButtonPress-1>", lambda e: self.on_drag_start(e, "output"))
        self.output_canvas.bind("<B1-Motion>", lambda e: self.on_drag_move(e, "output"))

        control_frame = tk.Frame(self.root)
        control_frame.pack(pady=10)

        self.select_button = tk.Button(control_frame, text="Выбрать медиафайл", command=self.load_media)
        self.select_button.grid(row=0, column=0, padx=10)

        self.device_label = tk.Label(control_frame, text="Устройство:")
        self.device_label.grid(row=0, column=1)

        self.device_combo = ttk.Combobox(control_frame, textvariable=self.device_mode, values=["GPU", "CPU"], state="readonly", width=4)
        self.device_combo.grid(row=0, column=2, padx=5)

        self.process_button = tk.Button(control_frame, text="Обработать фото", command=self.start_upscale_thread, state=tk.DISABLED)
        self.process_button.grid(row=0, column=3, padx=10)

        self.cancel_flag = threading.Event()
        self.cancel_button = tk.Button(control_frame, text="Отмена", command=self.cancel_processing, state=tk.DISABLED)
        self.cancel_button.grid(row=0, column=4, padx=10)

        param_frame = tk.LabelFrame(self.root, text="Параметры", padx=10, pady=10)
        param_frame.pack(pady=5)

        tk.Label(param_frame, text="Модель:").grid(row=0, column=0, sticky="e")
        ttk.Combobox(param_frame, textvariable=self.model_name, values=[
            "RealESRGAN_x4plus", "RealESRNet_x4plus", "RealESRGAN_x4plus_anime_6B",
            "RealESRGAN_x2plus", "realesr-animevideov3", "realesr-general-x4v3"
        ], state="readonly", width=30).grid(row=0, column=1)

        tk.Label(param_frame, text="Увеличение (разы):").grid(row=1, column=0, sticky="e")
        ttk.Combobox(param_frame, textvariable=self.outscale, values=["1", "2", "3", "4", "6", "8"], width=10).grid(row=1, column=1, sticky="w")

        tk.Label(param_frame, text="Сжатие кадров:").grid(row=1, column=2, sticky="e", padx=(10, 0))
        self.frame_downscale_entry = tk.Entry(param_frame, textvariable=self.frame_downscale, width=4, justify="right")
        self.frame_downscale_entry.grid(row=1, column=3, sticky="w")

        tk.Label(param_frame, text="Размер плитки:").grid(row=2, column=0, sticky="e")
        ttk.Combobox(param_frame, textvariable=self.tile, values=["0", "32", "64", "92", "128", "192", "256", "512"], width=10).grid(row=2, column=1, sticky="w", padx=(0, 10))

        tk.Label(param_frame, text="Потоков:").grid(row=2, column=2, sticky="e", padx=(10, 0))
        self.parallel_jobs_entry = tk.Entry(param_frame, textvariable=self.parallel_jobs, width=4, justify="right")
        self.parallel_jobs_entry.grid(row=2, column=3, sticky="w")

        tk.Checkbutton(param_frame, text="Улучшение лиц", variable=self.face_enhance).grid(row=3, column=0, columnspan=2, sticky="w")
        tk.Checkbutton(param_frame, text="FP32 режим", variable=self.fp32).grid(row=4, column=0, sticky="w")

        tk.Label(param_frame, text="Папка для сохранения:").grid(row=5, column=0, sticky="e")
        tk.Entry(param_frame, textvariable=self.output_dir, width=30).grid(row=5, column=1, sticky="w")
        tk.Button(param_frame, text="Выбрать...", command=self.choose_output_dir).grid(row=5, column=2, padx=5)

        self.clear_temp = tk.BooleanVar(value=True)
        self.clear_temp_check = tk.Checkbutton(param_frame, text="Очистить TEMP после видео", variable=self.clear_temp, state=tk.NORMAL)
        self.clear_temp_check.grid(row=6, column=0, columnspan=2, sticky="w")

        self.status = tk.Label(self.root, text="", fg="blue")
        self.status.pack(pady=5)

        self.progress = ttk.Progressbar(self.root, orient="horizontal", length=400, mode="determinate")
        self.progress.pack(pady=5)

    def venv_pip_install(self, args, check=True, cwd=None):
        # Delegate to module-level venv_pip_install which accepts python_exec
        # If the background prepare_env hasn't finished yet, wait a short
        # while for PYTHON_EXEC to become available (best-effort). If it
        # still isn't available after the timeout, fall back to the current
        # interpreter and warn the user via the status label.
        wait_secs = 30.0
        poll = 0.25
        waited = 0.0
        while PYTHON_EXEC is None and waited < wait_secs:
            time.sleep(poll)
            waited += poll

            # Require PYTHON_EXEC to be set (strict). This mirrors the original
            # behaviour: the venv must be prepared before attempting constrained
            # pip installs into it.
            if PYTHON_EXEC is None:
                raise RuntimeError("PYTHON_EXEC is not set")
            return venv_pip_install(PYTHON_EXEC, args, check=check, cwd=cwd, constraints_file=CONSTRAINTS_FILE)


    def choose_output_dir(self):
        directory = filedialog.askdirectory()
        if directory:
            self.output_dir.set(directory)

    def detect_cuda_version(self):
        try:
            output = subprocess.check_output(["nvidia-smi"], encoding="utf-8")
            for line in output.splitlines():
                if "CUDA Version" in line:
                    return line.split("CUDA Version:")[-1].strip().split()[0]
        except Exception:
            return None

    def find_python310(self):
        # Delegate to the refactored function in prepare_environment.py
        return find_python310(self)

    def prepare_env(self):
<<<<<<< HEAD
        # Backwards-compatible wrapper that invokes the refactored function
        # synchronously. Ensure the module-level PYTHON_EXEC is set by the
        # setter passed to prepare_env.
        def _setter(python_path: str):
            try:
                globals()["PYTHON_EXEC"] = python_path
            except Exception:
                pass

        prepare_env(self, REAL_ESRGAN_DIR, CONSTRAINTS_FILE, _setter)
=======
        self.status.config(text="Поиск Python 3.10...", fg="blue")
        self.root.update_idletasks()
        python_cmd = self.find_python310()

        if not os.path.exists(REAL_ESRGAN_DIR):
            self.status.config(text="Клонирование Real-ESRGAN...", fg="blue")
            self.root.update_idletasks()
            subprocess.run(["git", "clone", REAL_ESRGAN_REPO, "Real-ESRGAN"], check=True)

        venv_dir = os.path.join(REAL_ESRGAN_DIR, "venv")
        if not os.path.exists(venv_dir):
            self.status.config(text="Создание виртуального окружения...", fg="blue")
            self.root.update_idletasks()
            subprocess.run([python_cmd, "-m", "venv", "venv"], cwd=REAL_ESRGAN_DIR, check=True)

        global PYTHON_EXEC
        PYTHON_EXEC = os.path.abspath(os.path.join(
            REAL_ESRGAN_DIR, "venv", "Scripts" if os.name == "nt" else "bin", "python"
        ))

        self.status.config(text="Обновление pip...", fg="blue")
        self.root.update_idletasks()
        subprocess.run([PYTHON_EXEC, "-m", "pip", "install", "--upgrade", "pip"], check=True)

        self.status.config(text="Установка numpy...", fg="blue")
        self.root.update_idletasks()
        subprocess.run([PYTHON_EXEC, "-m", "pip", "install", "numpy<2"], check=True)

        self.status.config(text="Установка realesrgan...", fg="blue")
        self.root.update_idletasks()
        subprocess.run([PYTHON_EXEC, "-m", "pip", "install", "realesrgan"], cwd=REAL_ESRGAN_DIR, check=True)

        # Исправление: создать realesrgan/version.py если его нет
        version_py = os.path.join(REAL_ESRGAN_DIR, "realesrgan", "version.py")
        if not os.path.exists(version_py):
            with open(version_py, "w", encoding="utf-8") as f:
                f.write("__version__ = '0.0.1'\n")
        # --- Конец блока ---

        cuda_version = self.detect_cuda_version()
        if cuda_version:
            self.status.config(text=f"CUDA {cuda_version} обнаружена, установка Torch...")
            cuda_str = cuda_version.replace(".", "")
            if cuda_str.startswith("12"):
                torch_cmd = ["pip", "install", "torch==2.1.0", "torchvision==0.16.0", "--index-url", "https://download.pytorch.org/whl/cu121"]
            elif cuda_str.startswith("11"):
                torch_cmd = ["pip", "install", "torch==2.0.1", "torchvision==0.15.2", "--index-url", "https://download.pytorch.org/whl/cu117"]
            else:
                torch_cmd = ["pip", "install", "torch==2.0.1+cpu", "torchvision==0.15.2+cpu", "--index-url", "https://download.pytorch.org/whl/cpu"]
        else:
            self.status.config(text="CUDA не обнаружена. Установка CPU версии Torch")
            torch_cmd = ["pip", "install", "torch==2.0.1+cpu", "torchvision==0.15.2+cpu", "--index-url", "https://download.pytorch.org/whl/cpu"]

        subprocess.run([PYTHON_EXEC, "-m"] + torch_cmd, check=True)

        result = subprocess.run(
            [PYTHON_EXEC, "-c", "import torch; print(torch.cuda.is_available())"],
            capture_output=True, text=True
        )
        self.gpu_available = "True" in result.stdout

        if not self.gpu_available:
            self.device_mode.set("CPU")
            self.device_combo.config(state="disabled")
            self.status.config(text="GPU не обнаружен. Работа в режиме CPU.")
        else:
            self.status.config(text="Готово. Выберите изображение.")
>>>>>>> 497bb3a (Обновлён)

    def on_canvas_click(self, event):
        # Определяем, по какому изображению кликнули (левое или правое)
        if event.x < 450:
            self.active_img_type = "input"
        else:
            self.active_img_type = "output"

    def on_drag_start(self, event, img_type):
        self.drag_data["x"] = event.x
        self.drag_data["y"] = event.y

    def on_drag_move(self, event, img_type):
        if self.zoom[img_type] <= 1.0:
            return  # Не двигаем при масштабе 1

        dx = event.x - self.drag_data["x"]
        dy = event.y - self.drag_data["y"]
        self.drag_data["x"] = event.x
        self.drag_data["y"] = event.y

        # Получаем размеры текущего изображения
        if img_type == "input" and self.input_path:
            img = Image.open(self.input_path)
        elif img_type == "output" and self.output_path:
            img = Image.open(self.output_path)
        else:
            return

        w, h = img.size
        scale = min(400 / w, 400 / h) * self.zoom[img_type]
        new_w, new_h = int(w * scale), int(h * scale)

        area_w = 450
        area_h = 400

        max_pan_x = max(0, (new_w - area_w) // 2)
        max_pan_y = max(0, (new_h - area_h) // 2)

        self.pan[img_type][0] += dx
        self.pan[img_type][1] += dy

        self.pan[img_type][0] = max(-max_pan_x, min(self.pan[img_type][0], max_pan_x))
        self.pan[img_type][1] = max(-max_pan_y, min(self.pan[img_type][1], max_pan_y))

        self.show_images()

    def on_mousewheel(self, event, img_type):
        if img_type == "input" and self.tk_image:
            self.zoom_image(event.delta, img_type)
        elif img_type == "output" and self.tk_output:
            self.zoom_image(event.delta, img_type)

    def zoom_image(self, delta, img_type):
        factor = 1.1 if delta > 0 else 0.9
        old_zoom = self.zoom[img_type]
        new_zoom = max(1.0, min(old_zoom * factor, 10))
        if abs(new_zoom - old_zoom) < 1e-3:
            return
        self.zoom[img_type] = new_zoom

        # Сбросить pan если zoom == 1
        if new_zoom == 1.0:
            self.pan[img_type] = [0, 0]
        self.show_images()


    def show_images(self):
        # Input (до)
        self.input_canvas.delete("all")
        if self.input_path:
            img = Image.open(self.input_path)
            w, h = img.size
            scale = min(400 / w, 400 / h) * self.zoom["input"]
            new_w, new_h = int(w * scale), int(h * scale)
            img = img.resize((new_w, new_h), RESAMPLE_LANCZOS)
            self.tk_image = ImageTk.PhotoImage(img)

            # Ограничения pan
            max_pan_x = max(0, (new_w - 450) // 2)
            max_pan_y = max(0, (new_h - 400) // 2)
            pan_x = max(-max_pan_x, min(self.pan["input"][0], max_pan_x))
            pan_y = max(-max_pan_y, min(self.pan["input"][1], max_pan_y))

            # Центрирование изображения
            x = 225 + pan_x
            y = 200 + pan_y

            self.input_canvas.create_image(x, y, image=self.tk_image)

        # Output (после)
        self.output_canvas.delete("all")
        if self.output_path:
            img = Image.open(self.output_path)
            w, h = img.size
            scale = min(400 / w, 400 / h) * self.zoom["output"]
            new_w, new_h = int(w * scale), int(h * scale)
            img = img.resize((new_w, new_h), RESAMPLE_LANCZOS)
            self.tk_output = ImageTk.PhotoImage(img)

            max_pan_x = max(0, (new_w - 450) // 2)
            max_pan_y = max(0, (new_h - 400) // 2)
            pan_x = max(-max_pan_x, min(self.pan["output"][0], max_pan_x))
            pan_y = max(-max_pan_y, min(self.pan["output"][1], max_pan_y))

            x = 225 + pan_x
            y = 200 + pan_y

            self.output_canvas.create_image(x, y, image=self.tk_output)

    def load_media(self):
        file_path = filedialog.askopenfilename(
            filetypes=[
                ("Медиафайлы", "*.jpg *.png *.jpeg *.bmp *.mp4 *.mov *.avi *.mkv"),
                ("Изображения", "*.jpg *.png *.jpeg *.bmp"),
                ("Видео", "*.mp4 *.mov *.avi *.mkv"),
            ]
        )
        if not file_path:
            return

        ext = os.path.splitext(file_path)[1].lower()
        image_exts = ('.jpg', '.jpeg', '.png', '.bmp')
        video_exts = ('.mp4', '.mov', '.avi', '.mkv')

        self.input_path = file_path
        self.zoom = {"input": 1.0, "output": 1.0}
        self.pan = {"input": [0, 0], "output": [0, 0]}
        self.output_path = None
        self.tk_output = None
        filename = os.path.basename(file_path)

        if ext in image_exts:
            self.show_images()
            self.status.config(text=f"Изображение выбрано: {filename}", fg="black")
            self.process_button.config(
                text="Обработать фото",
                command=self.start_upscale_thread,
                state=tk.NORMAL
            )
            self.clear_temp_check.config(state=tk.DISABLED)
            self.parallel_jobs_entry.config(state=tk.DISABLED)
        elif ext in video_exts:
            self.show_video_preview(file_path)
            self.output_canvas.delete("all")
            self.status.config(text=f"Видео выбрано: {filename}", fg="black")
            self.process_button.config(
                text="Обработать видео",
                command=self.start_video_processing,
                state=tk.NORMAL
            )
            self.clear_temp_check.config(state=tk.NORMAL)
            self.parallel_jobs_entry.config(state=tk.NORMAL)
        else:
            self.status.config(text="Файл не является поддерживаемым медиафайлом", fg="red")
            self.process_button.config(state=tk.DISABLED)
            self.clear_temp_check.config(state=tk.DISABLED)
            self.parallel_jobs_entry.config(state=tk.DISABLED)

    def on_drop(self, event):
        file_path = event.data.strip('{}')
        if not os.path.isfile(file_path):
            self.status.config(text="Файл не найден", fg="red")
            return

        ext = os.path.splitext(file_path)[1].lower()
        image_exts = ('.jpg', '.jpeg', '.png', '.bmp')
        video_exts = ('.mp4', '.mov', '.avi', '.mkv')
        filename = os.path.basename(file_path)

        self.input_path = file_path
        self.zoom = {"input": 1.0, "output": 1.0}
        self.pan = {"input": [0, 0], "output": [0, 0]}
        self.output_path = None
        self.tk_output = None

        if ext in image_exts:
            self.show_images()
            self.status.config(text=f"Изображение выбрано: {filename}", fg="black")
            self.process_button.config(
                text="Обработать фото",
                command=self.start_upscale_thread,
                state=tk.NORMAL
            )
            self.clear_temp_check.config(state=tk.DISABLED)
            self.parallel_jobs_entry.config(state=tk.DISABLED)
        elif ext in video_exts:
            self.show_video_preview(file_path)
            self.output_canvas.delete("all")
            self.status.config(text=f"Видео выбрано: {filename}", fg="black")
            self.process_button.config(
                text="Обработать видео",
                command=self.start_video_processing,
                state=tk.NORMAL
            )
            self.clear_temp_check.config(state=tk.NORMAL)
            self.parallel_jobs_entry.config(state=tk.NORMAL)
        else:
            self.status.config(text="Файл не является поддерживаемым медиафайлом", fg="red")
            self.process_button.config(state=tk.DISABLED)
            self.clear_temp_check.config(state=tk.DISABLED)
            self.parallel_jobs_entry.config(state=tk.DISABLED)

    def run_upscale(self):
        print(">>> run_upscale called")
        if not self.input_path:
            return
        # Попытка захватить lock — если уже обрабатывается, выходим
        if not self.processing_lock.acquire(blocking=False):
            self.status.config(text="Процесс уже запущен", fg="orange")
            return
        self.is_processing = True
        try:
            self.process_button.config(state=tk.DISABLED)
            self.cancel_flag.clear()
            self.cancel_button.config(state=tk.NORMAL)

            # Сбросить статус и цвет перед новым запуском
            self.status.config(text="")  # <--- очищаем статус
            self.root.update_idletasks()

            filename = os.path.basename(self.input_path)
            self.status.config(text=f"                                 Обработка файла: {filename}... Пожалуйста, подождите.                                 ", fg="black")
            self.root.update_idletasks()

            start_time = time.time()  # --- Счетчик времени ---

            try:
                input_name = os.path.splitext(os.path.basename(self.input_path))[0]

                # Determine output directory. If the user selected a folder,
                # use it. Otherwise, default to the source file's folder so
                # results appear next to the original file.
                if self.output_dir.get():
                    output_dir = self.output_dir.get()
                else:
                    output_dir = os.path.dirname(os.path.abspath(self.input_path))
                os.makedirs(output_dir, exist_ok=True)
                # --- End output dir selection ---

                # If the input path contains non-ASCII characters, many
                # downstream tools (OpenCV, ffmpeg, external scripts) may
                # fail to open it. Create an ASCII-only temp copy and pass
                # that to the inference script. We'll clean it up afterwards.
                def _make_ascii_temp(src_path, suffix=""):
                    # quick check for ASCII-only path
                    try:
                        src_path.encode('ascii')
                        return None  # no copy needed
                    except Exception:
                        pass
                    base = os.path.basename(src_path)
                    name, ext = os.path.splitext(base)
                    # create a safe ascii name using tempfile
                    fd, temp_path = tempfile.mkstemp(suffix=ext, prefix="aesr_", dir=tempfile.gettempdir())
                    os.close(fd)
                    try:
                        shutil.copy2(src_path, temp_path)
                    except Exception:
                        # fallback: try reading/writing bytes
                        with open(src_path, 'rb') as fsrc, open(temp_path, 'wb') as fdst:
                            fdst.write(fsrc.read())
                    return temp_path

                ascii_copy = _make_ascii_temp(self.input_path)

                input_for_cmd = ascii_copy or self.input_path

                # Use the basename of the file that was actually passed to
                # the inference script when searching for the output file.
                search_input_name = os.path.splitext(os.path.basename(input_for_cmd))[0]

                cmd = [
                    PYTHON_EXEC,
                    os.path.join(REAL_ESRGAN_DIR, "inference_realesrgan.py"),
                    "-n", self.model_name.get(),
                    "-i", input_for_cmd,
                    "--outscale", self.outscale.get(),
                    "--suffix", "_upscaled",
                    "--ext", "jpg",
                    "--tile", self.tile.get(),
                    "--output", output_dir,
                    "--alpha_upsampler", "bicubic",
                ]
                if self.fp32.get():
                    cmd.append("--fp32")
                if self.face_enhance.get():
                    cmd.append("--face_enhance")
                if self.device_mode.get() == "GPU":
                    self.ensure_torch_for_device("GPU")
                    cmd += ["-g", "0"]
                else:
                    self.ensure_torch_for_device("CPU")

                self.status.config(text=f"                                 Обработка файла: {filename}... Пожалуйста, подождите.                                 ", fg="black")
                self.current_process = subprocess.Popen(cmd, cwd=REAL_ESRGAN_DIR)
                self.current_process.wait()
                self.current_process = None

                # cleanup ascii temp copy if created
                try:
                    if ascii_copy and os.path.exists(ascii_copy):
                        os.remove(ascii_copy)
                except Exception:
                    pass

                # Look for files based on the input name used by the script
                pattern = os.path.join(output_dir, f"{search_input_name}_upscaled.*")
                matches = glob.glob(pattern)
                if not matches:
                    pattern = os.path.join(output_dir, f"{search_input_name}*_upscaled.*")
                    matches = glob.glob(pattern)

                elapsed = time.time() - start_time  # --- Счетчик времени ---
                elapsed_str = self.format_time(elapsed)
                if matches:
                    found = matches[0]
                    # If we used an ASCII copy, rename the produced file back
                    # to the original input base name so the GUI shows the
                    # expected filename to the user.
                    if ascii_copy and os.path.exists(found):
                        orig_name = f"{input_name}_upscaled{os.path.splitext(found)[1]}"
                        dest_path = os.path.join(output_dir, orig_name)
                        # avoid overwriting an existing file
                        if os.path.exists(dest_path):
                            base, ext = os.path.splitext(orig_name)
                            i = 1
                            while True:
                                candidate = os.path.join(output_dir, f"{base}_{i}{ext}")
                                if not os.path.exists(candidate):
                                    dest_path = candidate
                                    break
                                i += 1
                        try:
                            shutil.move(found, dest_path)
                            found = dest_path
                        except Exception:
                            # if rename fails, keep the original found path
                            pass

                    self.output_path = found
                    self.zoom["output"] = 1.0
                    self.pan["output"] = [0, 0]
                    self.show_images()
                    self.status.config(
                        text=f"Готово! ({elapsed_str}) Улучшенное изображение справа: {os.path.basename(self.output_path)}",
                        fg="green"
                    )
                else:
                    self.output_path = None
                    self.status.config(text="Ошибка: результат не найден", fg="red")

            except Exception as e:
                self.current_process = None
                self.status.config(text=f"Ошибка запуска: {e}", fg="red")
                self.cancel_button.config(state=tk.DISABLED)
        finally:
            # Всегда сбрасываем флаги и освобождаем lock
            self.cancel_button.config(state=tk.DISABLED)
            self.process_button.config(state=tk.NORMAL)
            self.is_processing = False
            try:
                self.processing_lock.release()
            except Exception:
                pass

    def get_vram_gb(self):
        try:
            output = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                encoding="utf-8"
            )
            vram_mb = int(output.strip().split('\n')[0])
            return vram_mb // 1024  # ГБ
        except Exception:
            return None

    def auto_set_tile(self):
        vram = self.get_vram_gb()
        if vram is None:
            self.tile.set("128")  # значение по умолчанию
            return
        if vram <= 2:
            self.tile.set("64")
        elif vram <= 4:
            self.tile.set("128")
        elif vram <= 6:
            self.tile.set("256")
        elif vram <= 8:
            self.tile.set("512")
        else:
            self.tile.set("1024")

    def process_video(self):
        video_path = self.input_path  # Используем уже выбранный путь
        if not video_path:
            return

        # Отдельный guard для видео — не запускать параллельно с другим процессом
        if not self.processing_lock.acquire(blocking=False):
            self.status.config(text="Процесс уже запущен", fg="orange")
            return
        self.is_processing = True
        try:
            self.process_button.config(state=tk.DISABLED)
            self.cancel_flag.clear()
            self.cancel_button.config(state=tk.NORMAL)

            start_time = time.time()  # <--- Добавлено для замера времени

            # If input video path contains non-ASCII characters, copy it to
            # an ASCII-only temp file so OpenCV/ffmpeg can open it reliably.
            def _make_ascii_temp(src_path, suffix=None):
                try:
                    src_path.encode('ascii')
                    return None
                except Exception:
                    pass
                if suffix is None:
                    _, ext = os.path.splitext(src_path)
                else:
                    ext = suffix
                fd, temp_path = tempfile.mkstemp(suffix=ext, prefix="aesr_vid_", dir=tempfile.gettempdir())
                os.close(fd)
                try:
                    shutil.copy2(src_path, temp_path)
                except Exception:
                    with open(src_path, 'rb') as fsrc, open(temp_path, 'wb') as fdst:
                        fdst.write(fsrc.read())
                return temp_path

            ascii_video_copy = _make_ascii_temp(video_path)
            video_for_processing = ascii_video_copy or video_path

            self.status.config(text="Извлечение кадров из видео...", fg="black")
            self.progress["value"] = 0
            self.root.update_idletasks()

            # --- Изменено: временные папки в ./TEMP ---
            base_temp = os.path.join(os.path.dirname(__file__), "TEMP")
            temp_dir = os.path.join(base_temp, "video_frames")
            upscaled_dir = os.path.join(base_temp, "video_upscaled")

            # --- Очистка TEMP перед началом обработки ---
            try:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                if os.path.exists(upscaled_dir):
                    shutil.rmtree(upscaled_dir)
            except Exception:
                pass

            os.makedirs(temp_dir, exist_ok=True)
            os.makedirs(upscaled_dir, exist_ok=True)

            cap = cv2.VideoCapture(video_for_processing)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            frame_paths = []
            index = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_downscale = max(1, self.frame_downscale.get())
                if frame_downscale > 1:
                    h, w = frame.shape[:2]
                    new_w = max(1, w // frame_downscale)
                    new_h = max(1, h // frame_downscale)
                    frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
                frame_path = os.path.join(temp_dir, f"frame_{index:05d}.jpg")
                cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                frame_paths.append(frame_path)
                index += 1
            cap.release()

            self.status.config(text=f"               Обработка {len(frame_paths)} кадров...               ", fg="black")
            self.progress["maximum"] = len(frame_paths)
            self.progress["value"] = 0
            self.root.update_idletasks()

            # --- Новые переменные для времени ---
            times = []
            processed = 0

            def upscale_single(path):
                if self.cancel_flag.is_set():
                    return None, 0
                t0 = time.time()
                cmd = [
                    PYTHON_EXEC,
                    os.path.join(REAL_ESRGAN_DIR, "inference_realesrgan.py"),
                    "-n", self.model_name.get(),
                    "-i", path,
                    "--outscale", self.outscale.get(),
                    "--suffix", "",
                    "--ext", "png",
                    "--tile", self.tile.get(),
                    "--output", upscaled_dir,
                    "--alpha_upsampler", "bicubic",
                ]
                if self.fp32.get():
                    cmd.append("--fp32")
                if self.face_enhance.get():
                    cmd.append("--face_enhance")
                if self.device_mode.get() == "GPU":
                    self.ensure_torch_for_device("GPU")
                    cmd += ["-g", "0"]
                else:
                    self.ensure_torch_for_device("CPU")

                try:
                    self.current_process = subprocess.Popen(cmd, cwd=REAL_ESRGAN_DIR, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    while self.current_process.poll() is None:
                        if self.cancel_flag.is_set():
                            self.current_process.terminate()
                            self.current_process = None
                            return None, 0
                        time.sleep(0.1)
                    self.current_process = None
                    t1 = time.time()
                    return path, t1 - t0
                except Exception as e:
                    self.current_process = None
                    return None, 0

            max_workers = self.parallel_jobs.get()
            self.progress["value"] = 0
            self.progress["maximum"] = len(frame_paths)
            self.root.update_idletasks()

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(upscale_single, path): path for path in frame_paths}
                for i, future in enumerate(as_completed(futures), start=1):
                    if self.cancel_flag.is_set():
                        self.status.config(text="Обработка отменена", fg="red")
                        self.cancel_button.config(state=tk.DISABLED)
                        self.process_button.config(state=tk.NORMAL)
                        return
                    result, elapsed = future.result()
                    if not result:
                        if self.cancel_flag.is_set():
                            self.status.config(text="Обработка отменена", fg="red")
                        else:
                            self.status.config(text="Ошибка при обработке одного из кадров", fg="red")
                        self.cancel_button.config(state=tk.DISABLED)
                        self.process_button.config(state=tk.NORMAL)
                        return
                    times.append(elapsed)
                    processed = i
                    avg_time = sum(times) / len(times)
                    remaining = len(frame_paths) - processed
                    eta = avg_time * remaining
                    percent = processed / len(frame_paths) * 100
                    self.progress["value"] = processed
                    self.status.config(
                        text=f"Кадр {processed}/{len(frame_paths)} ({percent:.1f}%) | Примерно осталось: {self.format_time(eta)}", fg="black"
                    )
                    self.root.update_idletasks()

            self.status.config(text="Сборка видео...", fg="black")
            self.root.update_idletasks()

            # Save video beside original if no output folder chosen
            if self.output_dir.get():
                output_dir = self.output_dir.get()
            else:
                output_dir = os.path.dirname(os.path.abspath(video_path))
            os.makedirs(output_dir, exist_ok=True)
            input_name = os.path.splitext(os.path.basename(video_path))[0]
            output_video = os.path.join(output_dir, f"{input_name}_upscaled.mp4")

            # Собрать видео из кадров
            try:
                # For ffmpeg, use the ASCII copy as the audio source if created
                audio_source = video_for_processing
                self.current_process = subprocess.Popen([
                    "ffmpeg", "-y",
                    "-framerate", str(fps),
                    "-i", os.path.join(upscaled_dir, "frame_%05d.png"),
                    "-i", audio_source,
                    "-map", "0:v:0",
                    "-map", "1:a:0",
                    "-c:v", "libx264",
                    "-c:a", "aac",
                    "-pix_fmt", "yuv420p",
                    "-shortest",
                    "-loglevel", "error",
                    output_video
                ])
                self.current_process.wait()
                self.current_process = None
            except Exception as e:
                self.current_process = None
                self.status.config(text="Ошибка при сборке видео", fg="red")
                self.cancel_button.config(state=tk.DISABLED)
                self.process_button.config(state=tk.NORMAL)
                return

            # После сборки видео
            elapsed = time.time() - start_time  # <--- Добавлено
            elapsed_str = self.format_time(elapsed)
            self.status.config(
                text=f"Видео обработано за {elapsed_str}: {output_video}", fg="green"
            )
            messagebox.showinfo("Готово", f"Файл сохранён:\n{output_video}\nВремя обработки: {elapsed_str}")
            self.cancel_button.config(state=tk.DISABLED)
            self.process_button.config(state=tk.NORMAL)

            # --- Очистка TEMP после обработки, если стоит галочка ---
            if self.clear_temp.get():
                try:
                    if os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir)
                    if os.path.exists(upscaled_dir):
                        shutil.rmtree(upscaled_dir)
                except Exception:
                    pass
            # Remove ASCII copy of original video if we created one
            try:
                if ascii_video_copy and os.path.exists(ascii_video_copy):
                    os.remove(ascii_video_copy)
            except Exception:
                pass
        finally:
            self.is_processing = False
            try:
                self.processing_lock.release()
            except Exception:
                pass

    def format_time(self, seconds):
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h:02}:{m:02}:{s:02}"

    # Добавьте метод отмены:
    def cancel_processing(self):
        self.cancel_flag.set()
        if self.current_process is not None:
            try:
                self.current_process.terminate()
            except Exception:
                pass
            self.current_process = None
        self.status.config(text="Отмена запрошена...", fg="red")
        self.cancel_button.config(state=tk.DISABLED)
        self.process_button.config(state=tk.NORMAL)

    def show_video_preview(self, video_path):
        try:
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            cap.release()
            if ret:
                # Преобразуем BGR (OpenCV) -> RGB (PIL)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                w, h = img.size
                scale = min(400 / w, 400 / h)
                new_w, new_h = int(w * scale), int(h * scale)
                img = img.resize((new_w, new_h), RESAMPLE_LANCZOS)
                self.tk_image = ImageTk.PhotoImage(img)
                self.input_canvas.delete("all")
                x = 225
                y = 200
                self.input_canvas.create_image(x, y, image=self.tk_image)
            else:
                self.input_canvas.delete("all")
        except Exception:
            self.input_canvas.delete("all")
        self.output_canvas.delete("all")

    def ensure_torch_for_device(self, device: str):
        # Delegate to module-level implementation. Wait a short while for
        # PYTHON_EXEC to become available; if it doesn't, fall back to the
        # host interpreter (best-effort).
        wait_secs = 30.0
        poll = 0.25
        waited = 0.0
        while PYTHON_EXEC is None and waited < wait_secs:
            time.sleep(poll)
            waited += poll

            # Require PYTHON_EXEC to be set (strict).
            if PYTHON_EXEC is None:
                raise RuntimeError("PYTHON_EXEC is not set")
            return ensure_torch_for_device(PYTHON_EXEC, device, self)

    def start_upscale_thread(self):
        # Быстрая защита от повторного запуска
        if self.is_processing:
            self.status.config(text="Процесс уже запущен", fg="orange")
            return
        self.process_button.config(state=tk.DISABLED)
        threading.Thread(target=self.run_upscale, daemon=True).start()

    def start_video_processing(self):
        if self.is_processing:
            self.status.config(text="Процесс уже запущен", fg="orange")
            return
        self.process_button.config(state=tk.DISABLED)
        threading.Thread(target=self.process_video, daemon=True).start()

if __name__ == "__main__":
    root = TkinterDnD.Tk()  # Используем TkinterDnD
    app = ESRGAN_GUI(root)
    root.mainloop()
