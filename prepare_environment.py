import os
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
import shutil
import tkinter.messagebox as messagebox


def pip_install(package):
    """Install a package into the current interpreter if missing."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    except Exception:
        # best-effort: ignore failures here; caller will handle missing imports
        pass


# Ensure third-party runtime libraries are available and expose them as module
# level names so the GUI can import them from here. 
try:
    from PIL import Image, ImageTk
except Exception:
    pip_install("pillow")
    from PIL import Image, ImageTk

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
except Exception:
    pip_install("tkinterdnd2")
    from tkinterdnd2 import DND_FILES, TkinterDnD

try:
    import cv2
except Exception:
    pip_install("opencv-python")
    import cv2

try:
    import urllib
except Exception:
    import urllib


# Constraints file path exported for use by other modules
CONSTRAINTS_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "pip_constraints.txt"))




def prepare_env(gui, real_esrgan_dir: str, constraints_file: str, set_python_exec):
    """Prepare a minimal environment for Real-ESRGAN.

    This is a compact, robust implementation intended to be called from the GUI
    in a background thread. It will ensure a virtualenv exists at
    real_esrgan_dir/venv, perform best-effort pip installs (upgrade pip,
    install numpy<2 and realesrgan), write a `.env_ready` marker and call
    set_python_exec(python_path) so the GUI can use the venv's Python.

    The function is intentionally lenient about failures during pip installs
    (they are best-effort) but will show an error message and close the GUI on
    fatal problems (like failing to create a venv).
    """
    REAL_ESRGAN_DIR = real_esrgan_dir
    CONSTRAINTS_FILE = constraints_file

    def safe_update_status(text: str, fg: str = "blue"):
        try:
            gui.status.config(text=text, fg=fg)
            gui.root.update_idletasks()
        except Exception:
            pass

    try:
        safe_update_status("Подготовка окружения...", "blue")

        # Clone repo if missing
        if not os.path.exists(REAL_ESRGAN_DIR):
            safe_update_status("Клонирование Real-ESRGAN...", "blue")
            subprocess.run(["git", "clone", "https://github.com/xinntao/Real-ESRGAN.git", "Real-ESRGAN"], check=True)

        # Create venv if missing
        venv_dir = os.path.join(REAL_ESRGAN_DIR, "venv")
        if not os.path.exists(venv_dir):
            safe_update_status("Создание виртуального окружения...", "blue")
            python_cmd = sys.executable
            try:
                res = subprocess.run(["python3.10", "--version"], capture_output=True, text=True)
                if res.returncode == 0:
                    python_cmd = "python3.10"
            except Exception:
                pass
            subprocess.run([python_cmd, "-m", "venv", "venv"], cwd=REAL_ESRGAN_DIR, check=True)

        python_exec = os.path.abspath(os.path.join(REAL_ESRGAN_DIR, "venv", "Scripts" if os.name == "nt" else "bin", "python"))
        set_python_exec(python_exec)

        # Best-effort installs: upgrade pip, install numpy<2 and realesrgan
        safe_update_status("Обновление pip...", "blue")
        subprocess.run([python_exec, "-m", "pip", "install", "--upgrade", "pip"], check=False)

        safe_update_status("Проверка numpy в venv...", "blue")
        try:
            # Try to get numpy version from the venv; if it's present and < 2, skip install
            check_cmd = [python_exec, "-c", "import numpy as np; print(np.__version__)" ]
            res = subprocess.run(check_cmd, capture_output=True, text=True)
            installed_version = (res.stdout or "").strip()
        except Exception:
            installed_version = ""

        need_install_numpy = True
        if installed_version:
            try:
                major = int(installed_version.split('.')[0])
                if major < 2:
                    need_install_numpy = False
            except Exception:
                # If parsing fails, fall back to installing to be safe
                need_install_numpy = True

        if need_install_numpy:
            safe_update_status("Установка numpy...", "blue")
            # Install numpy constrained to <2 but avoid force-reinstall unless necessary
            subprocess.run([python_exec, "-m", "pip", "install", "numpy<2", "--constraint", CONSTRAINTS_FILE], check=False)
        else:
            safe_update_status(f"numpy {installed_version} уже установлен в venv, пропускаем установку", "green")

        safe_update_status("Установка realesrgan...", "blue")
        subprocess.run([python_exec, "-m", "pip", "install", "realesrgan"], check=False, cwd=REAL_ESRGAN_DIR)

        # Write marker file (best-effort)
        try:
            with open(os.path.join(REAL_ESRGAN_DIR, ".env_ready"), "w", encoding="utf-8") as f:
                f.write("ok")
        except Exception:
            pass

        safe_update_status("Окружение готово.", "green")

    except Exception as exc:
        try:
            messagebox.showerror("Ошибка при подготовке окружения", str(exc))
        except Exception:
            pass
        try:
            gui.root.destroy()
        except Exception:
            pass


def find_python310(gui):
    """Look for or install a local embeddable Python 3.10 and return its path.

    This mirrors the previous `ESRGAN_GUI.find_python310` method but is
    implemented as a standalone function that accepts the `gui` instance so it
    can update UI status and show message boxes.
    Returns the python executable path or None if not found / user cancelled.
    """
    def safe_update_status(text: str, fg: str = "blue"):
        try:
            gui.status.config(text=text, fg=fg)
            gui.root.update_idletasks()
        except Exception:
            pass

    def test_python(path):
        try:
            result = subprocess.run([path, "--version"], capture_output=True, text=True)
            out = (result.stdout or "") + (result.stderr or "")
            return result.returncode == 0 and "3.10" in out
        except Exception:
            return False

    if test_python("python3.10"):
        return "python3.10"

    # Check for previously installed local embeddable Python
    install_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "local_python310"))
    local_python = os.path.join(install_dir, "python.exe")
    marker = os.path.join(install_dir, ".ready")
    if os.path.exists(marker) and os.path.exists(local_python):
        return local_python

    if os.path.exists(local_python) and test_python(local_python):
        return local_python

    common_paths = [
        r"C:\\Program Files\\Python310\\python.exe",
        r"C:\\Program Files (x86)\\Python310\\python.exe",
        os.path.expandvars(r"%LocalAppData%\\Programs\\Python\\Python310\\python.exe"),
    ]
    for path in common_paths:
        if os.path.exists(path) and test_python(path):
            return path

    if os.name == "nt":
        answer = messagebox.askyesno(
            "Python 3.10 не найден",
            "Python 3.10 не найден. Установить локально в папку проекта (без глобальной установки)?"
        )
        if not answer:
            try:
                gui.root.destroy()
            except Exception:
                pass
            return None

        safe_update_status("Загрузка embeddable Python 3.10 и распаковка локально...", "blue")

        # Use embeddable ZIP to avoid system install
        zip_url = "https://www.python.org/ftp/python/3.10.11/python-3.10.11-embed-amd64.zip"
        zip_path = os.path.join(tempfile.gettempdir(), "python310_embed.zip")
        try:
            urllib.request.urlretrieve(zip_url, zip_path)
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось скачать embeddable Python: {e}")
            try:
                gui.root.destroy()
            except Exception:
                pass
            return None

        # Clean and create install dir
        if os.path.exists(install_dir):
            try:
                shutil.rmtree(install_dir)
            except Exception:
                pass
        os.makedirs(install_dir, exist_ok=True)

        try:
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(install_dir)
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось распаковать embeddable Python: {e}")
            try:
                gui.root.destroy()
            except Exception:
                pass
            return None

        local_python = os.path.join(install_dir, "python.exe")
        if not os.path.exists(local_python):
            messagebox.showerror("Ошибка", "python.exe не найден в embeddable раскладке.")
            try:
                gui.root.destroy()
            except Exception:
                pass
            return None

        # Install pip (ensurepip or get-pip.py)
        get_pip_url = "https://bootstrap.pypa.io/get-pip.py"
        get_pip_path = os.path.join(tempfile.gettempdir(), "get-pip.py")
        try:
            urllib.request.urlretrieve(get_pip_url, get_pip_path)
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось скачать get-pip.py: {e}")
            try:
                gui.root.destroy()
            except Exception:
                pass
            return None

        try:
            try:
                subprocess.run([local_python, "-m", "ensurepip", "--upgrade"], check=True)
            except Exception:
                subprocess.run([local_python, get_pip_path], check=True)
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось установить pip в локальном Python: {e}")
            try:
                gui.root.destroy()
            except Exception:
                pass
            return None

        # mark ready
        try:
            with open(marker, "w", encoding="utf-8") as mf:
                mf.write(local_python)
        except Exception:
            pass

        if test_python(local_python):
            return local_python

        messagebox.showerror("Ошибка", "Локальная embeddable установка завершена, но python не работает. Перезапустите приложение.")
        try:
            gui.root.destroy()
        except Exception:
            pass
        return None
    else:
        messagebox.showerror("Ошибка", "Python 3.10 не найден. Установите его вручную.")
        return None


def venv_pip_install(python_exec: str, args, check=True, cwd=None, constraints_file: str = None):
    """Run pip in the given python executable with optional constraints file.

    This is the standalone equivalent of the previous class method. It does
    not depend on GUI state except when called from functions that pass `gui`.
    """
    if not python_exec:
        raise RuntimeError("python_exec is required for venv_pip_install")
    base_cmd = [python_exec, "-m", "pip"]
    cmd = base_cmd + list(args)
    cf = constraints_file or CONSTRAINTS_FILE
    if cf:
        cmd += ["--constraint", cf]
    return subprocess.run(cmd, check=check, cwd=cwd)


def ensure_torch_for_device(python_exec: str, device: str, gui):
    """Ensure an appropriate torch is installed into the venv referenced by python_exec.

    - python_exec: path to the venv/python
    - device: "GPU" or "CPU"
    - gui: GUI instance to update status/messages
    """
    # Check CUDA availability inside the venv
    try:
        result = subprocess.run([python_exec, "-c", "import torch; print(torch.cuda.is_available())"], capture_output=True, text=True)
        cuda_installed = "True" in (result.stdout or result.stderr or "")
    except Exception:
        cuda_installed = False

    torch_args = None
    if device == "GPU":
        if cuda_installed:
            return
        cuda_version = None
        try:
            # try to detect system CUDA via nvidia-smi
            out = subprocess.check_output(["nvidia-smi"], encoding="utf-8")
            for line in out.splitlines():
                if "CUDA Version" in line:
                    cuda_version = line.split("CUDA Version:")[-1].strip().split()[0]
                    break
        except Exception:
            cuda_version = None

        if cuda_version:
            gui.status.config(text=f"CUDA {cuda_version} обнаружена, установка Torch...", fg="blue")
            cuda_str = cuda_version.replace(".", "")
            if cuda_str.startswith("12"):
                torch_args = ["install", "torch==2.1.0", "torchvision==0.16.0", "--index-url", "https://download.pytorch.org/whl/cu121"]
            elif cuda_str.startswith("11"):
                torch_args = ["install", "torch==2.0.1", "torchvision==0.15.2", "--index-url", "https://download.pytorch.org/whl/cu117"]
            else:
                torch_args = ["install", "torch==2.0.1+cpu", "torchvision==0.15.2+cpu", "--index-url", "https://download.pytorch.org/whl/cpu"]
        else:
            gui.status.config(text="CUDA не обнаружена. Установка CPU версии Torch", fg="blue")
            torch_args = ["install", "torch==2.0.1+cpu", "torchvision==0.15.2+cpu", "--index-url", "https://download.pytorch.org/whl/cpu"]
    else:
        if not cuda_installed:
            return
        gui.status.config(text="Установка Torch для CPU...", fg="blue")
        torch_args = ["install", "torch==2.0.1+cpu", "torchvision==0.15.2+cpu", "--index-url", "https://download.pytorch.org/whl/cpu"]

    # Uninstall existing torch versions and reinstall
    try:
        subprocess.run([python_exec, "-m", "pip", "uninstall", "-y", "torch", "torchvision"], check=False)
    except Exception:
        pass

    if torch_args:
        # use venv_pip_install which will apply constraints by default
        venv_pip_install(python_exec, torch_args, check=True)
