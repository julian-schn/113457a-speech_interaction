# Python Virtual Environment Mini-Tutorial

## Why Use a Virtual Environment?
- Keeps project dependencies isolated so they do not conflict with global packages.
- Allows you to pin exact versions per project for reproducible setups.
- Makes it easy to clean up: deleting the folder removes the environment.

## Prerequisites
- Python 3.8+ installed (`python3 --version` to check).
- `pip` available (bundled with recent Python releases).
- Recommended: keep the virtual environment folder out of version control (e.g., add `venv/` to `.gitignore`).

## 1. Create the Environment
```bash
python3 -m venv venv
```
- `venv` at the end is the folder name; change it if you prefer (`.venv`, `env`, etc.).
- You can place the environment anywhere—just activate from its path—but keeping it beside your project makes onboarding simple.
- On Windows PowerShell, the command is the same if `python` maps to Python 3: `python -m venv venv`.

## 2. Activate It
- **macOS / Linux (bash/zsh):**
  ```bash
  source venv/bin/activate
  ```
- **Windows PowerShell:**
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```
- **Windows CMD:**
  ```cmd
  venv\Scripts\activate.bat
  ```
When active, you will see the environment name prefixed in your shell prompt.

## 3. Install Project Dependencies
```bash
pip install -r requirements.txt
```
- If there is no `requirements.txt`, install packages manually (`pip install numpy`).
- Save the current package list so teammates can reproduce it:
  ```bash
  pip freeze > requirements.txt
  ```

## 4. Deactivate When Finished
```bash
deactivate
```
- Closing the terminal session also deactivates it automatically.

## 5. Removing / Rebuilding the Environment
- Delete the folder to remove it completely:
  ```bash
  rm -rf venv
  ```
- Re-run steps 1–3 to recreate from scratch if dependencies break.

## Tips & Troubleshooting
- If activation fails on Windows, allow script execution: `Set-ExecutionPolicy -Scope Process RemoteSigned`.
- Use `python -m pip` to ensure you are using the `pip` tied to the active environment.
- Integrate with IDEs by selecting the interpreter inside `venv/bin/python` (or `venv\Scripts\python.exe` on Windows).
