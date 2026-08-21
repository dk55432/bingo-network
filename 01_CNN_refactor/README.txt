To install package dependencies:
python -m pip install --upgrade pip
python -m pip install torch torchvision pillow

3. Tell VSCode to use the right Python

Even after installing the packages, VSCode may show warnings if it is using a different Python interpreter.

In VSCode:

    Press Ctrl+Shift+P on Windows/Linux, or Cmd+Shift+P on macOS.
    Search for Python: Select Interpreter.
    Select the interpreter inside your .venv folder.

It will usually be one of these:
text

.venv\Scripts\python.exe

or:
text

.venv/bin/python

Then close and reopen the Python file, or reload VSCode.

