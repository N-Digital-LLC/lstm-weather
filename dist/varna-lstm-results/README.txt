Varna Weather LSTM — results explorer (offline package)
=======================================================

This is a self-contained, READ-ONLY snapshot of the finished LSTM weather
experiments. It runs entirely in your web browser — there is no model
training and nothing is sent over the internet. All numbers and charts come
from JSON files bundled in the "snapshots" folder (real results, captured
from the trained model).

What you can explore
--------------------
  • Forecast      — pick a date/horizon; see the LSTM vs. the baselines and
                    the actual measured temperature (a fixed set of real
                    backtests is precomputed).
  • Training      — the full table of all 128 runs; click "View" on any run.
  • Run report    — config, data split, test metrics, training curve, and
                    RMSE / MAE / Bias vs. forecast-horizon charts.
  • Comparison    — put several runs side by side.


HOW TO RUN IT
=============

A static website must be served by a tiny local web server (just opening
index.html directly will NOT work because the browser blocks loading the
data files from a file:// path).

Option 1 — double-click start.bat  (recommended)
------------------------------------------------
  1. Double-click  start.bat
  2. A small black window opens (the local server) and your browser opens at
     http://localhost:8099/
  3. Browse the app. When finished, close the black window.

  start.bat automatically uses Python or Node.js if either is installed
  (most machines have one). If neither is found, install Python from
  https://www.python.org/downloads/  (tick "Add python.exe to PATH"),
  then double-click start.bat again.

Option 2 — run a server yourself
--------------------------------
  Open a terminal in THIS folder and run any one of:

    python -m http.server 8099
    py -m http.server 8099
    npx serve -l 8099 .

  Then open  http://localhost:8099/  in your browser.

Option 3 — VS Code
------------------
  Open this folder in VS Code, install the "Live Server" extension,
  right-click index.html → "Open with Live Server".


Notes
-----
  • Everything works offline; no backend and no internet are required.
  • The default port is 8099. If it is busy, edit PORT in start.bat or pass a
    different port to the commands above.
  • Weather data: Open-Meteo (ECMWF ERA5), CC BY 4.0.
