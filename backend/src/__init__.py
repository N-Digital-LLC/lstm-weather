"""Backend source package for the Varna hourly-weather LSTM project."""

# On Windows, torch's c10.dll must initialize its OpenMP runtime before pandas/numpy
# load theirs; otherwise importing pandas first makes torch fail with
# "OSError: [WinError 1114] A dynamic link library (DLL) initialization routine failed".
# Importing torch here — before any submodule body runs — guarantees the safe order.
import torch as _torch  # noqa: F401
