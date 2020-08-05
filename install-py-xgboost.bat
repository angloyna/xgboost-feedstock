echo on

cd python-package

where python
%PYTHON% setup.py install --single-version-externally-managed --record=record.txt
if errorlevel 1 exit 1
