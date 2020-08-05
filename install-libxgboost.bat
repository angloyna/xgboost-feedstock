echo on
mkdir %LIBRARY_BIN%
mkdir %LIBRARY_INC%

copy /y xgboost.exe %LIBRARY_BIN%\xgboost.exe
if errorlevel 1 exit 1

copy /y lib\xgboost.dll %LIBRARY_BIN%\xgboost.dll
if errorlevel 1 exit 1

xcopy /s /y /i include\xgboost %LIBRARY_INC%\xgboost
if errorlevel 1 exit 1

xcopy /s /y  src\c_api\*.h %LIBRARY_INC%\xgboost\
if errorlevel 1 exit 1

xcopy /s /y /i rabit\include\rabit %LIBRARY_INC%\xgboost\rabit
if errorlevel 1 exit 1
