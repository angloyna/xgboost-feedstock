
:: py-xgboost has issue if whole program optimization is enabled so remove
:: it from CFLAGS and CXXFLAGS
set "CFLAGS= -MD"
set "CXXFLAGS= -MD"

mkdir build
cd build
cmake .. -G"Visual Studio 14 2015 Win64" -DUSE_CUDA=ON
cmake --build . --target xgboost --config Release
if errorlevel 1 exit 1
exit 0
