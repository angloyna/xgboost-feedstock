#!/bin/bash

set -e
set -x

mv xgboost-$PKG_VERSION-py3-none-any.whl.dummy xgboost-$PKG_VERSION-py3-none-any.whl
$PYTHON -m pip install --no-deps xgboost-$PKG_VERSION-py3-none-any.whl
