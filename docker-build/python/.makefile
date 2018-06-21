#!/bin/bash
.DEFAULT_GOAL := help
# .makefile - to be copied into dockerfile to run commands replacing
# shell files.
.DEFAULT help:
	@echo "tfbuild commands"
	@echo "\ttfbuild configure"
	@echo "\t\tRuns Tensorflow's configure, required before building"
	@echo "\ttfbuild build"
	@echo "\t\tBuild tensorflow using bazel - deletes old wheel, if exists."
	@echo "\ttfbuild install"
	@echo "\t\tUses pip to Tensorflow wheel - removes old wheel from install, if exists."
	@echo "\ttfbuild clean"
	@echo "\t\tCleans bazel build tool"
	@echo "\ttfbuild doall"
	@echo "\t\tBuild Tensorflow wheel"
	@echo "\ttfbuild help"
	@echo "\t\tThis screen"
configure:
	cd /root/tensorflow && ./configure
build:
	cd /root/tensorflow \
	&& bazel clean \
	&& bazel build --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" --copt=-msse4.1 --copt=-msse4.2 --copt=-mavx --copt=-mavx2 --copt=-mfma --config=opt //tensorflow/tools/pip_package:build_pip_package \
	&& rm -rf /host/* \
	&& bazel-bin/tensorflow/tools/pip_package/build_pip_package /host/
install:
	-pip uninstall -y tensorflow
	-rm -rf ~/.cache/pip  \ # remove cache, for true uninstall
	pip install --upgrade --force-reinstall /host/tensorflow-*.whl
doall:
	make -f .makefile configure
	make -f .makefile build
	make -f .makefile install
clean:
	cd /root/tensorflow && bazel clean
