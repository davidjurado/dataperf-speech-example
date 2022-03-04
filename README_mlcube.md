# DataPerf Speech Example - MLCube integration

Project information: [Readme](README.md)

## Project setup

```bash
# Create Python environment and install MLCube Docker runner 
virtualenv -p python3 ./env && source ./env/bin/activate && pip install mlcube-docker

# Fetch the implementation from GitHub
git clone https://github.com/harvard-edge/dataperf-speech-example && cd ./dataperf-speech-example
git fetch origin pull/xx/head:feature/MLCube-integration && git checkout feature/MLCube-integration
```

## Tasks execution

```bash
# Run download task
mlcube run --task=download -Pdocker.build_strategy=always

# Run select task
mlcube run --task=select -Pdocker.build_strategy=always

# Run evaluate task
mlcube run --task=evaluate -Pdocker.build_strategy=always
```