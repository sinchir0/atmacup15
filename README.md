# atmacup15
https://www.guruguru.science/competitions/21/

# Docker環境の立ち上げ
docker run -it --name atmacup15 -v .:/app python:3.11.4-bookworm /bin/bash
その後、vscodeのOpen In Browser

```
# update
apt update
# Vim
apt install vim
# zsh
apt install zsh
# zsh-autosuggestions
https://github.com/zsh-users/zsh-autosuggestions/blob/master/INSTALL.md
# starship
https://starship.rs/ja-jp/guide/
# lightgbm
apt install cmake
https://lightgbm.readthedocs.io/en/latest/Installation-Guide.html#linux
# poetry
https://python-poetry.org/docs/
poetry config virtualenvs.in-project true --local
# Extention
vscodeのextentionとして、pythonとjupyterをinstall


# TODO
takapyさんみたいに自動で分析環境が出来上がるDockerFileを作りたい
https://github.com/takapy0210/Dockerfile-for-MachineLearning/blob/master/Dockerfile

```