sudo apt-get install mecab libmecab-dev mecab-ipadic-utf8 git make curl xz-utils file
git clone --depth 1 https://github.com/neologd/mecab-ipadic-neologd.git
cd mecab-ipadic-neologd
yes yes | ./bin/install-mecab-ipadic-neologd -n
sudo touch `mecab-config --dicdir`"/mecab-ipadic-neologd/mecabrc" # file needs to exist, even if it's empty