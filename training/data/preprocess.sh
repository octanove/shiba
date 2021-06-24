python preprocess.py \
--input_file $1 \
--output_file $2 \
--min_text_length 10 \
--max_text_length 256 \
--mecab_option "-r "`mecab-config --dicdir`"/mecab-ipadic-neologd/mecabrc"" -d "`mecab-config --dicdir`"/mecab-ipadic-neologd/"