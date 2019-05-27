fasttextbin=/home/hldai/software/fastText-0.2.0/fasttext
datadir=/home/hldai/data/aspect/decnndata
$fasttextbin print-word-vectors $datadir/laptop_emb.vec.bin < data/prep_data/laptop_emb.vec.oov.txt > data/prep_data/laptop_oov.vec
$fasttextbin print-word-vectors $datadir/restaurant_emb.vec.bin < data/prep_data/restaurant_emb.vec.oov.txt > data/prep_data/restaurant_oov.vec

