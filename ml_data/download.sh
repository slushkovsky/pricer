#gdown https://drive.google.com/uc?id=0B0Ux_fvsLMdAblh1LVNGbjhzYVk -O symbols_iter_10000.caffemodel
#gdown https://drive.google.com/uc?id=0B0Ux_fvsLMdAblh1LVNGbjhzYVk -O symbols_iter_10000.caffemodel

rm -r class_store
mkdir class_store
gdown https://drive.google.com/uc?id=0B0Ux_fvsLMdAWlVYaVdyUnplTXM -O class_store/store_net_dict.json
gdown https://drive.google.com/uc?id=0B0Ux_fvsLMdASl9ESGVvMlc4WWM -O class_store/store_net.prototxt
gdown https://drive.google.com/uc?id=0B7DUPovz20nWUkpQYTZLTzRETlE -O class_store/store_net.caffemodel

