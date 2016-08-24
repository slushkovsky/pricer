rm -r class_store
mkdir class_store
gdown https://drive.google.com/uc?id=0B0Ux_fvsLMdAWlVYaVdyUnplTXM -O class_store/store_net_dict.json
gdown https://drive.google.com/uc?id=0B0Ux_fvsLMdAaEdGMWF2RE12R2s -O class_store/store_net.prototxt
gdown https://drive.google.com/uc?id=0B0Ux_fvsLMdALUNtS0MyMUFoSnM -O class_store/store_net.caffemodel

rm -r class_digits
mkdir class_digits
gdown https://drive.google.com/uc?id=0B0Ux_fvsLMdAYmktSHV4WEFRVGM -O class_digits/digits_net_dict.json
gdown https://drive.google.com/uc?id=0B0Ux_fvsLMdAQmYzTUZoTFU0UzA -O class_digits/digits_net.prototxt
gdown https://drive.google.com/uc?id=0B0Ux_fvsLMdAUUYwczlpVmh5cE0 -O class_digits/digits_net.caffemodel

rm -r class_symb
mkdir class_symb
gdown https://drive.google.com/uc?id=0B0Ux_fvsLMdAS181bHl4M0NXYXM -O class_symb/symbols_net_dict.json
gdown https://drive.google.com/uc?id=0B0Ux_fvsLMdAM2J2b2NmeEZ3UXc -O class_symb/symbols_net.prototxt
gdown https://drive.google.com/uc?id=0B0Ux_fvsLMdAblh1LVNGbjhzYVk -O class_symb/symbols_net.caffemodel

rm -r loc_names
mkdir loc_names
gdown https://drive.google.com/uc?id=0B0Ux_fvsLMdASl9ESGVvMlc4WWM -O loc_names/names_net.prototxt
gdown https://drive.google.com/uc?id=0B0Ux_fvsLMdAQmt0Nmo4amxndlE -O loc_names/names_net.caffemodel

rm -r loc_rubles
mkdir loc_rubles
gdown https://drive.google.com/uc?id=0B0Ux_fvsLMdAeTFkdXp5eFNkVzA -O loc_rubles/rubles_net.prototxt
gdown https://drive.google.com/uc?id=0B0Ux_fvsLMdAUHVmXzBXOXNzRVU -O loc_rubles/rubles_net.caffemodel

rm -r loc_kopecks
mkdir loc_kopecks
gdown https://drive.google.com/uc?id=0B0Ux_fvsLMdAbWc3MlkxdnhLbHM -O loc_kopecks/kopecks_net.prototxt
gdown https://drive.google.com/uc?id=0B0Ux_fvsLMdAX2ZvZjBNQVdEczg -O loc_kopecks/kopecks_net.caffemodel

rm -r NM
mkdir NM
gdown https://drive.google.com/uc?id=0B0Ux_fvsLMdASlE3OVJtMThnUVE -O NM/trained_classifierNM1.xml
gdown https://drive.google.com/uc?id=0B0Ux_fvsLMdAdFl5U2NKcXVBTkE -O NM/trained_classifierNM2.xml


