# Run these commands in two seperate terminals
nice -n 19 python3 /s/chopin/k/grad/sarmst/CR/train_temporal_model.py --size 216 --rank kenai
pdsh -w ^/s/chopin/k/grad/sarmst/CR/hosts python3 /s/chopin/k/grad/sarmst/CR/train_temporal_model.py --size 216 --rank %h
