# Run these commands in two seperate terminals
echo "Start!"
input="/s/chopin/k/grad/sarmst/CR/hosts"
while IFS= read -r line
do
  ssh "$line" "python3 /s/chopin/k/grad/sarmst/CR/train_temporal_model.py" &
done < "$input"

# nice -n 19 python3 /s/chopin/k/grad/sarmst/CR/train_temporal_model.py --size 21 --rank kenai
# pdsh -w ^/s/chopin/k/grad/sarmst/CR/hosts_20 python3 /s/chopin/k/grad/sarmst/CR/train_temporal_model.py --size 21 --rank %h
