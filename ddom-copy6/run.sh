chmod +x run_classifier.sh
chmod +x run_diff_branin.sh

./run_classifier.sh ./configs/new_classifier.cfg branin
./run_diff_branin.sh ./configs/diff_branin.cfg branin

