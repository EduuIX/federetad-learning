python main.py -data fmnist -m cnn -algo FedAvg -gr 100 -nc 58 -did 0 -mal True -mv 10 2>&1 | tee "command1.log"

cat "command1.log" > "log_NormalSelection.txt"

mv saida.txt "NormalSelection.txt"
mv NormalSelection.txt log_NormalSelection.txt ./saidas/



python main.py -data fmnist -m cnn -algo FedAvg -gr 100 -nc 58 -did 0 -ent True -mal True -mv 10 2>&1 | tee "command2.log"

cat "command2.log" > "log_BestEntropy.txt"

mv saida.txt "BestEntropy.txt"
mv BestEntropy.txt log_BestEntropy.txt ./saidas/



python main.py -data fmnist -m cnn -algo FedAvg -gr 100 -nc 58 -did 0 -ba True -mal True -mv 10 2>&1 | tee "command3.log"

cat "command3.log" > "log_BellowAverage.txt"

mv saida.txt "BellowAverage.txt"
mv BellowAverage.txt log_BellowAverage.txt ./saidas/

python main.py -data fmnist -m cnn -algo FedAvg -gr 100 -nc 58 -did 0 -pow True -mal True -mv 10 2>&1 | tee "command4.log"

cat "command4.log" > "log_Power_of_Choice.txt"

mv saida.txt "Power_of_Choice.txt"
mv Power_of_Choice.txt log_Power_of_Choice.txt ./saidas/

rm "command1.log" "command2.log" "command3.log" "command4.log"