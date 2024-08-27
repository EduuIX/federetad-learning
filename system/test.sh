cdr_values=(0.16)
jr_values=(0.20)
rc_values=(0 1 2 3)
for cdr_value in "${cdr_values[@]}"; do
    for jr_value in "${jr_values[@]}"; do
        for rc_value in "${rc_values[@]}"; do
        
            python main.py -data fmnist -m cnn -algo FedAvg -gr 100 -nc 60 -jr "$jr_value" -cdr "$cdr_value" -rc "$rc_value" -did 0 2>&1 | tee "command1_$cdr_value+_$jr_value+_$rc_value.log"
            
            cat "command1_$cdr_value+_$jr_value+_$rc_value.log" > "log_NormalSelection$cdr_value+_$jr_value+_$rc_value.txt"

            mv saida.txt "NormalSelection_$cdr_value+_$jr_value+_$rc_value.txt"
            mv NormalSelection_$cdr_value+_$jr_value+_$rc_value.txt ./saidas/NormalSelection/



            python main.py -data fmnist -m cnn -algo FedAvg -gr 100 -nc 60 -jr "$jr_value" -cdr "$cdr_value" -rc "$rc_value" -did 0 -ent True 2>&1 | tee "command2_$cdr_value+_$jr_value+_$rc_value.log"

            cat "command2_$cdr_value+_$jr_value+_$rc_value.log" > "log_BestEntropy$cdr_value+_$jr_value+_$rc_value.txt"

            mv saida.txt "BestEntropy_$cdr_value+_$jr_value+_$rc_value.txt"
            mv BestEntropy_$cdr_value+_$jr_value+_$rc_value.txt ./saidas/BestEntropy/



            python main.py -data fmnist -m cnn -algo FedAvg -gr 100 -nc 60 -jr "$jr_value" -cdr "$cdr_value" -rc "$rc_value" -did 0 -ba True 2>&1 | tee "command3_$cdr_value+_$jr_value+_$rc_value.log"
            
            cat "command3_$cdr_value+_$jr_value+_$rc_value.log" > "log_BellowAverage_$cdr_value+_$jr_value+_$rc_value.txt"

            mv saida.txt "BellowAverage_$cdr_value+_$jr_value+_$rc_value.txt"
            mv BellowAverage_$cdr_value+_$jr_value+_$rc_value.txt ./saidas/BellowAverage/



            rm "command1_$cdr_value+_$jr_value+_$rc_value.log" "command2_$cdr_value+_$jr_value+_$rc_value.log" "command3_$cdr_value+_$jr_value+_$rc_value.log"
        done
    done
done