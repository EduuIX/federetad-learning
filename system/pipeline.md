# 🔄 Pipeline do experimento

1. **Formação das coortes**  
   Distribui os clientes em grupos (coortes), que podem ter tamanhos e proporções diferentes.

2. **Agendamento de drift**  
   Define em quais rounds cada coorte sofrerá mudanças na distribuição dos dados (concept drift).

3. **Inicialização do modelo global**  
   O servidor começa com pesos iniciais e define a estratégia de agregação (FedAlert).

4. **Loop de rounds federados**  
   Para cada rodada:
   
   a. Verifica se há drift agendado e aplica na coorte correspondente.  
   b. Seleciona aleatoriamente uma fração de clientes para participar.  
   c. Clientes treinam localmente com seus dados atuais.  
   d. Cada cliente envia pesos atualizados e métricas ao servidor.  
   e. O servidor agrega os resultados e atualiza o modelo global.

5. **Monitoramento do desempenho (FedAlert)**  
   Métricas são analisadas por coorte, permitindo detectar e reagir ao drift.

6. **Finalização**  
   Resultados e logs são salvos para análise posterior.
