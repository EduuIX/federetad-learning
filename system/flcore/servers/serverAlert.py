import time
from flcore.clients.clientAlert import clientAlert
from flcore.servers.serverbase import Server
from threading import Thread


class FedAlert(Server):
    def __init__(self, args, times):
        """
        Inicializa a instância do servidor e configura os clientes, incluindo a seleção de clientes lentos e a definição 
        do tipo de clientes a serem utilizados (neste caso, `clientAlert`).

        Este método é o construtor da classe, responsável por inicializar os parâmetros do servidor, selecionar os 
        clientes a serem utilizados, e configurar o ambiente necessário para o treinamento. Ele também calcula e 
        exibe informações sobre a taxa de adesão dos clientes e o número total de clientes.

        Fluxo de Execução:
        ------------------
        1. O método `__init__` é chamado com os parâmetros `args` e `times`.
        2. O construtor da classe pai é chamado com `super().__init__(args, times)`, que inicializa os atributos básicos 
        da classe base.
        3. Seleciona os clientes lentos através da função `set_slow_clients`.
        4. Define o tipo de clientes a serem utilizados para treinamento, nesse caso `clientAlert`, através da função 
        `set_clients`.
        5. Exibe informações sobre a taxa de adesão e o número total de clientes.
        6. Inicializa uma lista `Budget` para armazenar os custos de tempo durante o treinamento.

        Parâmetros:
        -----------
        args : tipo de dado
            Parâmetros de configuração do servidor, como opções de treinamento, dados, etc.
        
        times : tipo de dado
            Parâmetro relacionado ao tempo, provavelmente usado para determinar os ciclos ou períodos de treinamento.

        Retorno:
        --------
        Nenhum. O método `__init__` apenas inicializa a instância e configura o ambiente para o servidor e clientes.

        Detalhes:
        ---------
        - O método chama o construtor da classe base para garantir que os atributos da classe pai sejam inicializados corretamente.
        - A função `set_slow_clients` é usada para identificar clientes que devem ter um comportamento mais lento durante o treinamento.
        - A função `set_clients` define que tipo de clientes serão utilizados, neste caso, os clientes da classe `clientAlert`.
        - A taxa de adesão dos clientes e o número total de clientes são exibidos para monitoramento do estado inicial do servidor.
        - A lista `Budget` é inicializada para controlar o tempo de treinamento de cada rodada.

        Exemplos:
        ---------
        # Inicializando o servidor com a configuração fornecida
        server = Server(args=config, times=training_times)

        # Saída esperada:
        # Join ratio / total clients: <taxa de adesão> / <número total de clientes>
        # Finished creating server and clients.
        """

        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientAlert)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []


    def train(self):
        """
        Realiza o treinamento global em várias rodadas, coordenando o envio de modelos para os clientes selecionados, 
        recebendo os modelos atualizados, agregando os parâmetros e realizando avaliações periódicas do modelo global.

        Este método executa o processo de treinamento federado, onde os modelos locais são treinados pelos clientes selecionados 
        e seus pesos são agregados para atualizar o modelo global. A função também inclui avaliações periódicas de desempenho 
        e monitora o tempo de treinamento para cada rodada.

        Fluxo de Execução:
        ------------------
        1. Inicia um loop de treinamento para um número de rodadas determinado por `global_rounds`.
        2. A cada rodada, seleciona os clientes a serem utilizados e envia os modelos globais para treinamento local.
        3. Realiza uma avaliação do modelo global a cada `eval_gap` rodadas.
        4. Cada cliente selecionado realiza uma rodada de treinamento local.
        5. Recebe os modelos treinados de volta dos clientes.
        6. Caso a avaliação de DLG (diferenciação de modelo) seja necessária, é realizada a cada `dlg_gap` rodadas.
        7. Agrega os parâmetros dos modelos locais para atualizar o modelo global.
        8. Monitora o tempo gasto em cada rodada de treinamento e armazena os resultados.
        9. Caso o critério de "auto_break" seja ativado, verifica se o treinamento pode ser interrompido antes de completar 
        todas as rodadas, com base em métricas de desempenho.
        10. Após o fim do treinamento, exibe a melhor precisão e o tempo médio de treinamento por rodada.
        11. Salva os resultados finais e o modelo global.
        12. Caso existam novos clientes, realiza um ajuste fino do modelo para incluir os novos clientes e os avalia.

        Parâmetros:
        -----------
        Nenhum.

        Retorno:
        --------
        Nenhum. O método realiza o treinamento e a avaliação internamente, atualizando o modelo global e salvando os 
        resultados ao final.

        Detalhes:
        ---------
        - O treinamento é realizado para um número de rodadas definido por `global_rounds`.
        - A função `select_clients` seleciona os clientes a serem usados na rodada.
        - A função `send_models` envia o modelo global para os clientes selecionados.
        - A avaliação do modelo é feita periodicamente de acordo com o valor de `eval_gap`, e a função `evaluate` é 
        chamada para realizar essa avaliação.
        - O treinamento local é feito por cada cliente selecionado através da função `train`.
        - Os modelos locais são recebidos e os parâmetros são agregados com a função `aggregate_parameters`.
        - O tempo de treinamento por rodada é registrado e armazenado na lista `Budget`.
        - A função de verificação `check_done` é usada com o parâmetro `auto_break` para determinar se o treinamento pode 
        ser interrompido antes de completar todas as rodadas.
        - Após o término do treinamento, o melhor desempenho e o tempo médio por rodada são impressos.
        - O modelo global final é salvo utilizando a função `save_global_model`.
        - Se houver novos clientes, uma rodada de ajuste fino é realizada para incluir esses novos clientes e avaliar seu 
        desempenho com a função `evaluate`.

        Exemplos:
        ---------
        # Iniciando o treinamento global com o modelo
        trainer.train()

        # Saída esperada:
        - Best accuracy.
        - <Melhor precisão obtida>
        - Average time cost per round.
        - <Tempo médio de cada rodada>
    """

        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()

            for client in self.selected_clients:
                client.train()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_models()
            if self.dlg_eval and i%self.dlg_gap == 0:
                self.call_dlg(i)
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientAlert)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()
