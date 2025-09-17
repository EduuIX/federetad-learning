import torch
import os
import numpy as np
import h5py
import copy
import time
import random
from utils.data_utils import read_client_data
from utils.dlg import DLG


class Server(object):
    def __init__(self, args, times):
        """
            Inicializa os parâmetros e configurações principais para o treinamento do modelo em um cenário distribuído.

            Este método de inicialização configura os atributos principais, como parâmetros do modelo, 
            número de clientes, número de rodadas globais e locais, taxa de aprendizado, entre outros, 
            para a execução de treinamento de aprendizado federado ou distribuído. Além disso, o método 
            define variáveis de controle, como as listas de clientes, taxas de treinamento e envio lentos, 
            e configurações de avaliação e controle de treinamento.

            Parâmetros:
            -----------
            args : objeto
                Um objeto contendo os parâmetros necessários para a configuração do modelo e treinamento, 
                incluindo atributos como `device`, `dataset`, `num_classes`, `global_rounds`, `local_epochs`, 
                `batch_size`, `local_learning_rate`, entre outros. Esses parâmetros são utilizados para configurar 
                o ambiente de treinamento.

            times : list
                Uma lista de tempos ou parâmetros relacionados a eventos temporais ou para controle de execução 
                do treinamento, podendo ser usada para configurar lapsos de avaliação ou execução.

            Atributos:
            ----------
            args : objeto
                O objeto contendo os parâmetros configuráveis para o treinamento.
                
            device : str
                O dispositivo (CPU ou GPU) onde o modelo será treinado.

            dataset : str
                O nome do conjunto de dados a ser utilizado para o treinamento.

            num_classes : int
                O número de classes no problema de classificação.

            global_rounds : int
                O número total de rodadas globais (iterações) de treinamento.

            local_epochs : int
                O número de épocas locais de treinamento.

            batch_size : int
                O tamanho do lote a ser utilizado durante o treinamento.

            learning_rate : float
                A taxa de aprendizado local para o treinamento.

            global_model : torch.nn.Module
                O modelo global, copiado a partir de `args.model`.

            num_clients : int
                O número total de clientes participantes no treinamento.

            join_ratio : float
                A razão de participação dos clientes em cada rodada de treinamento.

            random_join_ratio : float
                A razão de clientes escolhidos aleatoriamente para participar.

            num_join_clients : int
                O número de clientes que irão participar do treinamento em cada rodada.

            current_num_join_clients : int
                O número atual de clientes participando no treinamento.

            few_shot : bool
                Indica se o treinamento será realizado em um cenário de poucos exemplos (few-shot).

            algorithm : str
                O algoritmo de treinamento a ser utilizado.

            time_select : str
                A estratégia de seleção de clientes, baseada em tempo ou outros critérios.

            goal : str
                O objetivo do treinamento, por exemplo, melhorar a precisão ou reduzir a perda.

            time_threthold : float
                O limiar de tempo utilizado para controlar o tempo de execução do treinamento.

            save_folder_name : str
                O diretório onde os modelos e resultados serão salvos.

            top_cnt : int
                O número de melhores modelos ou clientes a serem selecionados para treinamento.

            auto_break : bool
                Indica se o treinamento será interrompido automaticamente sob determinadas condições.

            clients : list
                Lista de todos os clientes envolvidos no treinamento.

            selected_clients : list
                Lista de clientes selecionados para a rodada atual de treinamento.

            train_slow_clients : list
                Lista de clientes com taxa de treinamento mais lenta.

            send_slow_clients : list
                Lista de clientes com taxa de envio de dados mais lenta.

            uploaded_weights : list
                Lista de pesos dos modelos enviados pelos clientes.

            uploaded_ids : list
                Lista de IDs dos clientes que enviaram seus modelos.

            uploaded_models : list
                Lista de modelos completos enviados pelos clientes.

            rs_test_acc : list
                Lista de acurácias de teste para cada rodada de treinamento.

            rs_test_auc : list
                Lista de AUCs (Área sob a Curva ROC) para cada rodada de treinamento.

            rs_train_loss : list
                Lista das perdas de treinamento para cada rodada.

            times : list
                Parâmetros relacionados a tempos ou intervalos usados durante o treinamento.

            eval_gap : int
                A cada quantas rodadas de treinamento uma avaliação será realizada.

            client_drop_rate : float
                A taxa de clientes que podem ser descartados em cada rodada.

            train_slow_rate : float
                A taxa de clientes com treinamento mais lento.

            send_slow_rate : float
                A taxa de clientes com envio de dados mais lento.

            dlg_eval : bool
                Se as avaliações de desempenho do modelo serão feitas de forma dialogada.

            dlg_gap : int
                O intervalo entre avaliações dialogadas.

            batch_num_per_client : int
                O número de lotes de dados que cada cliente processará por rodada.

            num_new_clients : int
                O número de novos clientes que serão adicionados ao treinamento.

            new_clients : list
                Lista de novos clientes a serem adicionados no treinamento.

            eval_new_clients : bool
                Indica se os novos clientes serão avaliados durante o treinamento.

            fine_tuning_epoch_new : int
                O número de épocas de fine-tuning para os novos clientes.

            Exemplo:
            --------
            >>> args = Namespace(device='cuda', dataset='CIFAR-10', num_classes=10, global_rounds=100, 
            >>>                   local_epochs=10, batch_size=32, local_learning_rate=0.01, model=my_model, 
            >>>                   num_clients=10, join_ratio=0.5, algorithm='FedAvg', save_folder_name='./models')
            >>> times = [0, 1, 2, 3]
            >>> trainer = FederatedTrainer(args, times)
        """

        # Set up the main attributes
        self.args = args
        self.device = args.device
        self.dataset = args.dataset
        self.num_classes = args.num_classes
        self.global_rounds = args.global_rounds
        self.local_epochs = args.local_epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.global_model = copy.deepcopy(args.model)
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.random_join_ratio = args.random_join_ratio
        self.num_join_clients = int(self.num_clients * self.join_ratio)
        self.current_num_join_clients = self.num_join_clients
        self.few_shot = args.few_shot
        self.algorithm = args.algorithm
        self.time_select = args.time_select
        self.goal = args.goal
        self.time_threthold = args.time_threthold
        self.save_folder_name = args.save_folder_name
        self.top_cnt = args.top_cnt
        self.auto_break = args.auto_break

        self.clients = []
        self.selected_clients = []
        self.train_slow_clients = []
        self.send_slow_clients = []

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []

        self.rs_test_acc = []
        self.rs_test_auc = []
        self.rs_train_loss = []

        self.times = times
        self.eval_gap = args.eval_gap
        self.client_drop_rate = args.client_drop_rate
        self.train_slow_rate = args.train_slow_rate
        self.send_slow_rate = args.send_slow_rate

        self.dlg_eval = args.dlg_eval
        self.dlg_gap = args.dlg_gap
        self.batch_num_per_client = args.batch_num_per_client

        self.num_new_clients = args.num_new_clients
        self.new_clients = []
        self.eval_new_clients = False
        self.fine_tuning_epoch_new = args.fine_tuning_epoch_new


    def set_clients(self, clientObj):
        """
            Configura e inicializa os clientes para o treinamento federado.

            Esta função cria e adiciona instâncias de clientes ao sistema, utilizando o objeto `clientObj` 
            para instanciar cada cliente com base nos dados de treinamento e teste, além das informações 
            relacionadas ao comportamento do cliente, como taxas de treinamento lento e envio lento.

            Para cada cliente, a função lê os dados de treinamento e teste usando a função `read_client_data()` 
            e passa essas informações, junto com as configurações específicas, para a criação do objeto do cliente.

            Parâmetros:
            -----------
            clientObj : class
                A classe do cliente a ser instanciado. Cada cliente é criado utilizando os parâmetros fornecidos 
                e a classe `clientObj` é responsável por inicializar as instâncias de cliente com base nos dados e 
                configurações fornecidas.

            Detalhes de Implementação:
            --------------------------
            - Para cada cliente, a função lê os dados de treinamento e teste correspondentes ao cliente usando 
            a função `read_client_data()`, com base no índice do cliente (`i`).
            - A função também considera se o cliente está em um cenário de poucos exemplos (few-shot) e as taxas 
            de treinamento lento e envio lento, passando essas informações para a criação de cada cliente.
            - Cada cliente criado é adicionado à lista `self.clients`, permitindo que os clientes sejam posteriormente 
            utilizados no treinamento federado.

            Exemplo:
            --------
            >>> trainer.set_clients(Client)
            >>> # Cada cliente será inicializado com os dados de treinamento e teste e adicionado à lista de clientes
        """
        for i, train_slow, send_slow in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients):
            train_data = read_client_data(self.dataset, i, is_train=True, few_shot=self.few_shot)
            test_data = read_client_data(self.dataset, i, is_train=False, few_shot=self.few_shot)
            client = clientObj(self.args, 
                            id=i, 
                            train_samples=len(train_data), 
                            test_samples=len(test_data), 
                            train_slow=train_slow, 
                            send_slow=send_slow)
            self.clients.append(client)


    def select_slow_clients(self, slow_rate):
        """
            Seleciona clientes com taxa de treinamento e envio lento.

            Esta função escolhe aleatoriamente uma porcentagem dos clientes para que eles tenham 
            uma taxa de treinamento e/ou envio lenta, com base na taxa de lentidão fornecida (`slow_rate`). 
            A função gera uma lista booleana onde `True` indica que o cliente correspondente foi 
            selecionado como um cliente "lento" e `False` caso contrário.

            Parâmetros:
            -----------
            slow_rate : float
                A taxa de clientes lentos a ser selecionada, representando a fração do total de clientes 
                que terão comportamento de treinamento e envio lento.

            Retorna:
            --------
            slow_clients : list
                Uma lista booleana de tamanho `num_clients`, onde `True` indica que o cliente tem 
                taxa de treinamento e envio lento, e `False` indica que o cliente não é lento.

            Detalhes de Implementação:
            --------------------------
            - A função começa criando uma lista `slow_clients` com todos os valores definidos como `False`.
            - Em seguida, ela seleciona aleatoriamente uma quantidade de clientes com base na taxa `slow_rate`, 
            e os marca como `True` na lista `slow_clients`.
            - A seleção aleatória é feita utilizando `np.random.choice()`, que escolhe índices dos clientes de 
            forma aleatória.

            Exemplo:
            --------
            >>> slow_clients = select_slow_clients(0.2)
            >>> # Aproximadamente 20% dos clientes serão selecionados como lentos (marcados como True)
        """
        slow_clients = [False for i in range(self.num_clients)]
        idx = [i for i in range(self.num_clients)]
        idx_ = np.random.choice(idx, int(slow_rate * self.num_clients))
        for i in idx_:
            slow_clients[i] = True

        return slow_clients


    def set_slow_clients(self):
        """
            Define os clientes com taxas de treinamento e envio lento.

            Esta função utiliza a função `select_slow_clients` para selecionar aleatoriamente os 
            clientes com taxas de treinamento lento e envio lento, com base nas taxas fornecidas 
            (`train_slow_rate` e `send_slow_rate`). A lista de clientes lentos para treinamento 
            e envio é atualizada com os valores retornados pela função de seleção.

            A função atualiza duas listas:
            - `self.train_slow_clients`: Lista de clientes selecionados com treinamento lento.
            - `self.send_slow_clients`: Lista de clientes selecionados com envio lento.

            Detalhes de Implementação:
            --------------------------
            - A função chama `select_slow_clients` duas vezes, uma para selecionar os clientes 
            com treinamento lento e outra para selecionar os clientes com envio lento.
            - Os resultados das seleções são armazenados nas listas `self.train_slow_clients` 
            e `self.send_slow_clients`, respectivamente.

            Exemplo:
            --------
            >>> set_slow_clients()
            >>> # As listas 'train_slow_clients' e 'send_slow_clients' serão preenchidas com os clientes lentos
        """
        self.train_slow_clients = self.select_slow_clients(
            self.train_slow_rate)
        self.send_slow_clients = self.select_slow_clients(
            self.send_slow_rate)


    def select_clients(self):
        """
            Seleciona um número de clientes para participar da rodada de treinamento.

            Esta função seleciona aleatoriamente um número de clientes que irão participar do treinamento 
            na rodada atual. O número de clientes pode ser ajustado de duas maneiras:
            - Se `random_join_ratio` for `True`, um número aleatório de clientes será selecionado, 
            variando entre `num_join_clients` e `num_clients`.
            - Caso contrário, o número de clientes selecionados será fixo, definido por `num_join_clients`.

            Retorna:
            --------
            selected_clients : list
                Uma lista contendo os clientes selecionados para participar da rodada de treinamento.

            Detalhes de Implementação:
            --------------------------
            - A função verifica se a seleção de clientes deve ser aleatória (`random_join_ratio`).
            - Se `random_join_ratio` for `True`, o número de clientes participantes é aleatoriamente 
            escolhido entre `num_join_clients` e `num_clients`, utilizando `np.random.choice()`.
            - Se `random_join_ratio` for `False`, o número de clientes participantes é fixo e 
            definido por `num_join_clients`.
            - Os clientes selecionados são escolhidos aleatoriamente da lista `self.clients`, 
            e o número de clientes selecionados é determinado pela variável `current_num_join_clients`.

            Exemplo:
            --------
            >>> selected_clients = select_clients()
            >>> # A lista 'selected_clients' conterá o número de clientes selecionados aleatoriamente
        """
        if self.random_join_ratio:
            self.current_num_join_clients = np.random.choice(range(self.num_join_clients, self.num_clients+1), 1, replace=False)[0]
        else:
            self.current_num_join_clients = self.num_join_clients
        selected_clients = list(np.random.choice(self.clients, self.current_num_join_clients, replace=False))

        return selected_clients


    def send_models(self):
        """
            Envia os parâmetros do modelo global para todos os clientes.

            Esta função envia os parâmetros do modelo global para todos os clientes, 
            atualizando o modelo de cada cliente com os parâmetros globais. Além disso, 
            a função mede o tempo de envio para cada cliente e registra o custo de tempo 
            de envio em cada cliente.

            Detalhes de Implementação:
            --------------------------
            - A função começa com uma asserção para garantir que há pelo menos um cliente na lista `self.clients`.
            - Para cada cliente na lista `self.clients`, a função realiza os seguintes passos:
            - Registra o tempo de início do envio dos parâmetros (`start_time`).
            - Chama o método `set_parameters` de cada cliente para atualizar os parâmetros do cliente com os parâmetros do modelo global.
            - Atualiza os custos de tempo de envio do cliente, incrementando o número de rodadas e o custo total com o tempo gasto no envio.
            - O custo de tempo é calculado como o dobro do tempo decorrido desde `start_time`, representando o tempo gasto no envio e atualização do modelo.

            Parâmetros:
            -----------
            Nenhum parâmetro adicional.

            Exemplo:
            --------
            >>> send_models()
            >>> # Os parâmetros do modelo global serão enviados para todos os clientes e o custo de tempo será registrado
        """
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()
            
            client.set_parameters(self.global_model)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)


    def receive_models(self):
        """
            Recebe e processa os modelos dos clientes ativos selecionados.

            Esta função seleciona aleatoriamente uma fração dos clientes ativos com base na taxa 
            de queda de clientes (`client_drop_rate`), e para cada cliente ativo, ela verifica 
            se o custo de tempo do cliente (tempo de treinamento e envio) está abaixo de um limiar 
            (`time_threthold`). Se estiver, o modelo do cliente é coletado, juntamente com os pesos 
            e o ID do cliente, e as amostras de treinamento são contabilizadas.

            O custo de tempo total de cada cliente é calculado a partir dos custos de treinamento e 
            envio, e os modelos dos clientes são ponderados de acordo com o número de amostras de 
            treinamento, com o peso normalizado pela soma total de amostras de todos os clientes ativos.

            Detalhes de Implementação:
            --------------------------
            - A função começa com uma asserção para garantir que há pelo menos um cliente selecionado (`self.selected_clients`).
            - A função seleciona aleatoriamente uma fração dos clientes ativos, usando a taxa de queda de clientes (`client_drop_rate`).
            - Para cada cliente ativo, a função calcula o custo de tempo total, considerando os tempos de treinamento e envio.
            - Se o custo de tempo de um cliente for menor ou igual ao limiar de tempo (`time_threthold`), as amostras do cliente são somadas 
            e o modelo do cliente é adicionado à lista de modelos recebidos.
            - A função calcula os pesos dos modelos com base no número de amostras de cada cliente, normalizando-os pela soma total das amostras.

            Parâmetros:
            -----------
            Nenhum parâmetro adicional.

            Exemplo:
            --------
            >>> receive_models()
            >>> # Os modelos dos clientes ativos são recebidos e os pesos são ajustados conforme o número de amostras
        """
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int((1-self.client_drop_rate) * self.current_num_join_clients))

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0
        for client in active_clients:
            try:
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                        client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost <= self.time_threthold:
                tot_samples += client.train_samples
                self.uploaded_ids.append(client.id)
                self.uploaded_weights.append(client.train_samples)
                self.uploaded_models.append(client.model)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples


    def aggregate_parameters(self):
        """
            Agrega os parâmetros dos modelos dos clientes para atualizar o modelo global.

            Esta função agrega os parâmetros dos modelos recebidos de todos os clientes ativos 
            para atualizar o modelo global. O modelo global é inicializado com os parâmetros do 
            primeiro modelo recebido e, em seguida, os parâmetros de cada modelo do cliente são 
            somados ao modelo global, ponderados de acordo com o número de amostras de treinamento 
            de cada cliente.

            Detalhes de Implementação:
            --------------------------
            - A função começa com uma asserção para garantir que pelo menos um modelo foi recebido 
            (`self.uploaded_models`).
            - O modelo global (`self.global_model`) é inicializado como uma cópia profunda do primeiro 
            modelo recebido, e seus parâmetros são zerados.
            - Para cada cliente, os parâmetros de seu modelo são adicionados ao modelo global, levando 
            em consideração os pesos relativos de cada cliente, que são baseados no número de amostras 
            de treinamento do cliente. A função `add_parameters` é chamada para somar os parâmetros 
            ponderados ao modelo global.

            Parâmetros:
            -----------
            Nenhum parâmetro adicional.

            Exemplo:
            --------
            >>> aggregate_parameters()
            >>> # O modelo global será atualizado com os parâmetros agregados dos modelos dos clientes
        """

        assert (len(self.uploaded_models) > 0)

        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data.zero_()
            
        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w, client_model)


    def add_parameters(self, w, client_model):
        """
            Adiciona os parâmetros de um modelo de cliente ao modelo global, ponderados por um fator.

            Esta função adiciona os parâmetros do modelo de um cliente ao modelo global, com base 
            no peso `w`, que geralmente é proporcional ao número de amostras de treinamento do cliente. 
            Os parâmetros do cliente são clonados para garantir que não haja referências compartilhadas, 
            e a soma ponderada dos parâmetros é realizada no modelo global.

            Parâmetros:
            -----------
            w : float
                O peso a ser aplicado aos parâmetros do cliente. Normalmente, esse peso é baseado 
                no número de amostras de treinamento do cliente em relação ao total de amostras.

            client_model : torch.nn.Module
                O modelo do cliente cujos parâmetros serão adicionados ao modelo global.

            Detalhes de Implementação:
            --------------------------
            - A função itera sobre os parâmetros do modelo global e do modelo do cliente simultaneamente 
            usando `zip()`.
            - Para cada par de parâmetros (um do modelo global e um do modelo do cliente), os parâmetros 
            do cliente são clonados usando `clone()` para evitar referências compartilhadas.
            - Os parâmetros clonados do cliente são multiplicados pelo peso `w` e somados aos parâmetros 
            correspondentes no modelo global.

            Exemplo:
            --------
            >>> add_parameters(w=0.5, client_model=model_1)
            >>> # Os parâmetros do modelo do cliente serão somados ao modelo global, ponderados por w.
        """
        for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w


    def save_global_model(self):
        """
        Salva o modelo global no diretório de modelos.

        Esta função salva o modelo global no diretório `models`, dentro de uma subpasta nomeada 
        de acordo com o conjunto de dados (`self.dataset`). Se a pasta não existir, ela será criada 
        automaticamente. O modelo global é salvo com um nome que inclui o algoritmo utilizado 
        (`self.algorithm`), seguido do sufixo `_server` e a extensão `.pt`.

        Detalhes de Implementação:
        --------------------------
        - A função cria o caminho completo para salvar o modelo, combinando o diretório `models` 
        com o nome do conjunto de dados (`self.dataset`).
        - Se o diretório de destino não existir, ele é criado usando `os.makedirs()`.
        - O caminho final do arquivo é gerado combinando o diretório de destino, o nome do algoritmo 
        e a extensão `.pt`.
        - O modelo global (`self.global_model`) é salvo no caminho especificado usando `torch.save()`.

        Parâmetros:
        -----------
        Nenhum parâmetro adicional.

        Exemplo:
        --------
        >>> save_global_model()
        >>> # O modelo global será salvo no diretório correspondente ao conjunto de dados e algoritmo
        """
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        torch.save(self.global_model, model_path)


    def load_model(self):
        """
        Carrega o modelo global a partir de um arquivo salvo.

        Esta função carrega o modelo global de um arquivo armazenado no diretório `models`, 
        dentro de uma subpasta nomeada de acordo com o conjunto de dados (`self.dataset`). 
        O arquivo do modelo é identificado pelo nome que inclui o algoritmo utilizado (`self.algorithm`), 
        seguido do sufixo `_server` e a extensão `.pt`.

        Detalhes de Implementação:
        --------------------------
        - O caminho do arquivo do modelo é gerado combinando o diretório `models`, o nome do 
        conjunto de dados (`self.dataset`), e o nome do algoritmo (`self.algorithm`).
        - A função verifica se o arquivo do modelo existe no caminho gerado, utilizando `assert`.
        - O modelo global é carregado utilizando `torch.load()` e atribuído a `self.global_model`.

        Parâmetros:
        -----------
        Nenhum parâmetro adicional.

        Exemplo:
        --------
        >>> load_model()
        >>> # O modelo global será carregado do diretório especificado e atribuído a 'self.global_model'
        """
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        assert (os.path.exists(model_path))
        self.global_model = torch.load(model_path)


    def model_exists(self):
        """
        Verifica se o modelo global existe no diretório de modelos.

        Esta função verifica a existência de um arquivo de modelo no diretório `models`, 
        dentro de uma subpasta nomeada de acordo com o conjunto de dados (`self.dataset`). 
        O arquivo do modelo é identificado pelo nome que inclui o algoritmo utilizado (`self.algorithm`), 
        seguido do sufixo `_server` e a extensão `.pt`.

        Retorna:
        --------
        bool
        `True` se o arquivo do modelo global existir no caminho especificado, 
        caso contrário, retorna `False`.

        Detalhes de Implementação:
        --------------------------
        - A função gera o caminho completo para o arquivo do modelo, combinando o diretório `models`, 
        o nome do conjunto de dados (`self.dataset`) e o nome do algoritmo (`self.algorithm`).
        - A existência do arquivo é verificada usando `os.path.exists()`.

        Exemplo:
        --------
        >>> if model_exists():
        >>>     print("O modelo global existe.")
        >>> else:
        >>>     print("O modelo global não foi encontrado.")
        """
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        return os.path.exists(model_path)

        
    def save_results(self):
        """
        Salva os resultados do treinamento em um arquivo HDF5.

        Esta função salva os resultados do treinamento, como acurácia de teste, AUC de teste e 
        perda de treinamento, em um arquivo HDF5. O nome do arquivo é gerado com base no nome do 
        conjunto de dados (`self.dataset`), no algoritmo utilizado (`self.algorithm`), no objetivo 
        do treinamento (`self.goal`), e nos tempos registrados (`self.times`). O arquivo é armazenado 
        em um diretório chamado `../results/`. Caso o diretório não exista, ele será criado automaticamente.

        Detalhes de Implementação:
        --------------------------
        - A função começa criando o caminho do arquivo com base no nome do conjunto de dados e no 
        algoritmo.
        - Se houver resultados para acurácia de teste (`rs_test_acc`), a função gera o nome do arquivo 
        e cria o diretório `../results/` caso ele não exista.
        - O arquivo HDF5 é criado usando `h5py.File`, e os resultados de acurácia de teste, AUC de teste 
        e perda de treinamento são salvos como datasets dentro do arquivo HDF5.

        Parâmetros:
        -----------
        Nenhum parâmetro adicional.

        Exemplo:
        --------
        >>> save_results()
        >>> # Os resultados de acurácia, AUC e perda de treinamento serão salvos no arquivo HDF5
        """
        algo = self.dataset + "_" + self.algorithm
        result_path = "../results/"
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        if (len(self.rs_test_acc)):
            algo = algo + "_" + self.goal + "_" + str(self.times)
            file_path = result_path + "{}.h5".format(algo)
            print("File path: " + file_path)

            with h5py.File(file_path, 'w') as hf:
                hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
                hf.create_dataset('rs_test_auc', data=self.rs_test_auc)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss)


    def save_item(self, item, item_name):
        """
        Salva um item em um arquivo no diretório de salvamento.

        Esta função salva um item, como um modelo ou tensor, em um arquivo no diretório de salvamento 
        especificado (`self.save_folder_name`). O nome do arquivo é gerado com o prefixo `server_` 
        seguido pelo nome do item (`item_name`) e a extensão `.pt`.

        Detalhes de Implementação:
        --------------------------
        - A função verifica se o diretório de salvamento (`self.save_folder_name`) existe. 
        Se não, o diretório é criado usando `os.makedirs()`.
        - O item é salvo utilizando `torch.save()`, e o arquivo é nomeado com o prefixo `server_` 
        seguido pelo nome do item e a extensão `.pt`.

        Parâmetros:
        -----------
        item : qualquer objeto
        O item a ser salvo. Pode ser um modelo, tensor ou qualquer objeto compatível com o 
        formato de salvamento do PyTorch.

        item_name : str
        O nome a ser atribuído ao arquivo salvo. Este nome será combinado com o prefixo `server_` 
        e a extensão `.pt` ao salvar o arquivo.

        Exemplo:
        --------
        >>> model = MyModel()
        >>> save_item(model, 'my_model')
        >>> # O modelo será salvo no diretório especificado com o nome 'server_my_model.pt'
        """
        if not os.path.exists(self.save_folder_name):
            os.makedirs(self.save_folder_name)
        torch.save(item, os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))


    def load_item(self, item_name):
        """
        Carrega um item a partir de um arquivo no diretório de salvamento.

        Esta função carrega um item previamente salvo em um arquivo no diretório de salvamento 
        especificado (`self.save_folder_name`). O arquivo é identificado pelo prefixo `server_` 
        seguido pelo nome do item (`item_name`) e a extensão `.pt`.

        Parâmetros:
        -----------
        item_name : str
        O nome do item a ser carregado. O nome será combinado com o prefixo `server_` 
        e a extensão `.pt` para localizar o arquivo correspondente.

        Retorna:
        --------
        item : qualquer objeto
        O item carregado do arquivo, que pode ser um modelo, tensor ou qualquer objeto 
        salvo com `torch.save()`.

        Detalhes de Implementação:
        --------------------------
        - O caminho completo do arquivo é gerado combinando o diretório de salvamento (`self.save_folder_name`), 
        o prefixo `server_`, o nome do item e a extensão `.pt`.
        - O item é carregado utilizando `torch.load()` e retornado.

        Exemplo:
        --------
        >>> model = load_item('my_model')
        >>> # O modelo será carregado do diretório de salvamento com o nome 'server_my_model.pt'
        """
        return torch.load(os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))


    def test_metrics(self):
        """
        Avalia o desempenho do modelo nos clientes, calculando métricas de acurácia e AUC.

        Esta função avalia o desempenho do modelo para todos os clientes, calculando a acurácia de teste 
        e a pontuação AUC (Área sob a Curva ROC) para cada cliente. Se houver novos clientes a serem 
        avaliados e a configuração `eval_new_clients` for ativada, o método realiza o ajuste fino desses 
        clientes e executa métricas específicas para eles. Caso contrário, as métricas gerais para todos 
        os clientes são calculadas.

        Retorna:
        --------
        ids : list
            Lista dos IDs dos clientes.

        num_samples : list
            Lista do número de amostras de teste para cada cliente.

        tot_correct : list
            Lista com o número total de acertos (corretos) de cada cliente.

        tot_auc : list
            Lista com o valor total da AUC ponderada (baseada no número de amostras) para cada cliente.

        Detalhes de Implementação:
        --------------------------
        - A função verifica se há novos clientes a serem avaliados (`eval_new_clients` e `num_new_clients`).
        - Se houver novos clientes, o ajuste fino é realizado com `fine_tuning_new_clients()` e 
            as métricas para novos clientes são calculadas através de `test_metrics_new_clients()`.
        - Se não houver novos clientes, as métricas de acurácia e AUC são calculadas para cada cliente 
            no conjunto de clientes (`self.clients`) utilizando `test_metrics()` de cada cliente.
        - As métricas de cada cliente são somadas e ponderadas pelo número de amostras de teste, 
            e os resultados finais são retornados.

        Exemplo:
        --------
        >>> ids, num_samples, tot_correct, tot_auc = test_metrics()
        >>> # As listas de IDs dos clientes, número de amostras, acertos e AUC serão retornadas
        """

        if self.eval_new_clients and self.num_new_clients > 0:
            self.fine_tuning_new_clients()
            return self.test_metrics_new_clients()
        
        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.clients:
            ct, ns, auc = c.test_metrics()
            tot_correct.append(ct*1.0)
            tot_auc.append(auc*ns)
            num_samples.append(ns)

        ids = [c.id for c in self.clients]

        return ids, num_samples, tot_correct, tot_auc


    def train_metrics(self):
        """
        Avalia o desempenho do modelo nos clientes, calculando métricas de acurácia e AUC.

        Esta função avalia o desempenho do modelo para todos os clientes, calculando a acurácia de teste 
        e a pontuação AUC (Área sob a Curva ROC) para cada cliente. Se houver novos clientes a serem 
        avaliados e a configuração `eval_new_clients` for ativada, o método realiza o ajuste fino desses 
        clientes e executa métricas específicas para eles. Caso contrário, as métricas gerais para todos 
        os clientes são calculadas.

        Retorna:
        --------
        ids : list
            Lista dos IDs dos clientes.

        num_samples : list
            Lista do número de amostras de teste para cada cliente.

        tot_correct : list
            Lista com o número total de acertos (corretos) de cada cliente.

        tot_auc : list
            Lista com o valor total da AUC ponderada (baseada no número de amostras) para cada cliente.

        Detalhes de Implementação:
        --------------------------
        - A função verifica se há novos clientes a serem avaliados (`eval_new_clients` e `num_new_clients`).
        - Se houver novos clientes, o ajuste fino é realizado com `fine_tuning_new_clients()` e 
        as métricas para novos clientes são calculadas através de `test_metrics_new_clients()`.
        - Se não houver novos clientes, as métricas de acurácia e AUC são calculadas para cada cliente 
        no conjunto de clientes (`self.clients`) utilizando `test_metrics()` de cada cliente.
        - As métricas de cada cliente são somadas e ponderadas pelo número de amostras de teste, 
        e os resultados finais são retornados.

        Exemplo:
        --------
        >>> ids, num_samples, tot_correct, tot_auc = test_metrics()
        >>> # As listas de IDs dos clientes, número de amostras, acertos e AUC serão retornadas
        """

        if self.eval_new_clients and self.num_new_clients > 0:
            return [0], [1], [0]
        
        num_samples = []
        losses = []
        for c in self.clients:
            cl, ns = c.train_metrics()
            num_samples.append(ns)
            losses.append(cl*1.0)

        ids = [c.id for c in self.clients]

        return ids, num_samples, losses


    def evaluate(self, acc=None, loss=None):
        """
        Avalia o desempenho do modelo calculando a acurácia e a AUC de teste, além da perda de treinamento.

        Esta função calcula as métricas de desempenho do modelo, incluindo a acurácia de teste, a AUC de teste, 
        e a perda de treinamento. As médias dessas métricas são calculadas para o conjunto de dados de teste e 
        de treinamento, e as métricas por cliente (como a acurácia e a AUC) também são computadas. Se os parâmetros 
        `acc` e `loss` forem fornecidos, os resultados serão adicionados a essas listas; caso contrário, eles 
        serão salvos nas listas internas `rs_test_acc` e `rs_train_loss`.

        Parâmetros:
        -----------
        acc : list, opcional
            Lista onde a acurácia de teste será armazenada. Se não fornecido, o valor é armazenado 
            em `self.rs_test_acc`.

        loss : list, opcional
            Lista onde a perda de treinamento será armazenada. Se não fornecido, o valor é armazenado 
            em `self.rs_train_loss`.

        Detalhes de Implementação:
        --------------------------
        - A função chama `test_metrics()` para calcular as métricas de teste (acurácia e AUC) e 
        `train_metrics()` para calcular as métricas de treinamento (perda).
        - As médias das métricas de teste e de treinamento são calculadas usando a soma ponderada dos valores 
        dividida pelo número total de amostras.
        - A função calcula a acurácia e a AUC para cada cliente e calcula o desvio padrão dessas métricas.
        - As métricas calculadas são armazenadas em `rs_test_acc` e `rs_train_loss` ou nas listas fornecidas 
        como parâmetros (`acc` e `loss`).

        Exemplo:
        --------
        >>> evaluate()
        >>> # As métricas de desempenho (perda, acurácia, AUC) serão calculadas e exibidas.
        """

        stats = self.test_metrics()
        stats_train = self.train_metrics()

        test_acc = sum(stats[2])*1.0 / sum(stats[1])
        test_auc = sum(stats[3])*1.0 / sum(stats[1])
        train_loss = sum(stats_train[2])*1.0 / sum(stats_train[1])
        accs = [a / n for a, n in zip(stats[2], stats[1])]
        aucs = [a / n for a, n in zip(stats[3], stats[1])]
        
        if acc == None:
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)
        
        if loss == None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)

        print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Test Accuracy: {:.4f}".format(test_acc))
        print("Averaged Test AUC: {:.4f}".format(test_auc))
        # self.print_(test_acc, train_acc, train_loss)
        print("Std Test Accuracy: {:.4f}".format(np.std(accs)))
        print("Std Test AUC: {:.4f}".format(np.std(aucs)))


    def print_(self, test_acc, test_auc, train_loss):
        """
        Exibe as métricas de desempenho médias do modelo, incluindo acurácia de teste, AUC de teste e perda de treinamento.

        Esta função imprime as médias de acurácia de teste, AUC de teste e perda de treinamento no console. 
        Os valores são formatados para exibição com quatro casas decimais.

        Parâmetros:
        -----------
        test_acc : float
            A média da acurácia de teste do modelo.

        test_auc : float
            A média da AUC de teste do modelo.

        train_loss : float
            A média da perda de treinamento do modelo.

        Detalhes de Implementação:
        --------------------------
        - A função recebe as métricas de desempenho (acurácia de teste, AUC de teste e perda de treinamento) 
        como parâmetros e os imprime formatados com quatro casas decimais.

        Exemplo:
        --------
        >>> print_(test_acc=0.85, test_auc=0.90, train_loss=0.25)
        >>> # Exibe:
        >>> # Average Test Accuracy: 0.8500
        >>> # Average Test AUC: 0.9000
        >>> # Average Train Loss: 0.2500
        """

        print("Average Test Accuracy: {:.4f}".format(test_acc))
        print("Average Test AUC: {:.4f}".format(test_auc))
        print("Average Train Loss: {:.4f}".format(train_loss))


    def check_done(self, acc_lss, top_cnt=None, div_value=None):
        """
        Verifica se o treinamento ou processo de avaliação deve ser considerado concluído com base nas métricas de acurácia.

        Esta função avalia se o treinamento ou o processo de avaliação pode ser considerado concluído 
        com base nas métricas de acurácia acumuladas ao longo de várias iterações. O critério de 
        conclusão pode ser determinado por dois parâmetros opcionais: `top_cnt` (número de iterações 
        superiores) e `div_value` (limiar de desvio padrão). A função avalia se a acurácia dos últimos 
        `top_cnt` ciclos supera um valor de referência e se o desvio padrão das últimas `top_cnt` iterações 
        é inferior a um valor de `div_value`. Se qualquer um dos critérios não for atendido, a função 
        retorna `False`, indicando que o processo ainda não foi concluído. Caso contrário, retorna `True`.

        Parâmetros:
        -----------
        acc_lss : list of lists
            Uma lista contendo as métricas de acurácia (ou outros valores) ao longo de várias iterações ou ciclos.
        
        top_cnt : int, opcional
            O número de iterações mais recentes a serem consideradas ao avaliar a acurácia. Se fornecido, 
            o processo de avaliação verifica se a melhor acurácia nas últimas `top_cnt` iterações é suficientemente alta.
        
        div_value : float, opcional
            O valor máximo de desvio padrão permitido entre as últimas `top_cnt` iterações. Se fornecido, 
            a função verifica se o desvio padrão da acurácia nas últimas `top_cnt` iterações é menor que `div_value`.

        Retorna:
        --------
        bool
            Retorna `True` se os critérios de `top_cnt` e/ou `div_value` forem atendidos para todas as listas de acurácia.
            Caso contrário, retorna `False`.

        Detalhes de Implementação:
        --------------------------
        - Se ambos `top_cnt` e `div_value` forem fornecidos, a função verifica se o número de iterações 
        superiores e o desvio padrão das últimas `top_cnt` iterações atendem aos critérios estabelecidos.
        - Se apenas `top_cnt` for fornecido, a função verifica se o número de iterações superiores às últimas 
        `top_cnt` iterações é adequado.
        - Se apenas `div_value` for fornecido, a função verifica se o desvio padrão das últimas `top_cnt` iterações 
        é menor que o valor fornecido.
        - Se nenhum dos critérios for atendido, a função retorna `False`.

        Exemplo:
        --------
        >>> check_done(acc_lss=[[0.85, 0.88, 0.9], [0.89, 0.91, 0.92]], top_cnt=3, div_value=0.02)
        >>> # Verifica se a acurácia das últimas iterações supera o número de iterações superiores e o desvio padrão.
        >>> # Retorna True ou False dependendo dos critérios.
        """

        for acc_ls in acc_lss:
            if top_cnt is not None and div_value is not None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_top and find_div:
                    pass
                else:
                    return False
            elif top_cnt is not None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                if find_top:
                    pass
                else:
                    return False
            elif div_value is not None:
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_div:
                    pass
                else:
                    return False
            else:
                raise NotImplementedError
        return True


    def call_dlg(self, R):
        """
        Calcula o valor PSNR (Peak Signal-to-Noise Ratio) para comparar os modelos dos clientes com o modelo global.

        Esta função avalia o desempenho dos modelos dos clientes comparando os gradientes originais 
        (do modelo global) e os gradientes calculados (do modelo do cliente). Para cada cliente, a função 
        calcula o PSNR, que é uma medida de qualidade de comparação entre o modelo global e o modelo do cliente. 
        O PSNR é calculado utilizando os gradientes originais e as saídas do modelo do cliente para um número 
        limitado de lotes de dados de treinamento. O valor final de PSNR é impresso após a avaliação de todos os clientes.

        Parâmetros:
        -----------
        R : int
            Número da rodada ou iteração atual. Este parâmetro pode ser usado para salvar ou organizar os resultados 
            de cada rodada, mas não é utilizado diretamente na lógica da função.

        Detalhes de Implementação:
        --------------------------
        - A função itera sobre os modelos dos clientes (`uploaded_models`) e seus IDs (`uploaded_ids`).
        - Para cada cliente, os gradientes entre o modelo global e o modelo do cliente são calculados.
        - Em seguida, o modelo do cliente é avaliado utilizando um conjunto de dados de treinamento do cliente, 
        e os gradientes calculados são comparados.
        - A função calcula o PSNR para cada cliente com base nas diferenças entre os gradientes originais e os gradientes do cliente.
        - O valor PSNR médio é calculado e impresso no final. Se não houver clientes para avaliar, é exibido um erro.

        Exemplo:
        --------
        >>> call_dlg(R=1)
        >>> # O valor PSNR médio será calculado e impresso para todos os clientes avaliados.
        >>> # Se não houver clientes para avaliar, será exibida a mensagem 'PSNR error'.
        """

        # items = []
        cnt = 0
        psnr_val = 0
        for cid, client_model in zip(self.uploaded_ids, self.uploaded_models):
            client_model.eval()
            origin_grad = []
            for gp, pp in zip(self.global_model.parameters(), client_model.parameters()):
                origin_grad.append(gp.data - pp.data)

            target_inputs = []
            trainloader = self.clients[cid].load_train_data()
            with torch.no_grad():
                for i, (x, y) in enumerate(trainloader):
                    if i >= self.batch_num_per_client:
                        break

                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    output = client_model(x)
                    target_inputs.append((x, output))

            d = DLG(client_model, origin_grad, target_inputs)
            if d is not None:
                psnr_val += d
                cnt += 1
            
            # items.append((client_model, origin_grad, target_inputs))
                
        if cnt > 0:
            print('PSNR value is {:.2f} dB'.format(psnr_val / cnt))
        else:
            print('PSNR error')

        # self.save_item(items, f'DLG_{R}')


    def set_new_clients(self, clientObj):
        """
        Inicializa novos clientes para o treinamento e os adiciona à lista de clientes.

        Esta função cria instâncias de novos clientes para o treinamento, utilizando o objeto `clientObj` 
        para instanciar cada cliente com base nos dados de treinamento e teste. Os novos clientes são adicionados 
        à lista `self.new_clients`. Para cada novo cliente, os dados de treinamento e teste são lidos utilizando 
        a função `read_client_data()`, e o cliente é inicializado com a quantidade de amostras de treinamento e teste.

        Parâmetros:
        -----------
        clientObj : class
            A classe do cliente a ser instanciado. Cada cliente é criado utilizando os parâmetros fornecidos 
            e a classe `clientObj` é responsável por inicializar as instâncias de cliente com base nos dados e 
            configurações fornecidas.

        Detalhes de Implementação:
        --------------------------
        - A função itera sobre o número de novos clientes (`num_new_clients`).
        - Para cada cliente, os dados de treinamento e teste são lidos usando a função `read_client_data()`, 
        com base no índice do cliente (`i`).
        - Para cada novo cliente, um objeto `clientObj` é instanciado com os dados de treinamento e teste 
        e adicionado à lista `self.new_clients`.

        Exemplo:
        --------
        >>> set_new_clients(Client)
        >>> # Cada novo cliente será inicializado com os dados de treinamento e teste e adicionado à lista de novos clientes
        """

        for i in range(self.num_clients, self.num_clients + self.num_new_clients):
            train_data = read_client_data(self.dataset, i, is_train=True, few_shot=self.few_shot)
            test_data = read_client_data(self.dataset, i, is_train=False, few_shot=self.few_shot)
            client = clientObj(self.args, 
                            id=i, 
                            train_samples=len(train_data), 
                            test_samples=len(test_data), 
                            train_slow=False, 
                            send_slow=False)
            self.new_clients.append(client)


    def fine_tuning_new_clients(self):
        """
        Realiza o ajuste fino (fine-tuning) dos novos clientes no modelo global.

        Esta função ajusta os parâmetros dos modelos dos novos clientes utilizando o modelo global 
        como ponto de partida. O ajuste fino é realizado utilizando os dados de treinamento de 
        cada cliente, com a otimização sendo feita por meio do algoritmo SGD (Stochastic Gradient Descent) 
        e a função de perda CrossEntropy. O número de épocas de ajuste fino (`fine_tuning_epoch_new`) 
        é determinado pelas configurações definidas.

        Detalhes de Implementação:
        --------------------------
        - A função itera sobre a lista `self.new_clients`, que contém os novos clientes.
        - Para cada cliente, o modelo global é carregado com os parâmetros do cliente através do método `set_parameters`.
        - O otimizador `SGD` é configurado para otimizar os parâmetros do modelo do cliente, com a taxa de aprendizado definida em `self.learning_rate`.
        - A função de perda utilizada para o ajuste fino é a `CrossEntropyLoss`, adequada para problemas de classificação.
        - O cliente é treinado por um número de épocas (`fine_tuning_epoch_new`), e durante cada época, a função calcula a perda, realiza a retropropagação e atualiza os parâmetros do modelo.
        - O loop de treinamento percorre os lotes de dados de treinamento do cliente (`trainloader`), movendo as entradas e os rótulos para o dispositivo apropriado (geralmente CPU ou GPU) antes de realizar a computação do modelo.

        Parâmetros:
        -----------
        Nenhum parâmetro adicional.

        Exemplo:
        --------
        >>> fine_tuning_new_clients()
        >>> # O modelo de cada novo cliente será ajustado usando o modelo global e os dados de treinamento específicos de cada cliente
        """

        for client in self.new_clients:
            client.set_parameters(self.global_model)
            opt = torch.optim.SGD(client.model.parameters(), lr=self.learning_rate)
            CEloss = torch.nn.CrossEntropyLoss()
            trainloader = client.load_train_data()
            client.model.train()
            for e in range(self.fine_tuning_epoch_new):
                for i, (x, y) in enumerate(trainloader):
                    if type(x) == type([]):
                        x[0] = x[0].to(client.device)
                    else:
                        x = x.to(client.device)
                    y = y.to(client.device)
                    output = client.model(x)
                    loss = CEloss(output, y)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()


    def test_metrics_new_clients(self):
        """
        Avalia as métricas de desempenho (acurácia e AUC) dos novos clientes.

        Esta função calcula as métricas de desempenho, como acurácia de teste e AUC de teste, 
        para os novos clientes. As métricas são ponderadas pelo número de amostras de cada cliente, 
        garantindo uma avaliação justa do desempenho do modelo nos novos clientes. As listas de IDs dos 
        clientes, número de amostras e as métricas totais de acertos e AUC são retornadas.

        Retorna:
        --------
        ids : list
            Lista contendo os IDs dos novos clientes.

        num_samples : list
            Lista contendo o número de amostras de teste para cada novo cliente.

        tot_correct : list
            Lista com o número total de acertos (acertos) de cada novo cliente.

        tot_auc : list
            Lista com o valor total da AUC ponderada (baseada no número de amostras) para cada novo cliente.

        Detalhes de Implementação:
        --------------------------
        - A função itera sobre a lista `self.new_clients`, que contém os novos clientes.
        - Para cada cliente, a função chama `test_metrics()` para calcular as métricas de desempenho (acurácia e AUC).
        - As métricas de acertos e AUC são somadas e ponderadas pelo número de amostras de teste de cada cliente.
        - As listas `ids`, `num_samples`, `tot_correct` e `tot_auc` são preenchidas com os resultados das métricas para cada cliente.

        Exemplo:
        --------
        >>> ids, num_samples, tot_correct, tot_auc = test_metrics_new_clients()
        >>> # As listas de IDs dos novos clientes, número de amostras, acertos e AUC serão retornadas
        """

        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.new_clients:
            ct, ns, auc = c.test_metrics()
            tot_correct.append(ct*1.0)
            tot_auc.append(auc*ns)
            num_samples.append(ns)

        ids = [c.id for c in self.new_clients]

        return ids, num_samples, tot_correct, tot_auc
