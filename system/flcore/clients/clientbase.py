import copy
import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from utils.data_utils import read_client_data


class Client(object):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        """
            Inicializa um objeto com os parâmetros necessários para o treinamento de um modelo de aprendizado de máquina.

            Este método de inicialização configura os principais componentes do treinamento, incluindo 
            o modelo, algoritmo, conjunto de dados, dispositivo de execução (CPU/GPU), parâmetros de 
            treinamento e otimização, além de verificar a presença de camadas de normalização de 
            lote (BatchNorm). Ele também define variáveis auxiliares para o controle de custos de tempo 
            durante o treinamento e envio de atualizações, e configura o otimizador e o agendador de 
            taxa de aprendizado.

            Parâmetros:
            -----------
            args : objeto
                Objeto contendo os parâmetros necessários para a configuração do modelo e treinamento, 
                como `model`, `algorithm`, `dataset`, `device`, `batch_size`, `num_classes`, 
                `local_learning_rate`, `local_epochs`, `learning_rate_decay_gamma`, etc.

            id : int
                Identificador único do cliente ou do agente.

            train_samples : int
                Número de amostras utilizadas para treinamento.

            test_samples : int
                Número de amostras utilizadas para teste.

            kwargs : dict, opcional
                Parâmetros adicionais, incluindo:
                - `train_slow` (bool): Indica se o treinamento deve ser feito de forma mais lenta.
                - `send_slow` (bool): Indica se o envio de atualizações deve ser feito de forma mais lenta.

            Atributos:
            ----------
            model : torch.nn.Module
                O modelo de aprendizado profundo copiado a partir de `args.model`.

            algorithm : str
                O algoritmo de treinamento especificado em `args.algorithm`.

            dataset : str
                O nome do conjunto de dados utilizado no treinamento, conforme definido em `args.dataset`.

            device : str
                O dispositivo de execução (CPU ou GPU), conforme configurado em `args.device`.

            id : int
                O identificador único para este cliente ou agente.

            save_folder_name : str
                O nome da pasta onde os resultados serão salvos, conforme configurado em `args.save_folder_name`.

            num_classes : int
                O número de classes no problema de classificação, conforme especificado em `args.num_classes`.

            train_samples : int
                O número de amostras de treinamento.

            test_samples : int
                O número de amostras de teste.

            batch_size : int
                O tamanho do lote a ser utilizado durante o treinamento, conforme configurado em `args.batch_size`.

            learning_rate : float
                A taxa de aprendizado local definida por `args.local_learning_rate`.

            local_epochs : int
                O número de épocas locais de treinamento, conforme especificado em `args.local_epochs`.

            few_shot : bool
                Indica se o treinamento é baseado em poucos exemplos (few-shot), conforme definido em `args.few_shot`.

            has_BatchNorm : bool
                Flag que indica se o modelo contém camadas de normalização de lote (BatchNorm).

            train_slow : bool
                Indica se o treinamento será feito de forma mais lenta (passado através de `kwargs`).

            send_slow : bool
                Indica se o envio das atualizações será feito de forma mais lenta (passado através de `kwargs`).

            train_time_cost : dict
                Dicionário para armazenar o número de rodadas de treinamento e o custo total de tempo de treinamento.

            send_time_cost : dict
                Dicionário para armazenar o número de rodadas de envio e o custo total de tempo de envio.

            loss : torch.nn.CrossEntropyLoss
                Função de perda utilizada para o treinamento (perda de entropia cruzada).

            optimizer : torch.optim.SGD
                Otimizador utilizado para otimizar os parâmetros do modelo (Stochastic Gradient Descent).

            learning_rate_scheduler : torch.optim.lr_scheduler.ExponentialLR
                Agendador de taxa de aprendizado que aplica uma redução exponencial na taxa de aprendizado.

            learning_rate_decay : bool
                Indica se a taxa de aprendizado deve ser decaída durante o treinamento, conforme configurado em `args.learning_rate_decay`.

            Exemplo:
            --------
            >>> args = Namespace(model=my_model, algorithm='SGD', dataset='CIFAR-10', device='cuda', batch_size=32, 
            >>>                  num_classes=10, local_learning_rate=0.01, local_epochs=10, learning_rate_decay_gamma=0.9)
            >>> model_trainer = MyModelTrainer(args, id=1, train_samples=500, test_samples=100)
            >>> print(model_trainer.model)
        """

        torch.manual_seed(0)
        self.model = copy.deepcopy(args.model)
        self.algorithm = args.algorithm
        self.dataset = args.dataset
        self.device = args.device
        self.id = id  # integer
        self.save_folder_name = args.save_folder_name

        self.num_classes = args.num_classes
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.local_epochs = args.local_epochs
        self.few_shot = args.few_shot

        # check BatchNorm
        self.has_BatchNorm = False
        for layer in self.model.children():
            if isinstance(layer, nn.BatchNorm2d):
                self.has_BatchNorm = True
                break

        self.train_slow = kwargs['train_slow']
        self.send_slow = kwargs['send_slow']
        self.train_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
        self.send_time_cost = {'num_rounds': 0, 'total_cost': 0.0}

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer, 
            gamma=args.learning_rate_decay_gamma
        )
        self.learning_rate_decay = args.learning_rate_decay


    def load_train_data(self, batch_size=None):
        """
        Carrega os dados de treinamento e retorna um DataLoader configurado.

        Esta função lê os dados de treinamento a partir de um conjunto de dados específico 
        (definido por `self.dataset` e `self.id`) e os organiza em lotes (batches) para 
        serem utilizados no treinamento do modelo. Caso o parâmetro `batch_size` não seja 
        fornecido, o valor de `self.batch_size` será utilizado como padrão.

        A função também configura o `DataLoader` para embaralhar os dados a cada nova iteração 
        e descartar o último lote caso ele tenha menos elementos do que o tamanho do lote 
        especificado.

        Parâmetros:
        -----------
        batch_size : int, opcional
            O tamanho do lote (batch) a ser utilizado ao carregar os dados. Se `None`, utiliza-se 
            o valor definido em `self.batch_size` (valor padrão).

        Retorna:
        --------
        DataLoader
            Um objeto `DataLoader` que contém os dados de treinamento organizados em lotes 
            do tamanho especificado, com o embaralhamento dos dados ativado e com o 
            parâmetro `drop_last=True`, o que garante que o último lote será descartado 
            se não tiver o tamanho exato.

        Detalhes de Implementação:
        --------------------------
        - A função usa `read_client_data()` para ler os dados de treinamento, com base no 
        conjunto de dados (`self.dataset`) e o ID (`self.id`) do cliente.
        - O parâmetro `few_shot` é passado para `read_client_data()` para indicar se o 
        treinamento deve ser realizado em um cenário de poucos exemplos (few-shot).
        - O `DataLoader` retornado pela função é configurado para embaralhar os dados e descartar 
        o último lote se ele não tiver o tamanho completo, o que é útil quando o número total 
        de amostras não é divisível pelo tamanho do lote.

        Exemplo:
        --------
        >>> train_loader = load_train_data(batch_size=64)
        >>> for batch in train_loader:
        >>>     # Iterar sobre os lotes de dados de treinamento
        >>>     inputs, targets = batch
        """

        if batch_size == None:
            batch_size = self.batch_size
        train_data = read_client_data(self.dataset, self.id, is_train=True, few_shot=self.few_shot)
        return DataLoader(train_data, batch_size, drop_last=True, shuffle=True)


    def load_test_data(self, batch_size=None):
        """
            Carrega os dados de teste e retorna um DataLoader configurado.

            Esta função lê os dados de teste a partir de um conjunto de dados específico 
            (definido por `self.dataset` e `self.id`) e os organiza em lotes (batches) 
            para serem utilizados durante a avaliação ou teste do modelo. Caso o parâmetro 
            `batch_size` não seja fornecido, o valor de `self.batch_size` será utilizado como padrão.

            A função configura o `DataLoader` para embaralhar os dados a cada nova iteração, 
            mas não descarta o último lote, permitindo que todos os dados sejam usados para 
            avaliação, mesmo que o último lote tenha menos elementos do que o tamanho do lote especificado.

            Parâmetros:
            -----------
            batch_size : int, opcional
                O tamanho do lote (batch) a ser utilizado ao carregar os dados de teste. Se `None`, 
                utiliza-se o valor definido em `self.batch_size` (valor padrão).

            Retorna:
            --------
            DataLoader
                Um objeto `DataLoader` que contém os dados de teste organizados em lotes 
                do tamanho especificado, com o embaralhamento dos dados ativado e sem 
                descartar o último lote.

            Detalhes de Implementação:
            --------------------------
            - A função usa `read_client_data()` para ler os dados de teste, com base no 
            conjunto de dados (`self.dataset`) e o ID (`self.id`) do cliente.
            - O parâmetro `few_shot` é passado para `read_client_data()` para indicar se o 
            teste deve ser realizado em um cenário de poucos exemplos (few-shot).
            - O `DataLoader` retornado pela função é configurado para embaralhar os dados, mas não 
            descartar o último lote, permitindo que todos os dados de teste sejam utilizados, 
            independentemente do número total de amostras.

            Exemplo:
            --------
            >>> test_loader = load_test_data(batch_size=64)
            >>> for batch in test_loader:
            >>>     # Iterar sobre os lotes de dados de teste
            >>>     inputs, targets = batch
        """
        if batch_size == None:
            batch_size = self.batch_size
        test_data = read_client_data(self.dataset, self.id, is_train=False, few_shot=self.few_shot)
        return DataLoader(test_data, batch_size, drop_last=False, shuffle=True)

        
    def set_parameters(self, model):
        """
            Define os parâmetros do modelo atual com os parâmetros de um novo modelo.

            Esta função atualiza os parâmetros do modelo atual com os valores dos parâmetros de 
            outro modelo fornecido. A função itera sobre os parâmetros de ambos os modelos, 
            clonando os dados dos parâmetros do novo modelo e atribuindo-os aos parâmetros do modelo atual.

            Parâmetros:
            -----------
            model : torch.nn.Module
                O novo modelo cujos parâmetros serão copiados para o modelo atual.

            Detalhes de Implementação:
            --------------------------
            - A função usa a função `zip()` para iterar simultaneamente sobre os parâmetros do 
            modelo atual (`self.model.parameters()`) e os parâmetros do modelo fornecido (`model.parameters()`).
            - Para cada par de parâmetros (um do modelo atual e um do novo modelo), os dados do 
            parâmetro do novo modelo são clonados usando `clone()` e atribuídos ao parâmetro correspondente 
            do modelo atual. Isso garante que os parâmetros sejam copiados de forma independente, 
            sem referência a objetos compartilhados.
            
            Exemplo:
            --------
            >>> model_1 = Model()
            >>> model_2 = Model()
            >>> set_parameters(model_2)  # Os parâmetros de model_2 agora são copiados para model_1
        """
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()


    def clone_model(self, model, target):
        """
            Clona os parâmetros de um modelo para outro modelo de destino.

            Esta função copia os parâmetros do modelo de origem (`model`) para o modelo de destino (`target`). 
            Os dados de cada parâmetro no modelo de origem são clonados e atribuídos aos parâmetros correspondentes 
            no modelo de destino. A clonagem dos parâmetros é realizada sem compartilhar referências entre os modelos.

            Parâmetros:
            -----------
            model : torch.nn.Module
                O modelo de origem, cujos parâmetros serão copiados.

            target : torch.nn.Module
                O modelo de destino, cujos parâmetros serão atualizados com os valores clonados do modelo de origem.

            Detalhes de Implementação:
            --------------------------
            - A função usa `zip()` para iterar sobre os parâmetros de ambos os modelos (origem e destino).
            - Para cada par de parâmetros correspondentes, os dados do parâmetro do modelo de origem são clonados 
            usando `clone()` e atribuídos ao parâmetro correspondente no modelo de destino. Isso garante que os 
            parâmetros sejam copiados de forma independente, sem referências compartilhadas.
            - A linha que copia os gradientes (comentada) pode ser descomentada se for necessário copiar também 
            os gradientes dos parâmetros.

            Exemplo:
            --------
            >>> model_1 = Model()
            >>> model_2 = Model()
            >>> clone_model(model_1, model_2)  # Os parâmetros de model_1 são clonados para model_2
        """
        for param, target_param in zip(model.parameters(), target.parameters()):
            target_param.data = param.data.clone()
            # target_param.grad = param.grad.clone()


    def update_parameters(self, model, new_params):
        """
            Atualiza os parâmetros de um modelo com novos valores fornecidos.

            Esta função atualiza os parâmetros de um modelo (`model`) com os valores dos novos parâmetros 
            fornecidos em `new_params`. Para cada parâmetro no modelo, os dados são clonados do parâmetro 
            correspondente em `new_params` e atribuídos ao parâmetro no modelo original.

            Parâmetros:
            -----------
            model : torch.nn.Module
                O modelo cujos parâmetros serão atualizados.

            new_params : iterável
                Um iterável (geralmente uma lista ou uma coleção) contendo os novos parâmetros 
                que serão copiados para o modelo. Cada elemento de `new_params` deve corresponder a um parâmetro 
                do modelo em `model`.

            Detalhes de Implementação:
            --------------------------
            - A função usa `zip()` para iterar simultaneamente sobre os parâmetros do modelo e os novos parâmetros.
            - Para cada par de parâmetros (um do modelo original e um de `new_params`), os dados do novo parâmetro 
            são clonados usando `clone()` e atribuídos ao parâmetro correspondente do modelo. Isso garante que os 
            parâmetros sejam copiados de forma independente e sem referências compartilhadas.

            Exemplo:
            --------
            >>> model = Model()
            >>> new_params = [new_param_1, new_param_2, ...]  # Parâmetros atualizados
            >>> update_parameters(model, new_params)  # Os parâmetros do modelo são atualizados com new_params
        """
        for param, new_param in zip(model.parameters(), new_params):
            param.data = new_param.data.clone()


    def test_metrics(self):
        """
            Avalia o desempenho do modelo no conjunto de dados de teste.

            Esta função realiza a avaliação do modelo no conjunto de dados de teste, calculando a 
            acurácia do teste e a pontuação AUC (Área sob a Curva ROC). Ela percorre o conjunto de 
            dados de teste, faz previsões usando o modelo, e calcula a acurácia juntamente com a 
            probabilidade predita para cada classe, que é utilizada para calcular a AUC.

            A função não realiza backpropagation e desativa o cálculo dos gradientes para otimizar o desempenho.

            Retorna:
            --------
            test_acc : float
                A acurácia total do modelo no conjunto de dados de teste.

            test_num : int
                O número total de amostras no conjunto de dados de teste.

            auc : float
                A pontuação AUC (Área sob a Curva ROC) do modelo no conjunto de dados de teste.

            Detalhes de Implementação:
            --------------------------
            - A função utiliza `load_test_data()` para carregar os dados de teste.
            - O modelo é colocado em modo de avaliação (`self.model.eval()`), desativando a 
            atualização dos parâmetros e o cálculo dos gradientes.
            - Durante a avaliação, as previsões do modelo são comparadas com os rótulos reais para 
            calcular a acurácia.
            - A função também coleta as probabilidades previstas pelo modelo e os rótulos binarizados 
            para calcular a pontuação AUC usando a função `roc_auc_score` da biblioteca `metrics`.
            - O cálculo da AUC é realizado utilizando a média micro, adequada para problemas multiclasse.

            Exemplo:
            --------
            >>> test_acc, test_num, auc = test_metrics()
            >>> print(f"Acurácia: {test_acc / test_num:.4f}, AUC: {auc:.4f}")
        """
        testloaderfull = self.load_test_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []
        
        with torch.no_grad():
            for x, y in testloaderfull:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

                y_prob.append(output.detach().cpu().numpy())
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        auc = metrics.roc_auc_score(y_true, y_prob, average='micro')
        
        return test_acc, test_num, auc


    def train_metrics(self):
        """
            Avalia o desempenho do modelo no conjunto de dados de treinamento.

            Esta função calcula a perda total (loss) durante o treinamento, percorrendo o conjunto de 
            dados de treinamento e realizando previsões usando o modelo. A perda é acumulada para 
            calcular o desempenho do modelo durante o processo de treinamento.

            A função desativa o cálculo de gradientes durante a avaliação para otimizar o desempenho.

            Retorna:
            --------
            losses : float
                A perda total acumulada durante o treinamento no conjunto de dados de treinamento.

            train_num : int
                O número total de amostras no conjunto de dados de treinamento.

            Detalhes de Implementação:
            --------------------------
            - A função utiliza `load_train_data()` para carregar os dados de treinamento.
            - O modelo é colocado em modo de avaliação (`self.model.eval()`), desativando a 
            atualização dos parâmetros e o cálculo dos gradientes.
            - Para cada lote de dados, a função calcula a perda usando a função de perda (`self.loss`), 
            somando os valores da perda multiplicados pelo número de amostras no lote.
            - A função retorna a perda total acumulada e o número total de amostras processadas.

            Exemplo:
            --------
            >>> losses, train_num = train_metrics()
            >>> print(f"Perda total: {losses:.4f}, Número de amostras: {train_num}")
        """
        trainloader = self.load_train_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        return losses, train_num


    def get_next_train_batch(self):
        """
            Obtém o próximo lote de dados de treinamento para personalização.

            Esta função retorna o próximo lote de dados de treinamento, utilizando um iterador sobre 
            o carregador de dados (`trainloader`). Se o iterador do carregador de dados for esgotado, 
            o gerador é reiniciado e começa novamente do início. O lote de dados é movido para o dispositivo 
            configurado (CPU ou GPU) antes de ser retornado.

            Retorna:
            --------
            x : torch.Tensor
                O tensor de entradas do próximo lote de dados de treinamento, movido para o dispositivo 
                configurado (CPU ou GPU).

            y : torch.Tensor
                O tensor de rótulos do próximo lote de dados de treinamento, movido para o dispositivo 
                configurado (CPU ou GPU).

            Detalhes de Implementação:
            --------------------------
            - A função usa um iterador (`iter_trainloader`) para obter o próximo lote de dados do 
            `trainloader`. Se o iterador atingir o final, ele é reiniciado para garantir que o 
            treinamento continue a partir do primeiro lote.
            - O lote de entradas (`x`) é verificado para determinar se é uma lista, e caso seja, o 
            primeiro elemento é selecionado.
            - Tanto as entradas quanto os rótulos são movidos para o dispositivo (`self.device`), 
            seja CPU ou GPU, antes de serem retornados.
            
            Exemplo:
            --------
            >>> x, y = get_next_train_batch()
            >>> print(x.shape, y.shape)  # Imprime o formato das entradas e rótulos do próximo lote
        """
        try:
            # Samples a new batch for persionalizing
            (x, y) = next(self.iter_trainloader)
        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            self.iter_trainloader = iter(self.trainloader)
            (x, y) = next(self.iter_trainloader)

        if type(x) == type([]):
            x = x[0]
        x = x.to(self.device)
        y = y.to(self.device)

        return x, y


    def save_item(self, item, item_name, item_path=None):
        """
            Salva um item (como um modelo ou tensor) em um diretório especificado.

            Esta função salva um item no sistema de arquivos, utilizando o nome do item fornecido 
            e o caminho de destino. Se o caminho não for especificado, o item será salvo na pasta 
            de salvamento padrão. Caso o diretório de destino não exista, ele será criado automaticamente.

            Parâmetros:
            -----------
            item : qualquer objeto
                O item a ser salvo. Pode ser um modelo, tensor ou qualquer objeto compatível com o 
                formato de salvamento do PyTorch.

            item_name : str
                O nome a ser atribuído ao arquivo salvo. Este nome será combinado com o ID do cliente 
                e a extensão `.pt` ao salvar o arquivo.

            item_path : str, opcional
                O caminho do diretório onde o item será salvo. Se `None`, o item será salvo no diretório 
                padrão (`self.save_folder_name`). (default: `None`)

            Detalhes de Implementação:
            --------------------------
            - Se o `item_path` não for especificado, ele será definido como o valor de `self.save_folder_name`.
            - Caso o diretório de destino (`item_path`) não exista, ele será criado usando `os.makedirs()`.
            - O item é salvo no caminho especificado usando `torch.save()`, com o nome do arquivo 
            gerado combinando o ID do cliente (`self.id`) e o nome do item fornecido, seguido pela 
            extensão `.pt`.

            Exemplo:
            --------
            >>> model = MyModel()
            >>> save_item(model, 'my_model')
            >>> # O modelo será salvo no diretório padrão com o nome 'client_1_my_model.pt'
        """
        if item_path == None:
            item_path = self.save_folder_name
        if not os.path.exists(item_path):
            os.makedirs(item_path)
        torch.save(item, os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))


    def load_item(self, item_name, item_path=None):
        """
            Carrega um item (como um modelo ou tensor) a partir de um diretório especificado.

            Esta função carrega um item previamente salvo no sistema de arquivos, utilizando o nome 
            do item fornecido e o caminho de origem. Se o caminho não for especificado, o item será 
            carregado da pasta de salvamento padrão. O nome do arquivo é gerado combinando o ID do cliente 
            e o nome do item, com a extensão `.pt`.

            Parâmetros:
            -----------
            item_name : str
                O nome do item a ser carregado. O nome será combinado com o ID do cliente e a 
                extensão `.pt` para localizar o arquivo correspondente.

            item_path : str, opcional
                O caminho do diretório onde o item está salvo. Se `None`, o item será carregado 
                do diretório padrão (`self.save_folder_name`). (default: `None`)

            Retorna:
            --------
            item : qualquer objeto
                O item carregado do arquivo, que pode ser um modelo, tensor ou qualquer objeto 
                salvo com `torch.save()`.

            Detalhes de Implementação:
            --------------------------
            - Se o `item_path` não for especificado, ele será definido como o valor de `self.save_folder_name`.
            - O caminho completo do arquivo é gerado combinando o diretório de destino, o ID do cliente 
            e o nome do item fornecido, seguido pela extensão `.pt`.
            - O item é carregado usando `torch.load()` a partir do caminho gerado.

            Exemplo:
            --------
            >>> model = load_item('my_model')
            >>> # O modelo será carregado do diretório padrão com o nome 'client_1_my_model.pt'
        """
        if item_path == None:
            item_path = self.save_folder_name
        return torch.load(os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))


    @staticmethod
    def model_exists():
        """
            Verifica se o modelo do servidor existe no diretório de modelos.

            Esta função verifica se o arquivo do modelo do servidor existe no diretório `models` 
            e retorna um valor booleano indicando a existência do arquivo.

            Retorna:
            --------
            bool
                `True` se o modelo do servidor existir no diretório `models`, caso contrário, 
                retorna `False`.

            Detalhes de Implementação:
            --------------------------
            - A função verifica a existência do arquivo chamado `server.pt` no diretório `models`.
            - Utiliza `os.path.exists()` para verificar a presença do arquivo.

            Exemplo:
            --------
            >>> if model_exists():
            >>>     print("O modelo do servidor existe.")
            >>> else:
            >>>     print("O modelo do servidor não foi encontrado.")
        """
        return os.path.exists(os.path.join("models", "server" + ".pt"))
