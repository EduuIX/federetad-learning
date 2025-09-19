import copy
import torch
import numpy as np
import time
from flcore.clients.clientbase import Client


class clientAlert(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        """
        Inicializa a instância do cliente com os parâmetros fornecidos e chama o construtor da classe base para inicializar 
        os atributos fundamentais.

        Este método é o construtor da classe `Client`, responsável por inicializar o cliente com o ID, amostras de 
        treinamento e teste, e os argumentos adicionais passados. Ele chama o construtor da classe base para garantir 
        que todos os atributos herdados sejam configurados corretamente.

        Fluxo de Execução:
        ------------------
        1. O método `__init__` é chamado com os parâmetros `args`, `id`, `train_samples`, `test_samples` e outros 
        parâmetros opcionais através de `**kwargs`.
        2. O construtor da classe base (`super().__init__`) é chamado com os mesmos parâmetros para inicializar os 
        atributos da classe `Client`.
        3. O cliente é configurado com os dados de treinamento (`train_samples`) e de teste (`test_samples`).
        4. Qualquer outro argumento adicional é passado para a classe base via `**kwargs`, permitindo a flexibilidade de 
        configuração adicional.

        Parâmetros:
        -----------
        args : tipo de dado
            Argumentos de configuração do cliente, como parâmetros de treinamento ou outras configurações gerais.
        
        id : tipo de dado
            Identificador único do cliente.

        train_samples : tipo de dado
            Conjunto de dados de treinamento utilizado para treinar o modelo local.

        test_samples : tipo de dado
            Conjunto de dados de teste utilizado para avaliar o desempenho do modelo local.

        kwargs : tipos de dados adicionais
            Argumentos opcionais que são passados para a classe base, permitindo personalizações adicionais.

        Retorno:
        --------
        Nenhum. O método `__init__` apenas inicializa a instância do cliente e configura os dados de treinamento e teste.

        Detalhes:
        ---------
        - O método chama o construtor da classe base `Client` para garantir que todos os atributos e comportamentos 
        herdados sejam corretamente configurados.
        - O cliente é configurado com os dados fornecidos para treinamento e teste.
        - A flexibilidade é proporcionada pelo uso de `**kwargs`, permitindo que a classe base seja personalizada com 
        parâmetros adicionais.

        Exemplos:
        ---------
        # Inicializando o cliente com os dados de treinamento e teste
        client = Client(args=config, id=1, train_samples=train_data, test_samples=test_data)

        # O cliente será configurado e preparado para o treinamento e avaliação.
        """

        super().__init__(args, id, train_samples, test_samples, **kwargs)

    def train(self):
        """
        Realiza o treinamento local do modelo utilizando os dados de treinamento fornecidos.

        Este método carrega os dados de treinamento, define o modelo para o modo de treinamento, e executa 
        múltiplas iterações de treinamento, ajustando os pesos do modelo através da otimização. Além disso, 
        inclui um mecanismo opcional para retardar o treinamento, simula uma execução mais lenta, e aplica 
        a decaimento da taxa de aprendizado, caso necessário.

        Fluxo de Execução:
        ------------------
        1. Carrega os dados de treinamento através da função `load_train_data`.
        2. Define o modelo para o modo de treinamento (ativo com `model.train()`).
        3. Inicia um temporizador para medir o tempo de treinamento.
        4. Define o número máximo de épocas locais (`max_local_epochs`), sendo ajustado caso o treinamento lento 
        (`train_slow`) esteja ativado.
        5. Para cada época de treinamento:
            - Carrega os lotes de dados (entradas `x` e rótulos `y`) do `trainloader`.
            - Move os dados para o dispositivo de computação (CPU ou GPU).
            - Se o treinamento lento estiver ativado, aplica um atraso aleatório entre as iterações.
            - Realiza a passagem direta (forward pass) através do modelo.
            - Calcula a perda (`loss`) entre a saída do modelo e o valor real.
            - Realiza o retropropagação (`backward`) para calcular os gradientes.
            - Aplica a otimização para ajustar os pesos do modelo.
        6. Caso o decaimento da taxa de aprendizado esteja ativado, ajusta a taxa de aprendizado utilizando o 
        `learning_rate_scheduler`.
        7. Atualiza as métricas de tempo de treinamento, registrando o número de rodadas e o tempo total gasto.

        Parâmetros:
        -----------
        Nenhum. A função utiliza variáveis de instância, como `local_epochs`, `train_slow`, `device`, `optimizer`, 
        `loss`, e `learning_rate_scheduler`.

        Retorno:
        --------
        Nenhum. O método realiza o treinamento local do modelo e atualiza as métricas de tempo internamente.

        Detalhes:
        ---------
        - Os dados de treinamento são carregados a partir da função `load_train_data`.
        - O modelo é colocado em modo de treinamento com `self.model.train()`.
        - O treinamento pode ser acelerado ou retardado com o parâmetro `train_slow`, que ajusta o número de 
        épocas locais e aplica um atraso aleatório entre as iterações.
        - Para cada lote de dados, o modelo realiza uma passagem direta, calcula a perda, faz a retropropagação 
        e ajusta os parâmetros com o otimizador.
        - Caso a taxa de aprendizado precise ser ajustada, a função `learning_rate_scheduler.step()` é chamada.
        - O tempo total de treinamento é calculado e armazenado em `self.train_time_cost`.

        Exemplos:
        ---------
        # Treinando o modelo com dados locais
        client.train()

        # Saída esperada:
        # - O modelo treinado será atualizado internamente
        # - O tempo total de treinamento será registrado
        """

        trainloader = self.load_train_data()
        # self.model.to(self.device)
        self.model.train()
        
        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                output = self.model(x)
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        # self.model.cpu()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time
