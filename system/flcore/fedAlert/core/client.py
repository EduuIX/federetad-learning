from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Tuple, Any

class TinyMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x):
        return self.net(x)

def get_loss_fn():
    return nn.CrossEntropyLoss(reduction="mean")

class FedAlertClient:
    """
    Minimal client that trains locally and computes A_k = |L_before - L_after|,
    then emits a 1-bit alert via local threshold tau_k = mu + z * sigma learned in warm-up.
    """
    def __init__(self, cid: str, x: np.ndarray, y: np.ndarray, cfg: Dict[str, Any]):
        """
        Inicializa o cliente para o aprendizado federado com uma rede neural simples (TinyMLP) e configura os parâmetros 
        necessários para treinamento local e detecção de anomalias com base na variação da perda.

        O cliente treina localmente com seus dados, calcula a diferença de perda (A_k) antes e depois do treinamento 
        e emite um alerta se a diferença for maior que um limiar local, com base em uma estratégia de thresholding 
        (definida pela média e desvio padrão da variação da perda).

        Parâmetros:
        ----------
        cid : str
            Identificador único do cliente no contexto do aprendizado federado.
        x : np.ndarray
            Array NumPy contendo os dados de entrada para o treinamento do modelo. O array será convertido para 
            um tensor PyTorch.
        y : np.ndarray
            Array NumPy contendo os rótulos (labels) correspondentes aos dados de entrada. O array será convertido 
            para um tensor PyTorch.
        cfg : Dict[str, Any]
            Dicionário contendo as configurações do cliente, incluindo parâmetros de treinamento e thresholds:
            - `hidden`: Número de neurônios na camada oculta (default: 32).
            - `trigger["warmup_B"]`: Número de rodadas de aquecimento (warm-up) para calcular o limiar do alerta.
            - `client_z`: Fator multiplicativo usado para o cálculo do limiar do alerta (default: 2.0).
            - `local_epochs`, `lr`, `batch`: Parâmetros para o treinamento local (não obrigatórios, valores padrões são usados se ausentes).
            - `ldp_q`: Probabilidade de aplicar resposta aleatória para garantir privacidade diferencial local (default: 0.0).

        Atributos:
        ----------
        cid : str
            Identificador único do cliente.
        x : torch.Tensor
            Dados de entrada convertidos para um tensor PyTorch.
        y : torch.Tensor
            Rótulos correspondentes aos dados de entrada convertidos para um tensor PyTorch.
        cfg : Dict[str, Any]
            Configurações passadas para o cliente.
        model : TinyMLP
            Instância da rede neural `TinyMLP`, com a camada de entrada definida com base no número de características 
            de `x` e a camada oculta definida conforme a configuração.
        loss_fn : torch.nn.CrossEntropyLoss
            Função de perda utilizada para o treinamento.
        device : torch.device
            Dispositivo em que o modelo será executado (no caso, CPU).
        B : int
            Número de rodadas de aquecimento (warm-up) para cálculo da média e desvio padrão da variação da perda (A_k).
        z : float
            Fator utilizado no cálculo do limiar de alerta (tau) como `mu + z * sigma`.
        ak_hist : list
            Lista para armazenar os valores de A_k durante o aquecimento.
        mu : float or None
            Média da variação da perda (A_k) após o aquecimento.
        sigma : float or None
            Desvio padrão da variação da perda (A_k) após o aquecimento.
        round : int
            Contador da rodada atual de treinamento.
        
        Exemplos:
        ---------
        # Inicializando um cliente para aprendizado federado
        cfg = {
            "trigger": {"warmup_B": 10},
            "client_z": 2.0,
            "hidden": 64,
            "local_epochs": 5,
            "lr": 0.01,
            "batch": 32,
            "ldp_q": 0.1
        }
        client = FedAlertClient(cid="client_1", x=train_data, y=train_labels, cfg=cfg)
        """

        self.cid = cid
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)
        self.cfg = cfg
        self.model = TinyMLP(self.x.shape[1], hidden=cfg.get("hidden", 32))
        self.loss_fn = get_loss_fn()
        self.device = torch.device("cpu")
        self.model.to(self.device)

        # Local thresholding state
        self.B = int(cfg["trigger"]["warmup_B"])
        self.z = float(cfg.get("client_z", 2.0))
        self.ak_hist = []  # store A_k during warm-up
        self.mu = None
        self.sigma = None
        self.round = 0

    def get_weights(self):
        """
        Retorna os pesos atuais do modelo como um dicionário de arrays NumPy.

        Este método coleta os pesos do modelo atual (`self.model`), os desvincula da árvore de computação 
        do PyTorch, transfere-os para a CPU (caso estejam em outro dispositivo como GPU) e converte-os 
        para arrays NumPy. O resultado é um dicionário onde as chaves são os nomes dos parâmetros 
        do modelo e os valores são os pesos correspondentes, no formato NumPy.

        Retorno:
        --------
        Dict[str, np.ndarray]
            Um dicionário onde as chaves são os nomes dos parâmetros do modelo (por exemplo, "weight", "bias")
            e os valores são os pesos correspondentes, armazenados como arrays NumPy.

        Exemplos:
        ---------
        # Obtendo os pesos do modelo
        client = FedAlertClient(cid="client_1", x=train_data, y=train_labels, cfg=cfg)
        weights = client.get_weights()
        print(weights)
        """

        return {k: v.detach().cpu().numpy() for k, v in self.model.state_dict().items()}

    def set_weights(self, weights: Dict[str, np.ndarray]):
        """
        Define os pesos do modelo a partir de um dicionário de arrays NumPy.

        Este método recebe um dicionário de pesos (onde as chaves são os nomes dos parâmetros do modelo 
        e os valores são arrays NumPy contendo os pesos correspondentes) e atualiza o modelo atual 
        (`self.model`) com esses pesos. Os valores dos pesos são convertidos de arrays NumPy para tensores 
        PyTorch antes de serem aplicados ao modelo. A atualização dos pesos é feita de forma estrita, 
        ou seja, os pesos fornecidos devem corresponder exatamente à estrutura do modelo.

        Parâmetros:
        -----------
        weights : Dict[str, np.ndarray]
            Um dicionário contendo os pesos a serem aplicados ao modelo. As chaves são os nomes dos 
            parâmetros do modelo (por exemplo, "weight", "bias") e os valores são os pesos correspondentes 
            em formato de arrays NumPy.

        Retorno:
        --------
        None
            Este método não retorna nenhum valor. Ele apenas modifica os pesos do modelo.

        Exemplos:
        ---------
        # Definindo os pesos do modelo a partir de um dicionário de pesos
        client = FedAlertClient(cid="client_1", x=train_data, y=train_labels, cfg=cfg)
        new_weights = {"weight": np.array([...]), "bias": np.array([...])}
        client.set_weights(new_weights)
        """

        state = {k: torch.from_numpy(v) for k, v in weights.items()}
        self.model.load_state_dict(state, strict=True)

    def _local_epoch(self, lr=0.05, epochs=1, batch=64):
        """
        Realiza uma rodada de treinamento local no modelo usando o algoritmo de Gradiente Descendente Estocástico (SGD).

        Este método realiza o treinamento local do modelo em um número específico de épocas e com um tamanho de 
        lote (batch) definido. Durante o treinamento, os dados de entrada são embaralhados, os gradientes são 
        calculados e os pesos do modelo são atualizados. O treinamento é feito utilizando a função de perda 
        (`self.loss_fn`) e o otimizador SGD.

        Parâmetros:
        -----------
        lr : float, opcional (default=0.05)
            A taxa de aprendizado usada pelo otimizador SGD. Controla o tamanho do passo durante a atualização dos pesos.

        epochs : int, opcional (default=1)
            O número de épocas (iterações completas sobre os dados de treinamento) para o qual o modelo será treinado.

        batch : int, opcional (default=64)
            O tamanho do lote (batch size) usado durante o treinamento. Define quantos exemplos são usados por vez 
            para calcular o gradiente e atualizar os pesos.

        Retorno:
        --------
        float
            O valor da última perda (loss) calculada durante o treinamento. Retorna 0.0 caso a perda não tenha sido calculada.

        Exemplos:
        ---------
        # Treinamento local do modelo com taxa de aprendizado de 0.01, 5 épocas e tamanho de lote 32
        client = FedAlertClient(cid="client_1", x=train_data, y=train_labels, cfg=cfg)
        loss = client._local_epoch(lr=0.01, epochs=5, batch=32)
        print("Última perda: ", loss)
        """

        self.model.train()
        opt = optim.SGD(self.model.parameters(), lr=lr, momentum=0.0)
        n = self.x.size(0)
        idx = torch.randperm(n)
        xb = self.x[idx]
        yb = self.y[idx]
        loss_val = None
        for _ in range(epochs):
            for i in range(0, n, batch):
                x_i = xb[i:i+batch].to(self.device)
                y_i = yb[i:i+batch].to(self.device)
                logits = self.model(x_i.float())
                loss = self.loss_fn(logits, y_i.long())
                opt.zero_grad()
                loss.backward()
                opt.step()
                loss_val = loss.detach().item()
        return float(loss_val if loss_val is not None else 0.0)

    def _evaluate_loss(self):
        """
        Avalia a perda do modelo no conjunto de dados atual.

        Este método coloca o modelo em modo de avaliação (`eval()`), realiza uma passagem para frente no conjunto de dados 
        de entrada (`self.x`), calcula a perda com base na função de perda definida (`self.loss_fn`) e retorna o valor da 
        perda. A avaliação é feita sem o cálculo de gradientes, o que é feito com o contexto `torch.no_grad()` para economizar 
        memória e acelerar o processo.

        Retorno:
        --------
        float
            O valor da perda calculada, convertido para um número flutuante. A perda é calculada utilizando a função 
            de perda definida (por exemplo, `CrossEntropyLoss`).

        Exemplos:
        ---------
        # Avaliando a perda do modelo no conjunto de dados atual
        client = FedAlertClient(cid="client_1", x=train_data, y=train_labels, cfg=cfg)
        loss = client._evaluate_loss()
        print("Perda do modelo: ", loss)
        """

        self.model.eval()
        with torch.no_grad():
            logits = self.model(self.x.float().to(self.device))
            loss = self.loss_fn(logits, self.y.long().to(self.device)).item()
        return float(loss)

    def fit_one_round(self, global_weights: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
        """
        Realiza o treinamento de uma rodada local e calcula os alertas baseados na variação de perda.

        Este método aplica os pesos globais fornecidos no modelo local e executa uma rodada de treinamento. 
        Após o treinamento, calcula a variação da perda (A_k) antes e depois do treinamento, emite um alerta 
        se a variação exceder um limiar calculado localmente e retorna os pesos atualizados junto com métricas 
        da rodada de treinamento.

        O limiar de alerta é baseado em uma média (`mu`) e desvio padrão (`sigma`) das variações de perda 
        acumuladas durante as rodadas anteriores de treinamento. Além disso, uma resposta aleatória pode ser 
        aplicada para garantir privacidade diferencial local (LDP).

        Fluxo de Execução:
        ------------------
        1. O método recebe os pesos globais atuais (global_weights) e os aplica ao modelo local.
        2. Inicia o treinamento local por uma única rodada, com base nos pesos globais.
        3. Após o treinamento, calcula a perda antes (`L_before`) e após (`L_after`) o treinamento.
        4. Calcula a variação da perda (`A_k`) como a diferença entre `L_before` e `L_after`.
        5. Compara a variação da perda (`A_k`) com um limiar calculado localmente, usando a média (`mu`) 
        e o desvio padrão (`sigma`) das variações de perda anteriores.
        6. Emite um alerta se a variação exceder o limiar, e pode aplicar uma resposta aleatória para garantir LDP.
        7. Retorna os pesos atualizados do modelo local juntamente com as métricas calculadas para a rodada.

        Parâmetros:
        -----------
        global_weights : Dict[str, np.ndarray]
            Dicionário contendo os pesos globais do modelo, onde as chaves são os nomes dos parâmetros do modelo 
            e os valores são os pesos correspondentes, no formato de arrays NumPy.

        Retorno:
        --------
        Tuple[Dict[str, np.ndarray], Dict[str, float]]
            - O primeiro elemento é um dicionário com os pesos do modelo local após a rodada de treinamento.
            - O segundo elemento é um dicionário com as métricas da rodada, incluindo:
            - `A_k`: A variação da perda calculada antes e depois do treinamento.
            - `a_bit`: O alerta emitido, baseado na variação da perda e no limiar calculado.
            - `round`: O número da rodada de treinamento.
            - `L_before`: A perda antes do treinamento.
            - `L_after`: A perda após o treinamento.

        Exemplos:
        ---------
        # Treinando uma rodada com pesos globais e recebendo os resultados
        global_weights = {"weight": np.array([...]), "bias": np.array([...])}
        weights, metrics = client.fit_one_round(global_weights)
        print("Pesos atualizados:", weights)
        print("Métricas da rodada:", metrics)
        """

        self.round += 1
        self.set_weights(global_weights)

        # Loss before
        L_before = self._evaluate_loss()
        # Train
        L_after = None
        L_last = None
        for _ in range(int(self.cfg.get("local_epochs", 1))):
            L_last = self._local_epoch(lr=self.cfg.get("lr", 0.05), epochs=1, batch=self.cfg.get("batch", 64))
        L_after = self._evaluate_loss()

        A_k = abs(L_before - L_after)

        # Warm-up collect
        if self.round <= self.B:
            self.ak_hist.append(A_k)
            a_bit = 0
        else:
            if self.mu is None:
                arr = np.array(self.ak_hist) if len(self.ak_hist) > 0 else np.array([A_k])
                self.mu = float(arr.mean())
                self.sigma = float(arr.std() + 1e-6)
            tau = self.mu + self.z * self.sigma
            a_bit = int(A_k > tau)

        # Optional randomized response for LDP
        q = float(self.cfg.get("ldp_q", 0.0))
        if q > 0.0:
            flip = (np.random.rand() < q)
            if flip:
                a_bit = 1 - a_bit

        weights = self.get_weights()
        metrics = {"A_k": float(A_k), "a_bit": int(a_bit), "round": int(self.round),
                    "L_before": float(L_before), "L_after": float(L_after)}
        return weights, metrics
