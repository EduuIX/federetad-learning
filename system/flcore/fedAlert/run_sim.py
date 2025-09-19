import argparse, yaml, numpy as np
from typing import Dict, Any
from fedalert.datasets import make_synthetic_partitions, apply_cohort_drift
from fedalert.client import FedAlertClient
from fedalert.strategy import FedAlertStrategy

def main():
    """
    Função principal para executar o treinamento federado com clientes e coortes, considerando eventos de 
    drift (mudança de dados ao longo das rodadas) e treinamento distribuído de modelos.

    A função configura e executa o treinamento federado usando a classe `FedAlertClient` para representar 
    os clientes que treinam localmente e uma estratégia de aprendizado federado para a agregação dos modelos. 
    Além disso, lida com a configuração de coortes de clientes e a simulação de drift nos dados durante o treinamento.

    Etapas principais:
    ---------------
    1. Carrega a configuração a partir de um arquivo YAML.
    2. Inicializa os dados e particiona-os para cada cliente.
    3. Cria e organiza os clientes em coortes, com uma distribuição proporcional opcional.
    4. Configura os eventos de drift que podem ocorrer ao longo das rodadas de treinamento.
    5. Executa o treinamento federado para um número específico de rodadas, com a aplicação de drift 
    e a seleção de participantes para cada rodada.
    6. Agrega os resultados de cada rodada e atualiza os pesos globais.
    7. Registra os logs no diretório de saída.

    Parâmetros:
    -----------
    Nenhum parâmetro é passado diretamente para a função. A configuração e outros parâmetros são 
    carregados a partir do arquivo de configuração YAML especificado na linha de comando.

    Fluxo de Execução:
    -------------------
    - A função começa com a análise dos argumentos da linha de comando utilizando `argparse`:
    - **`--config`**: Especifica o caminho do arquivo de configuração (default: "configs/exp.yaml").
    - **`--baseline`**: Caso especificado, executa o treinamento no modo de baseline (Stage-2 sempre ativado).
    
    - O arquivo YAML de configuração é carregado e utilizado para definir os parâmetros do experimento, como:
    - Número de clientes (`num_clients`), número de características (`n_features`), e o número de amostras por cliente (`samples_per_client`).
    - Proporções e contagem das coortes de clientes, bem como eventos de drift e suas programações.
    
    - Para cada rodada de treinamento, a função realiza os seguintes passos:
    1. Aplica eventos de drift, se necessário, para simular a mudança nas distribuições de dados.
    2. Seleciona aleatoriamente um subconjunto de clientes para participar da rodada.
    3. Executa o treinamento local para cada cliente selecionado, coletando os pesos atualizados e as métricas de desempenho.
    4. Agrega os resultados de todos os clientes participantes e atualiza os pesos globais.

    Retorno:
    --------
    Nenhum. A função não retorna um valor explícito, mas imprime uma mensagem de conclusão e gera logs 
    no diretório de saída (`outputs`).

    Exemplo de Uso:
    ---------------
    # Executando o treinamento federado a partir da linha de comando:
    python script.py --config configs/exp.yaml --baseline

    """

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/exp.yaml")
    ap.add_argument("--baseline", action="store_true", help="Run in baseline (always-on Stage-2) mode.")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    num_clients = int(cfg.get("num_clients", 30))
    feats = int(cfg.get("n_features", 20))
    parts = make_synthetic_partitions(
        num_clients, n_features=feats, samples_per_client=cfg.get("samples_per_client", 400)
    )

    # ---- Cohorts ----
    cohorts = {}
    props = (cfg.get("cohorts", {}) or {}).get("proportions", None)
    C = int((cfg.get("cohorts", {}) or {}).get("count", 1))
    if props is None:
        props = [1.0 / C] * C
    sizes = np.random.multinomial(num_clients, props)
    ids = np.arange(num_clients)
    np.random.shuffle(ids)
    start = 0
    for c, n in enumerate(sizes):
        cohorts[c] = [str(x) for x in ids[start:start+n]]
        start += n

    # Drift schedule
    drift_events = (cfg.get("cohorts", {}) or {}).get("drift", [])
    drift_schedule = {int(ev["round"]): ev for ev in drift_events}
    drift_active = {c: False for c in cohorts.keys()}

    # Reverse map: client -> cohort
    client_to_cohort = {}
    for c, idlist in cohorts.items():
        for cid in idlist:
            client_to_cohort[str(cid)] = int(c)

    # ---- Clients ----
    clients: Dict[str, FedAlertClient] = {}
    for cid in range(num_clients):
        cl = FedAlertClient(str(cid), parts[str(cid)]["x"], parts[str(cid)]["y"], cfg)
        clients[str(cid)] = cl

    # Init global weights
    init_w = clients["0"].get_weights()
    strat = FedAlertStrategy(cfg, init_w, cohorts=cohorts, baseline_mode=args.baseline)

    rounds = int(cfg.get("rounds", 200))
    fit_frac = float(cfg.get("fit_fraction", 0.2))
    m_participants = max(1, int(round(num_clients * fit_frac)))

    for r in range(1, rounds + 1):
        # Apply drift events
        if r in drift_schedule:
            apply_cohort_drift(parts, cohorts, r, drift_schedule, drift_active)

        # Sample participants
        selected = np.random.choice(list(clients.keys()), size=m_participants, replace=False).tolist()
        w_curr = strat.weights

        # Local fit
        results = []
        for cid in selected:
            nw, met = clients[cid].fit_one_round(w_curr)
            met["cohort"] = int(client_to_cohort[cid])  # pass cohort id with metrics
            results.append((cid, nw, met))

        # Aggregate + server hook
        strat.aggregate_fit(results)
        strat.on_round_end(results)

    print("Done. Check the 'outputs' directory for logs.")

if __name__ == "__main__":
    main()