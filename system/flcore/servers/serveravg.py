import torch
import time
from flcore.clients.clientavg import clientAVG
from flcore.servers.serverbase import Server
from threading import Thread
import os
import sys
import numpy as np


class FedAvg(Server):


    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientAVG)
        self.set_new_clients(clientAVG)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print(f"\nCurrent total users: {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
    

    def select_best_entropy(self):
        dict_clients = {}
        entropies = [0] * len(self.clients)  # initialize entropies with zeros
        for i, client in enumerate(self.clients):
            entropies[i] = client.client_entropy()
            dict_clients[client] = entropies[i]
        #print(entropies)  # print the list of entropies
        #print(dict_clients)
        
        # Sort the dict_clients dictionary by its values (i.e., the entropies) in descending order
        sorted_clients = sorted(dict_clients.items(), key=lambda x: x[1], reverse=True)
    
        # Select the top percentage of the clients with the highest entropies
        num_clients = len(sorted_clients)
        selected_clients = [client for client, entropy in sorted_clients[:int(num_clients * self.join_ratio)]]

        
        print(f'Selected Clients: {len(selected_clients)} clients')
        return selected_clients
    

    def select_clients_bellow_average(self):
        list_of_accuracies = []
        average_accuracy = [] 


        for i in range(len(self.clients)):
            list_of_accuracies.append(self.clients[i].test_metrics()[0])


        average_accuracy = np.mean(list_of_accuracies)
        selected_clients = []


        for idx_accuracy in range(len(list_of_accuracies)):
            if list_of_accuracies[idx_accuracy] < average_accuracy:
                selected_clients.append(self.clients[idx_accuracy])
        print(f'Selected Clients: {len(selected_clients)} clients')
        return selected_clients


    def select_power_of_choice(self):
        """
        Função para implementar a estratégia POWER-OF-CHOICE de seleção de clientes.

        Parâmetros:
        - total: tamanho do dataset
        - pk: Lista de frações de dados para cada cliente.
        - d: Número de escolhas a serem consideradas.
        - CK: Parâmetro adicional para a seleção de clientes.

        Retorna:
        - Lista dos de clientes selecionados para a próxima rodada.
        """
        
        # Parametros:
        total = sum(cliente.total_samples for cliente in self.clients)
        pk = [(cliente.total_samples / total) for cliente in self.clients]
        d = 50
        CK = 20

        # Passo 1: Amostra o conjunto de clientes candidatos A
        candidate_set = np.random.choice(len(pk), size=d, replace=False, p=pk)

        # Passo 2: Perdas Locais
        local_losses = []
        for i in candidate_set:
            self.clients[i].set_parameters(self.global_model)
            local_losses.append(self.clients[i].test_metrics()[0])

        # Passo 3: Seleciona os clientes com as maiores perdas
        active_set = candidate_set[np.argsort(local_losses)[-max(CK, 1):]]
        
        selected_clients = [self.clients[i] for i in active_set]

        # indice_final = int(len(selected_clients) * self.join_ratio)
        # selected_clients = selected_clients[:indice_final]
        print(f'Selected Clients: {len(selected_clients)} clients')
        return selected_clients

    
    def simula_mobilidade(self, i, clientes):
        if i == 0:
            self.selected_clients = clientes[0:20]
        elif i > 0 and len(self.selected_clients) < 60:
            self.selected_clients = clientes[0:(20+i)]
        else:
            self.selected_clients = clientes

        print(f'Nº clientes selecionados=>>>>>>>> {len(self.selected_clients)}')


    def manipula_gradiente(self, client):
        client.set_parameters_malicioso(client.model)


    def treinamento(self, args, i):
        s_t = time.time()

        self.send_models()

        if i%self.eval_gap == 0:
            print(f"\n-------------Round number: {i}-------------")
            print("\nEvaluate global model")
            self.evaluate()

        for client in self.selected_clients:
            client.train()
            if self.args.client_malicious and client.id == 1:
                self.manipula_gradiente(client)
                

        # threads = [Thread(target=client.train)
        #            for client in self.selected_clients]
        # [t.start() for t in threads]
        # [t.join() for t in threads]

        self.receive_models(i)
        # self.new_clients.extend(self.client_drop)
        if self.dlg_eval and i%self.dlg_gap == 0:
            self.call_dlg(i)
        
        self.users += self.aggregate_parameters()

        self.Budget.append(time.time() - s_t)
        
        print('-'*25, 'time cost', '-'*25, self.Budget[-1])
        with open("tempo.txt", "a") as arquivo:
            arquivo.write(str(self.Budget[-1]) + "\n")


    def train(self, args):
        caminhoAtual = os.getcwd()
        destino = f'{caminhoAtual}/models/fmnist/FedAvg_server.pt'
    
        for i in range(self.global_rounds+1):


            if args.entropy:
                self.selected_clients = self.select_best_entropy()
                print('Entropy Selection')
            elif args.bellow_average:
                self.selected_clients = self.select_clients_bellow_average()
                print('Bellow Average Selection')
            elif args.power_of_choice:
                self.selected_clients = self.select_power_of_choice()
                print('Power of Choice Selection')
            else:
                self.selected_clients = self.select_clients()
                print('Normal Selection')

            # if i == 0 and os.path.exists(destino):
            #     self.load_model()
            #     for cliente in self.selected_clients:
            #         cliente.model = self.global_model
            # self.treinamento(args, i)
            # if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
            #     break{}

            #else:


            # self.simula_mobilidade(i, clientes)
            
            
            self.treinamento(args, i)
            self.users = []
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
            self.set_new_clients(clientAVG)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()

