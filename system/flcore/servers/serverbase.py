import math
import torch
import os
import numpy as np
import h5py
import copy
import time
import random
import sys

from utils.data_utils import read_client_data
from utils.dlg import DLG


class Server(object):


    def __init__(self, args, times):
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
        self.algorithm = args.algorithm
        self.time_select = args.time_select
        self.goal = args.goal
        self.time_threthold = args.time_threthold
        self.save_folder_name = args.save_folder_name
        self.top_cnt = 100
        self.auto_break = args.auto_break

        self.clients = []
        self.selected_clients = []
        self.train_slow_clients = []
        self.send_slow_clients = []
    #-----------------------------------------------------------------
        self.users = []
    #-----------------------------------------------------------------
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
        self.fine_tuning_epoch = args.fine_tuning_epoch

        self.client_drop = []

    def set_clients(self, clientObj):
        for i, train_slow, send_slow in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients):
            train_data = read_client_data(self.dataset, i, is_train=True)
            test_data = read_client_data(self.dataset, i, is_train=False)
            client = clientObj(self.args, 
                            id=i, 
                            train_samples=len(train_data), 
                            test_samples=len(test_data), 
                            train_slow=train_slow, 
                            send_slow=send_slow)
            self.clients.append(client)

    # random select slow clients
    def select_slow_clients(self, slow_rate):
        slow_clients = [False for i in range(self.num_clients)]
        idx = [i for i in range(self.num_clients)]
        idx_ = np.random.choice(idx, int(slow_rate * self.num_clients))
        for i in idx_:
            slow_clients[i] = True

        return slow_clients


    def set_slow_clients(self):
        self.train_slow_clients = self.select_slow_clients(
            self.train_slow_rate)
        self.send_slow_clients = self.select_slow_clients(
            self.send_slow_rate)

    
    def client_malicious(self, cliente):
        cliente.set_parameters_malicioso(cliente.model)


    def select_clients(self):
        if self.random_join_ratio:
            self.current_num_join_clients = np.random.choice(range(self.num_join_clients, self.num_clients+1), 1, replace=False)[0]
            print(self.current_num_join_clients)
        else:
            self.current_num_join_clients = self.num_join_clients
        selected_clients = list(np.random.choice(self.clients, self.current_num_join_clients, replace=False))

        print(f'Selected Clients: {len(selected_clients)} clients')
        
        return selected_clients


    def send_models(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()
            
            client.set_parameters(self.global_model)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)


    def replace_clients(self):
        substitutes_clients = []
        removed_clients = set()  # Conjunto para armazenar IDs de clientes que já saíram

        print('')
        print([client.id for client in self.client_drop])
        print([client.id for client in self.new_clients])

        for client_drop in self.client_drop[:]:
            # Encontra um cliente substituto que ainda não foi removido
            substitute_client = min(
                [client for client in self.new_clients if client.id not in removed_clients],
                key=lambda new_clients: abs(new_clients.train_samples - client_drop.train_samples)
            )
            
            print(f'\t\tSai: {client_drop.id}\t\t =============> \tEntra: {substitute_client.id}')
            
            # Marca o cliente como removido
            removed_clients.add(substitute_client.id)
            removed_clients.add(client_drop.id)

            # Atualiza as listas
            self.new_clients.append(client_drop)
            self.client_drop.remove(client_drop)
            substitutes_clients.append(substitute_client)
            self.new_clients.remove(substitute_client)

        print([client.id for client in self.client_drop])
        print([client.id for client in self.new_clients])
        print('')

        return substitutes_clients



    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        # active_clients = random.sample(
        #     self.selected_clients, int((1-self.client_drop_rate) * len(self.selected_clients)))

        active_clients = random.sample(
            self.selected_clients, int((1-self.client_drop_rate) * self.current_num_join_clients))

        self.client_drop = [client for client in self.selected_clients if client not in active_clients]
        
        print('===========================================================================')
        print(f'Selected_Clients: {len([client.id for client in self.selected_clients])}')
        print(f'Active_clients: {len([client.id for client in active_clients])}')
        print(f'Client_drop: {len([client.id for client in self.client_drop])}')
        print(f'Client_not_selected: {len([client.id for client in self.new_clients])}')

        if len(self.client_drop) > 0:
            substitutes = self.replace_clients()
            active_clients.extend(substitutes)
            

        print('===========================================================================')
        print(f'Selected_Clients: {len([client.id for client in self.selected_clients])}')
        print(f'Active_clients: {len([client.id for client in active_clients])}')
        print(f'Client_drop: {len([client.id for client in self.client_drop])}')
        print(f'Client_not_selected: {len([client.id for client in self.new_clients])}')

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
        assert (len(self.uploaded_models) > 0)
        add_pr = []

        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data.zero_()
        

        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            add_pr += self.add_parameters(w, client_model)
        add_pr = [self.valueOfList(add_pr)]

        return add_pr


    def add_parameters(self, w, client_model):
        vp = np.array([])
        valores = []
        media = []
        for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            # client_param = client_param * 1000
            # print(type(client_param))
            # print(len(client_param))
            # print(client_param)
            # sys.exit()
            valores.extend(self.valueOfList(client_param.data.clone()))
            server_param.data += client_param.data.clone() * w
        
        vp = np.append(vp, valores)
        media.append(float(vp.sum() / len(vp)))
        
        return media


    #-------------------------------- My functions---------------------------------------#
    

    def valueOfList(self, value):
        """
    Retorna uma lista que contém todos os valores inteiros e flutuantes contidos em uma estrutura de dados aninhada.

    Args:
        value (list, np.ndarray, torch.Tensor): A estrutura de dados da qual você deseja extrair valores inteiros e flutuantes.

    Returns:
        list: Uma lista contendo todos os valores inteiros e flutuantes encontrados na estrutura de dados.

    Note:
        - A função aceita estruturas de dados aninhadas (listas dentro de listas).
        - Se `value` for uma instância de `torch.Tensor`, ela será convertida em um array numpy e depois em uma lista.
        - Se `value` for uma instância de `np.ndarray`, ela será convertida em uma lista.
        - A função recursivamente percorre estruturas de dados aninhadas para extrair todos os valores inteiros e flutuantes.
        - Os valores inteiros e flutuantes extraídos são adicionados à lista `valueList`.

    Exemplos:
        >>> obj = SuaClasse()
        >>> tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        >>> lista = [1, [2, 3.5], [[4, 5.0], 6]]
        >>> obj.valueOfList(tensor)
        [1.0, 2.0, 3.0, 4.0]
        >>> obj.valueOfList(lista)
        [1, 2, 3.5, 4, 5.0, 6]
        """

        valueList = list()
    
        if type(value) == torch.Tensor:
            value = value.cpu().numpy()
            value = value.tolist()
            self.valueOfList(value)

        if type(value) == np.ndarray:
            value = value.tolist()
            self.valueOfList(value)

        if type(value) == list and len(value) > 0:
            for i in range(len(value)):
                if type(value[i]) == list and len(value[i]) > 0:
                    valueList += self.valueOfList(value[i])

                elif type(value[i]) == int or type(value[i]) == float:
                    valueList.append(value[i])

            return valueList
    
    
    #-------------------------------- My functions---------------------------------------#

    
    def save_global_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        torch.save(self.global_model, model_path)


    def load_model(self):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        assert (os.path.exists(model_path))
        self.global_model = torch.load(model_path)


    def model_exists(self):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        return os.path.exists(model_path)
        

    def save_results(self):
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
        if not os.path.exists(self.save_folder_name):
            os.makedirs(self.save_folder_name)
        torch.save(item, os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))


    def load_item(self, item_name):
        return torch.load(os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))


    def test_metrics(self):
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


    # evaluate selected clients
    def evaluate(self, acc=None, loss=None):
        stats = self.test_metrics()
        stats_train = self.train_metrics()

        test_acc = sum(stats[2])*1.0 / sum(stats[1])
        test_auc = sum(stats[3])*1.0 / sum(stats[1])
        train_loss = sum(stats_train[2])*1.0 / sum(stats_train[1])
        accs = [a / n for a, n in zip(stats[2], stats[1])]
        aucs = [a / n for a, n in zip(stats[3], stats[1])]
        
        # Escrever o valor da variável no arquivo
        with open("saida.txt", "a") as arquivo:
            arquivo.write(str(test_acc) + "," + str(train_loss)+ "," + str(test_auc)+ "\n")


        if acc == None:
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)
        
        if loss == None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)

        print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Test Accurancy: {:.4f}".format(test_acc))
        print("Averaged Test AUC: {:.4f}".format(test_auc))
        # self.print_(test_acc, train_acc, train_loss)
        print("Std Test Accurancy: {:.4f}".format(np.std(accs)))
        print("Std Test AUC: {:.4f}".format(np.std(aucs)))


    def print_(self, test_acc, test_auc, train_loss):
        print("Average Test Accurancy: {:.4f}".format(test_acc))
        print("Average Test AUC: {:.4f}".format(test_auc))
        print("Average Train Loss: {:.4f}".format(train_loss))


    def check_done(self, acc_lss, top_cnt=None, div_value=None):
        for acc_ls in acc_lss:
            if top_cnt != None and div_value != None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_top and find_div:
                    pass
                else:
                    return False
            elif top_cnt != None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                if find_top:
                    pass
                else:
                    return False
            elif div_value != None:
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_div:
                    pass
                else:
                    return False
            else:
                raise NotImplementedError
        return True


    def call_dlg(self, R):
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
        for i in range(self.num_clients, self.num_clients + self.num_new_clients):
            train_data = read_client_data(self.dataset, i, is_train=True)
            test_data = read_client_data(self.dataset, i, is_train=False)
            client = clientObj(self.args, 
                            id=i, 
                            train_samples=len(train_data), 
                            test_samples=len(test_data), 
                            train_slow=False, 
                            send_slow=False)
            self.new_clients.append(client)


    # fine-tuning on new clients
    def fine_tuning_new_clients(self):
        for client in self.new_clients:
            client.set_parameters(self.global_model)
            opt = torch.optim.SGD(client.model.parameters(), lr=self.learning_rate)
            CEloss = torch.nn.CrossEntropyLoss()
            trainloader = client.load_train_data()
            client.model.train()
            for e in range(self.fine_tuning_epoch):
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


    # evaluating on new clients
    def test_metrics_new_clients(self):
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

