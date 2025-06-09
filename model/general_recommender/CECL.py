import torch
from torch.serialization import save
import torch.sparse as torch_sp
import torch.nn as nn
import torch.nn.functional as F
from model.base import AbstractRecommender
from util.pytorch import inner_product, l2_loss
from util.pytorch import get_initializer
from util.common import Reduction
from data import PointwiseSamplerV2, PairwiseSamplerV2
import numpy as np
from time import time
from reckit import timer
import scipy.sparse as sp
from util.common import normalize_adj_matrix, ensureDir
from util.pytorch import sp_mat_to_sp_tensor
from reckit import randint_choice
import sys
# python搜索模块路径设置
sys.path.append('/home/admin/duhh/SGL/model/general_recommender/')
from get_params import get_user_params, get_item_params





class _LightGCN(nn.Module):
    def __init__(self, num_users, num_items, embed_dim, norm_adj, n_layers):
        super(_LightGCN, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embed_dim = embed_dim
        self.norm_adj = norm_adj
        self.n_layers = n_layers
        self.user_embeddings = nn.Embedding(self.num_users, self.embed_dim)
        self.item_embeddings = nn.Embedding(self.num_items, self.embed_dim)
        self.dropout = nn.Dropout(0.3)
        self._user_embeddings_final = None
        self._item_embeddings_final = None

        # # weight initialization
        # self.reset_parameters()

        self.lstm = nn.LSTM(input_size=self.embed_dim, hidden_size=self.embed_dim,
                            num_layers=self.n_layers)  # 引入LSTM时间信息

    def reset_parameters(self, pretrain=0, init_method="uniform", dir=None):
        if pretrain == '1':
            pretrain_user_embedding = np.load(dir + 'user_embeddings.npy')
            pretrain_item_embedding = np.load(dir + 'item_embeddings.npy')
            pretrain_user_tensor = torch.FloatTensor(pretrain_user_embedding).cuda()
            pretrain_item_tensor = torch.FloatTensor(pretrain_item_embedding).cuda()
            self.user_embeddings = nn.Embedding.from_pretrained(pretrain_user_tensor)
            self.item_embeddings = nn.Embedding.from_pretrained(pretrain_item_tensor)

        else:
            # init = get_initializer(init_method)
            # init(self.user_embeddings.weight)
            # init(self.item_embeddings.weight)

            # # #--- * 添加代码 *---
            # 定义一个随机的embedding向量
            # random_embedding = torch.randn(19, self.embed_dim) # movie
            # random_embedding = torch.randn(9749, self.embed_dim) # music
            # random_embedding = torch.randn(68, self.embed_dim) # clothing
            # random_embedding = torch.randn(7, self.embed_dim)  # modcloth_fit
            random_embedding = torch.randn(4, self.embed_dim)  # modcloth
            # random_embedding = torch.randn(10, self.embed_dim)  # electronics
            # random_embedding = torch.randn(104, self.embed_dim)  # beeradvocate
            # random_embedding = torch.randn(89, self.embed_dim)  # beer

            # 加载user_category.txt和item_category.txt中的固定参数
            user_params = torch.from_numpy(get_user_params('dataset/modcloth/user_category.txt')).float()
            item_params = torch.from_numpy(get_item_params('dataset/modcloth/item_category.txt')).float()

            # 创建新的可学习的参数矩阵
            user_params_learnable = torch.zeros_like(user_params)
            item_params_learnable = torch.zeros_like(item_params)

            # 遍历user_params和item_params，判断是否为0，并设置对应位置的参数值
            for i in range(user_params.shape[0]):
                for j in range(user_params.shape[1]):
                    if user_params[i][j] != 0:
                        # user_params_learnable[i][j] = torch.nn.Parameter(torch.tensor(0.1))  # 设置为学习的可训练参数
                        user_params_learnable[i][j] = torch.nn.Parameter(0.1 + 0.9 * torch.rand(1))  # 设置为学习的可训练参数

            for i in range(item_params.shape[0]):
                for j in range(item_params.shape[1]):
                    if item_params[i][j] != 0:
                        # item_params_learnable[i][j] = torch.nn.Parameter(torch.tensor(0.1))  # 设置为学习的可训练参数
                        item_params_learnable[i][j] = torch.nn.Parameter(0.1 + 0.9 * torch.rand(1))  # 设置为学习的可训练参数

            # 将user_params_learnable和item_params_learnable与random_embedding相乘，得到新的可学习的参数矩阵
            user_params_learnable = user_params_learnable.mm(random_embedding)
            item_params_learnable = item_params_learnable.mm(random_embedding)

            # 加载u_c_r.txt文件中的参数矩阵
            u_c_r_matrix = torch.from_numpy(np.loadtxt('dataset/modcloth/u_c_r.txt')).float()

            # 确保u_c_r_matrix与user_params_learnable具有相同的设备（CPU或CUDA）
            u_c_r_matrix = u_c_r_matrix.to(user_params_learnable.device)

            # 将两个矩阵相加
            # user_params_learnable = user_params_learnable + u_c_r_matrix
            user_params_learnable = 0.5 * user_params_learnable + 0.5 * u_c_r_matrix

            # 将新的可学习的参数矩阵赋值给self.user_embeddings.weight.data和self.item_embeddings.weight.data
            self.user_embeddings.weight.data.copy_(user_params_learnable.float())
            self.item_embeddings.weight.data.copy_(item_params_learnable.float())

            # 设置可学习的参数矩阵为模型的可学习参数
            self.user_embeddings.weight.requires_grad = True
            self.item_embeddings.weight.requires_grad = True

    def forward(self, sub_graph1, sub_graph2, users, items, neg_items):
        user_embeddings, item_embeddings = self._forward_gcn(self.norm_adj)
        user_embeddings1, item_embeddings1 = self._forward_gcn(sub_graph1)
        user_embeddings2, item_embeddings2 = self._forward_gcn(sub_graph2)

        # Normalize embeddings learnt from sub-graph to construct SSL loss
        user_embeddings1 = F.normalize(user_embeddings1, dim=1)
        item_embeddings1 = F.normalize(item_embeddings1, dim=1)
        user_embeddings2 = F.normalize(user_embeddings2, dim=1)
        item_embeddings2 = F.normalize(item_embeddings2, dim=1)

        user_embs = F.embedding(users, user_embeddings)
        item_embs = F.embedding(items, item_embeddings)
        neg_item_embs = F.embedding(neg_items, item_embeddings)
        user_embs1 = F.embedding(users, user_embeddings1)
        item_embs1 = F.embedding(items, item_embeddings1)
        user_embs2 = F.embedding(users, user_embeddings2)
        item_embs2 = F.embedding(items, item_embeddings2)

        sup_pos_ratings = inner_product(user_embs, item_embs)  # [batch_size]
        sup_neg_ratings = inner_product(user_embs, neg_item_embs)  # [batch_size]
        sup_logits = sup_pos_ratings - sup_neg_ratings  # [batch_size]

        pos_ratings_user = inner_product(user_embs1, user_embs2)  # [batch_size]
        pos_ratings_item = inner_product(item_embs1, item_embs2)  # [batch_size]
        tot_ratings_user = torch.matmul(user_embs1,
                                        torch.transpose(user_embeddings2, 0, 1))  # [batch_size, num_users]
        tot_ratings_item = torch.matmul(item_embs1,
                                        torch.transpose(item_embeddings2, 0, 1))  # [batch_size, num_items]

        ssl_logits_user = tot_ratings_user - pos_ratings_user[:, None]  # [batch_size, num_users]
        ssl_logits_item = tot_ratings_item - pos_ratings_item[:, None]  # [batch_size, num_users]

        return sup_logits, ssl_logits_user, ssl_logits_item

    def _forward_gcn(self, norm_adj):
        ego_embeddings = torch.cat([self.user_embeddings.weight, self.item_embeddings.weight], dim=0)
        all_embeddings = [ego_embeddings]

        for k in range(self.n_layers):
            if isinstance(norm_adj, list):
                ego_embeddings = torch_sp.mm(norm_adj[k], ego_embeddings)
            else:
                ego_embeddings = torch_sp.mm(norm_adj, ego_embeddings)
            all_embeddings += [ego_embeddings]

        all_embeddings = torch.stack(all_embeddings, dim=1).mean(dim=1)
        user_embeddings, item_embeddings = torch.split(all_embeddings, [self.num_users, self.num_items], dim=0)

        return user_embeddings, item_embeddings

    def predict(self, users):
        if self._user_embeddings_final is None or self._item_embeddings_final is None:
            raise ValueError("Please first switch to 'eval' mode.")
        user_embs = F.embedding(users, self._user_embeddings_final)
        temp_item_embs = self._item_embeddings_final
        ratings = torch.matmul(user_embs, temp_item_embs.T)
        return ratings

    def eval(self):
        super(_LightGCN, self).eval()
        self._user_embeddings_final, self._item_embeddings_final = self._forward_gcn(self.norm_adj)


class SGL(AbstractRecommender):
    def __init__(self, config):
        super(SGL, self).__init__(config)

        self.config = config
        self.model_name = config["recommender"]
        self.dataset_name = config["dataset"]

        # General hyper-parameters
        self.reg = config['reg']
        self.emb_size = config['embed_size']
        self.batch_size = config['batch_size']
        self.test_batch_size = config['test_batch_size']
        self.epochs = config["epochs"]
        self.verbose = config["verbose"]
        self.stop_cnt = config["stop_cnt"]
        self.learner = config["learner"]
        self.lr = config['lr']
        self.param_init = config["param_init"]

        # Hyper-parameters for GCN
        self.n_layers = config['n_layers']

        # Hyper-parameters for SSL
        self.ssl_aug_type = config["aug_type"].lower()
        assert self.ssl_aug_type in ['nd', 'ed', 'rw']
        self.ssl_reg = config["ssl_reg"]
        self.ssl_ratio = config["ssl_ratio"]
        self.ssl_mode = config["ssl_mode"]
        self.ssl_temp = config["ssl_temp"]

        # Other hyper-parameters
        self.best_epoch = 0
        self.best_result = np.zeros([2], dtype=float)

        self.model_str = '#layers=%d-reg=%.0e' % (
            self.n_layers,
            self.reg
        )
        self.model_str += '/ratio=%.1f-mode=%s-temp=%.2f-reg=%.0e' % (
            self.ssl_ratio,
            self.ssl_mode,
            self.ssl_temp,
            self.ssl_reg
        )
        self.pretrain_flag = config["pretrain_flag"]
        if self.pretrain_flag:
            self.epochs = 0
        self.save_flag = config["save_flag"]
        self.save_dir, self.tmp_model_dir = None, None
        if self.pretrain_flag or self.save_flag:
            self.tmp_model_dir = config.data_dir + '%s/model_tmp/%s/%s/' % (
                self.dataset_name,
                self.model_name,
                self.model_str)
            self.save_dir = config.data_dir + '%s/pretrain-embeddings/%s/n_layers=%d/' % (
                self.dataset_name,
                self.model_name,
                self.n_layers,)
            ensureDir(self.tmp_model_dir)
            ensureDir(self.save_dir)

        self.num_users, self.num_items, self.num_ratings = self.dataset.num_users, self.dataset.num_items, self.dataset.num_train_ratings

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        adj_matrix = self.create_adj_mat()
        adj_matrix = sp_mat_to_sp_tensor(adj_matrix).to(self.device)

        self.lightgcn = _LightGCN(self.num_users, self.num_items, self.emb_size,
                                  adj_matrix, self.n_layers).to(self.device)
        if self.pretrain_flag:
            self.lightgcn.reset_parameters(pretrain=self.pretrain_flag, dir=self.save_dir)
        else:
            self.lightgcn.reset_parameters(init_method=self.param_init)
        self.optimizer = torch.optim.Adam(self.lightgcn.parameters(), lr=self.lr)

    @timer
    def create_adj_mat(self, is_subgraph=False, aug_type='ed'):
        n_nodes = self.num_users + self.num_items
        users_items = self.dataset.train_data.to_user_item_pairs()
        users_np, items_np = users_items[:, 0], users_items[:, 1]

        if is_subgraph and self.ssl_ratio > 0:
            if aug_type == 'nd':
                drop_user_idx = randint_choice(self.num_users, size=self.num_users * self.ssl_ratio, replace=False)
                drop_item_idx = randint_choice(self.num_items, size=self.num_items * self.ssl_ratio, replace=False)
                indicator_user = np.ones(self.num_users, dtype=np.float32)
                indicator_item = np.ones(self.num_items, dtype=np.float32)
                indicator_user[drop_user_idx] = 0.
                indicator_item[drop_item_idx] = 0.
                diag_indicator_user = sp.diags(indicator_user)
                diag_indicator_item = sp.diags(indicator_item)
                R = sp.csr_matrix(
                    (np.ones_like(users_np, dtype=np.float32), (users_np, items_np)),
                    shape=(self.num_users, self.num_items))
                R_prime = diag_indicator_user.dot(R).dot(diag_indicator_item)
                (user_np_keep, item_np_keep) = R_prime.nonzero()
                ratings_keep = R_prime.data
                tmp_adj = sp.csr_matrix((ratings_keep, (user_np_keep, item_np_keep + self.num_users)),
                                        shape=(n_nodes, n_nodes))
            if aug_type in ['ed', 'rw']:
                keep_idx = randint_choice(len(users_np), size=int(len(users_np) * (1 - self.ssl_ratio)), replace=False)
                user_np = np.array(users_np)[keep_idx]
                item_np = np.array(items_np)[keep_idx]
                ratings = np.ones_like(user_np, dtype=np.float32)
                tmp_adj = sp.csr_matrix((ratings, (user_np, item_np + self.num_users)), shape=(n_nodes, n_nodes))
        else:
            ratings = np.ones_like(users_np, dtype=np.float32)
            tmp_adj = sp.csr_matrix((ratings, (users_np, items_np + self.num_users)), shape=(n_nodes, n_nodes))
        adj_mat = tmp_adj + tmp_adj.T

        # normalize adjcency matrix
        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj_tmp = d_mat_inv.dot(adj_mat)
        adj_matrix = norm_adj_tmp.dot(d_mat_inv)

        return adj_matrix

    def train_model(self):
        data_iter = PairwiseSamplerV2(self.dataset.train_data, num_neg=1, batch_size=self.batch_size, shuffle=True)
        self.logger.info(self.evaluator.metrics_info())
        stopping_step = 0
        for epoch in range(1, self.epochs + 1):
            total_loss, total_bpr_loss, total_reg_loss = 0.0, 0.0, 0.0
            training_start_time = time()
            if self.ssl_aug_type in ['nd', 'ed']:
                sub_graph1 = self.create_adj_mat(is_subgraph=True, aug_type=self.ssl_aug_type)
                sub_graph1 = sp_mat_to_sp_tensor(sub_graph1).to(self.device)
                sub_graph2 = self.create_adj_mat(is_subgraph=True, aug_type=self.ssl_aug_type)
                sub_graph2 = sp_mat_to_sp_tensor(sub_graph2).to(self.device)
            else:
                sub_graph1, sub_graph2 = [], []
                for _ in range(0, self.n_layers):
                    tmp_graph = self.create_adj_mat(is_subgraph=True, aug_type=self.ssl_aug_type)
                    sub_graph1.append(sp_mat_to_sp_tensor(tmp_graph).to(self.device))
                    tmp_graph = self.create_adj_mat(is_subgraph=True, aug_type=self.ssl_aug_type)
                    sub_graph2.append(sp_mat_to_sp_tensor(tmp_graph).to(self.device))
            self.lightgcn.train().to(self.device)
            for bat_users, bat_pos_items, bat_neg_items in data_iter:
                bat_users = torch.from_numpy(bat_users).long().to(self.device)
                bat_pos_items = torch.from_numpy(bat_pos_items).long().to(self.device)
                bat_neg_items = torch.from_numpy(bat_neg_items).long().to(self.device)
                sup_logits, ssl_logits_user, ssl_logits_item = self.lightgcn(
                    sub_graph1, sub_graph2, bat_users, bat_pos_items, bat_neg_items)

                # BPR Loss
                bpr_loss = -torch.sum(F.logsigmoid(sup_logits))

                # Reg Loss
                reg_loss = l2_loss(
                    self.lightgcn.user_embeddings(bat_users),
                    self.lightgcn.item_embeddings(bat_pos_items),
                    self.lightgcn.item_embeddings(bat_neg_items),
                )

                # InfoNCE Loss
                clogits_user = torch.logsumexp(ssl_logits_user / self.ssl_temp, dim=1)
                clogits_item = torch.logsumexp(ssl_logits_item / self.ssl_temp, dim=1)
                infonce_loss = torch.sum(clogits_user + clogits_item)

                loss = bpr_loss + self.ssl_reg * infonce_loss + self.reg * reg_loss
                # loss = bpr_loss

                total_loss += loss
                total_bpr_loss += bpr_loss
                total_reg_loss += self.reg * reg_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            self.logger.info("[iter %d : loss : %.4f = %.4f + %.4f + %.4f, time: %f]" % (
                epoch,
                total_loss / self.num_ratings,
                total_bpr_loss / self.num_ratings,
                (total_loss - total_bpr_loss - total_reg_loss) / self.num_ratings,
                total_reg_loss / self.num_ratings,
                time() - training_start_time,))

            if epoch % self.verbose == 0 and epoch > self.config['start_testing_epoch']:
                result, flag = self.evaluate_model()
                self.logger.info("epoch %d:\t%s" % (epoch, result))

                # ### 添加代码
                # # Add the following lines to print top recommendations for user ID 0
                # top_recommendations_user_0 = self.get_top_recommendations(user_id=0)
                # self.logger.info("Top-5 recommendations for User ID 0: %s" % top_recommendations_user_0)
                # ### 截至

                if flag:
                    self.best_epoch = epoch
                    stopping_step = 0
                    self.logger.info("Find a better model.")
                    if self.save_flag:
                        self.logger.info("Save model to file as pretrain.")
                        torch.save(self.lightgcn.state_dict(), self.tmp_model_dir)
                        self.saver.save(self.sess, self.tmp_model_dir)
                else:
                    stopping_step += 1
                    if stopping_step >= self.stop_cnt:
                        self.logger.info("Early stopping is trigger at epoch: {}".format(epoch))
                        break

        self.logger.info("best_result@epoch %d:\n" % self.best_epoch)
        if self.save_flag:
            self.logger.info('Loading from the saved best model during the training process.')
            self.lightgcn.load_state_dict(torch.load(self.tmp_model_dir))
            uebd = self.lightgcn.user_embeddings.weight.detach().numpy()
            iebd = self.lightgcn.item_embeddings.weight.detach().numpy()
            np.save(self.save_dir + 'user_embeddings.npy', uebd)
            np.save(self.save_dir + 'item_embeddings.npy', iebd)
            buf, _ = self.evaluate_model()
        elif self.pretrain_flag:
            buf, _ = self.evaluate_model()
        else:
            buf = '\t'.join([("%.4f" % x).ljust(12) for x in self.best_result])
        self.logger.info("\t\t%s" % buf)

    ## 原始代码
    # @timer
    def evaluate_model(self):
        flag = False
        self.lightgcn.eval()
        current_result, buf = self.evaluator.evaluate(self)
        if self.best_result[1] < current_result[1]:
            self.best_result = current_result
            flag = True
        return buf, flag
    # @timer
    # def evaluate_model(self):
    #     flag = False
    #     self.lightgcn.eval()
    #     current_result, buf = self.evaluator.evaluate(self)
    #     if self.best_result[1] < current_result[1]:
    #         self.best_result = current_result
    #         flag = True
    #
    #     # 添加代码开始
    #     target_items = [367, 1529, 2473]
    #     cecl_top20 = [157, 12, 148, 54, 65, 201, 3, 183, 150, 29, 1, 35, 16, 151, 304, 307, 188, 107, 349, 1302]
    #
    #     # 获取项目嵌入
    #     item_embeddings = self.lightgcn._item_embeddings_final
    #
    #     # 计算相似度
    #     target_embs = F.embedding(torch.LongTensor(target_items).to(self.device), item_embeddings)
    #     cecl_embs = F.embedding(torch.LongTensor(cecl_top20).to(self.device), item_embeddings)
    #
    #     similarity = F.cosine_similarity(target_embs.unsqueeze(1), cecl_embs.unsqueeze(0), dim=2)
    #
    #     # 输出相似度
    #     self.logger.info("Item Embedding Similarities:")
    #     for i, target_id in enumerate(target_items):
    #         self.logger.info(f"Target Item {target_id}:")
    #         for j, cecl_id in enumerate(cecl_top20):
    #             self.logger.info(f"  - With Item {cecl_id}: {similarity[i][j].item():.4f}")
    #     # 添加代码结束
    #
    #     return buf, flag

    def predict(self, users):
        users = torch.from_numpy(np.asarray(users)).long().to(self.device)
        return self.lightgcn.predict(users).cpu().detach().numpy()

    def get_top_recommendations(self, user_id, top_k=5):
        user = torch.tensor([user_id]).long().to(self.device)
        item_scores = self.lightgcn.predict(user)
        _, top_indices = torch.topk(item_scores, top_k)
        top_items = top_indices.cpu().numpy().flatten()
        return top_items

