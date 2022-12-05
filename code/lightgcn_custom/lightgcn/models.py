from typing import Optional, Union
from torch_geometric.typing import Adj, OptTensor

import os
import numpy as np
import torch
from torch import Tensor
from sklearn.metrics import accuracy_score, roc_auc_score
from torch_geometric.nn.models import LightGCN
from torch_geometric.nn import MessagePassing
from torch.nn import Embedding, ModuleList
from torch_geometric.nn.conv import LGConv
from torch_sparse import SparseTensor


class MyLightGCN(torch.nn.Module):
    """ Initializes MyLightGCN Model
    LightGCN 기본 클래스의 predict_link 등을 그대로 사용한다. 
    https://arxiv.org/abs/2002.02126

    Args:
        num_nodes (int): 그래프의 노드 수 
        embedding_dim (int): 노드 embedding dimension
        num_layers (int): LGConv layer 수 
        embedding_info (list): embedding 할 information dict list 
                       ( name : column name, value : column 의 element 수, weight : 가중치 )
        alpha (float or Tensor, optional): 레이어에 가산될 가중치
        **kwargs (optional): Additional arguments of the underlying
            :class:`~torch_geometric.nn.conv.LGConv` layers.
    """

    def __init__(
            self,
            num_info:dict,
            embedding_dim: int,
            num_layers: int,
            alpha: Optional[Union[float, Tensor]] = None,
            **kwargs,
        ):
            """
            - userid embedding 
            - assessmentItemId embedding
            - assessmentItemId 의 각 속성 값 
                - knowledge tag
                - category value 
            
            - base embedding value 
                - the number of user 
                - the number of item 
            
            - additional information
                - user additional information 
                - item additional information 
            """
            # super().__init__(num_info["n_user"], embedding_dim,num_layers)
            super().__init__()
            self.embedding_dim = embedding_dim
            self.num_layers = num_layers

            if alpha is None:
                alpha = 1. / (num_layers + 1)

            if isinstance(alpha, Tensor):
                assert alpha.size(0) == num_layers + 1
            else:
                alpha = torch.tensor([alpha] * (num_layers + 1))
            self.register_buffer('alpha', alpha)

            self.user_embedding = Embedding(num_info["n_user"], embedding_dim)
            self.item_embedding = Embedding(num_info["n_item"], embedding_dim)
            self.tag_embedding  = Embedding(num_info["n_tags"], embedding_dim)
            self.testId_embedding  = Embedding(num_info["n_testids"], embedding_dim)
            self.bigcat_embedding  = Embedding(num_info["n_bigcat"], embedding_dim)
            self.convs = ModuleList([LGConv(**kwargs) for _ in range(num_layers)])
        
            # embedding layer 및 convolutional layer 의 weight 초기화 
            self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.user_embedding.weight)
        torch.nn.init.xavier_uniform_(self.item_embedding.weight)
        torch.nn.init.xavier_uniform_(self.tag_embedding.weight)
        torch.nn.init.xavier_uniform_(self.testId_embedding.weight)
        torch.nn.init.xavier_uniform_(self.bigcat_embedding.weight)

        for conv in self.convs:
            conv.reset_parameters()

    def get_embedding(self, edge_index: Adj, additional_info:dict=None, edge_weight: OptTensor = None) -> Tensor:

        item_embedding_weight = self.item_embedding.weight 
        tag_embedding_weight  = self.tag_embedding(additional_info["item"]["KnowledgeTag"] )
        testId_embedding_weight  = self.testId_embedding(additional_info["item"]["testId"] )
        bigcat_embedding_weight  = self.bigcat_embedding(additional_info["item"]["big_category"] )

        total_embedding_weight = item_embedding_weight

        embedding_list = [
                          tag_embedding_weight,
                          testId_embedding_weight,
                          bigcat_embedding_weight]

        for emb in embedding_list : 
            total_embedding_weight = total_embedding_weight + emb

        total_embedding_weight = total_embedding_weight / (len(embedding_list) + 1 )

        # total_embedding_weight = (  item_embedding_weight + tag_embedding_weight + testId_embedding_weigsht + bigcat_embedding_weight ) / 4

        # if additional_info["item"] is not None :
        #     for k in additional_info["item"] :
        #         self.item_embedding( additional_info["item"][k] )

        x = torch.cat([
            self.user_embedding.weight, 
            total_embedding_weight
            ],dim= 0)
        
        # x = self.embedding.weight
        out = x * self.alpha[0]
        
        for i in range(self.num_layers):
            # edge_weight =  torch.ones(edge_index.size(1))
            # print(edge_weight)
            # print(edge_weight.shape)
            x = self.convs[i](x, edge_index, edge_weight = edge_weight)
            out = out + x * self.alpha[i + 1]

        return out


    def forward(self, edge_index: Adj, additional_info:dict=None,
            edge_label_index: OptTensor = None, edge_weight: OptTensor = None) -> Tensor:
        r"""Computes rankings for pairs of nodes.

        Args:
            edge_index (Tensor or SparseTensor): Edge tensor specifying the
                connectivity of the graph.
            additional_info (dict) 
            edge_label_index (Tensor, optional): Edge tensor specifying the
                node pairs for which to compute rankings or probabilities.
                If :obj:`edge_label_index` is set to :obj:`None`, all edges in
                :obj:`edge_index` will be used instead. (default: :obj:`None`)
        """
        if edge_label_index is None:
            if isinstance(edge_index, SparseTensor):
                edge_label_index = torch.stack(edge_index.coo()[:2], dim=0)
            else:
                edge_label_index = edge_index

        out = self.get_embedding(edge_index, additional_info,edge_weight)

        out_src = out[edge_label_index[0]]
        out_dst = out[edge_label_index[1]]
        return (out_src * out_dst).sum(dim=-1)

    def predict_link(self, edge_index: Adj, additional_info:dict=None,edge_label_index: OptTensor = None, edge_weight: OptTensor = None,
                     prob: bool = False) -> Tensor:
        r"""Predict links between nodes specified in :obj:`edge_label_index`.

        Args:
            prob (bool): Whether probabilities should be returned. (default:
                :obj:`False`)
        """
        pred = self(edge_index, additional_info,edge_label_index,edge_weight = edge_weight).sigmoid()
        return pred if prob else pred.round()

    def link_pred_loss(self, pred: Tensor, edge_label: Tensor,
                    **kwargs) -> Tensor:
        r"""Computes the model loss for a link prediction objective via the
        :class:`torch.nn.BCEWithLogitsLoss`.

        Args:
            pred (Tensor): The predictions.
            edge_label (Tensor): The ground-truth edge labels.
            **kwargs (optional): Additional arguments of the underlying
                :class:`torch.nn.BCEWithLogitsLoss` loss function.
        """
        loss_fn = torch.nn.BCEWithLogitsLoss(**kwargs)
        return loss_fn(pred, edge_label.to(pred.dtype))


def build( num_info:dict, weight=None, logger=None, **kwargs):
    model = MyLightGCN( num_info=num_info, **kwargs)
    if weight:
        if not os.path.isfile(weight):
            logger.fatal("Model Weight File Not Exist")
        logger.info("Load model")
        state = torch.load(weight)["model"]
        model.load_state_dict(state)
        return model
    else:
        logger.info("No load model")
        return model


def train(
    model,
    train_data,
    additional_data = None,
    valid_data=None,
    n_epoch=100,
    early_stop = 10,
    learning_rate=0.01,
    use_wandb=False,
    weight=None,
    logger=None,
):
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if not os.path.exists(weight):
        os.makedirs(weight)

    if valid_data is None:
        eids = np.arange(len(train_data["label"]))
        eids = np.random.permutation(eids)[:1000]
        edge, label = train_data["edge"], train_data["label"]
        weight = train_data["weight"]
        label = label.to("cpu").detach().numpy()
        weight = weight.to("cpu").detach().numpy()
        valid_data = dict(edge=edge[:, eids], label=label[eids], weight=weight[eids])

    logger.info(f"Training Started : n_epoch={n_epoch}")
    best_auc, best_epoch = 0, -1
    stop_check = 0 
    
    for e in range(n_epoch):
        # forward
        pred = model(train_data["edge"],additional_data,edge_weight = train_data["weight"])
        loss = model.link_pred_loss(pred, train_data["label"])

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            # additional_data 는 user 와 item 을 key 로 가지고, 각 key 의 값도 dictionary 
            # { user: {} , item : { knowledgeTag : tensor, ...} }
            prob = model.predict_link(valid_data["edge"],additional_data,edge_weight = valid_data["weight"],prob=True)
            prob = prob.detach().cpu().numpy()
            acc = accuracy_score(valid_data["label"].cpu().numpy(), prob > 0.5)
            auc = roc_auc_score(valid_data["label"].cpu().numpy(), prob)
            logger.info(
                f" * In epoch {(e+1):04}, loss={loss:.03f}, acc={acc:.03f}, AUC={auc:.03f}"
            )
            if use_wandb:
                import wandb

                wandb.log(dict(loss=loss, acc=acc, auc=auc))

        if weight:
            if auc > best_auc:
                logger.info(
                    f" * In epoch {(e+1):04}, loss={loss:.03f}, acc={acc:.03f}, AUC={auc:.03f}, Best AUC"
                )
                best_auc, best_epoch = auc, e
                torch.save(
                    {"model": model.state_dict(), "epoch": e + 1},
                    os.path.join(weight, f"best_model.pt"),
                )
                stop_check = 0 
            elif auc < best_auc : 
                stop_check += 1 
            
            if ( stop_check >= early_stop ):
                break

            
    torch.save(
        {"model": model.state_dict(), "epoch": e + 1},
        os.path.join(weight, f"last_model.pt"),
    )
    logger.info(f"Best Weight Confirmed : {best_epoch+1}'th epoch")


def train_kfold(
    model,
    train_data,
    additional_data = None,
    valid_data=None,
    n_epoch=100,
    early_stop = 10,
    learning_rate=0.01,
    use_wandb=False,
    weight=None,
    logger=None,
):
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if not os.path.exists(weight):
        os.makedirs(weight)

    if valid_data is None:
        eids = np.arange(len(train_data["label"]))
        eids = np.random.permutation(eids)[:1000]
        edge, label = train_data["edge"], train_data["label"]
        weight = train_data["weight"]
        label = label.to("cpu").detach().numpy()
        weight = weight.to("cpu").detach().numpy()
        valid_data = dict(edge=edge[:, eids], label=label[eids], weight=weight[eids])

    logger.info(f"Training Started : n_epoch={n_epoch}")
    best_auc, best_epoch = 0, -1
    stop_check = 0 
    
    for e in range(n_epoch):
        # forward
        pred = model(train_data["edge"],additional_data,edge_weight = train_data["weight"])
        loss = model.link_pred_loss(pred, train_data["label"])

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            # additional_data 는 user 와 item 을 key 로 가지고, 각 key 의 값도 dictionary 
            # { user: {} , item : { knowledgeTag : tensor, ...} }
            prob = model.predict_link(valid_data["edge"],additional_data,edge_weight = valid_data["weight"],prob=True)
            prob = prob.detach().cpu().numpy()
            acc = accuracy_score(valid_data["label"].cpu().numpy(), prob > 0.5)
            auc = roc_auc_score(valid_data["label"].cpu().numpy(), prob)
            logger.info(
                f" * In epoch {(e+1):04}, loss={loss:.03f}, acc={acc:.03f}, AUC={auc:.03f}"
            )
            if use_wandb:
                import wandb

                wandb.log(dict(loss=loss, acc=acc, auc=auc))

        if weight:
            if auc > best_auc:
                logger.info(
                    f" * In epoch {(e+1):04}, loss={loss:.03f}, acc={acc:.03f}, AUC={auc:.03f}, Best AUC"
                )
                best_auc, best_epoch = auc, e
                torch.save(
                    {"model": model.state_dict(), "epoch": e + 1},
                    os.path.join(weight, f"best_model.pt"),
                )
                stop_check = 0 
            elif auc < best_auc : 
                stop_check += 1 
            
            if ( stop_check >= early_stop ):
                break

            
    torch.save(
        {"model": model.state_dict(), "epoch": e + 1},
        os.path.join(weight, f"last_model.pt"),
    )
    logger.info(f"Best Weight Confirmed : {best_epoch+1}'th epoch")

def inference(model, data,additional_data, logger=None):
    model.eval()
    with torch.no_grad():
        pred = model.predict_link(data["edge"],additional_data,edge_weight=data["weight"], prob=True)
        return pred
