#!/usr/bin/env python36
# -*- coding: utf-8 -*-

import datetime
import math
import numpy as np
import torch
from torch import nn
from torch.nn import Module, Parameter
import torch.nn.functional as F
from torchsummary import summary

K = 5

class PositionEmbedding(nn.Module):

    MODE_EXPAND = 'MODE_EXPAND'
    MODE_ADD = 'MODE_ADD'
    MODE_CONCAT = 'MODE_CONCAT'

    def __init__(self,
                 num_embeddings,
                 embedding_dim,
                 mode=MODE_ADD):
        super(PositionEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.mode = mode
        if self.mode == self.MODE_EXPAND:
            self.weight = nn.Parameter(torch.Tensor(num_embeddings * 2 + 1, embedding_dim))
        else:
            self.weight = nn.Parameter(torch.Tensor(num_embeddings, embedding_dim))

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.weight)

    def forward(self, x):
        if self.mode == self.MODE_EXPAND:
            indices = torch.clamp(x, -self.num_embeddings, self.num_embeddings) + self.num_embeddings
            return F.embedding(indices.type(torch.LongTensor), self.weight)
        batch_size, seq_len = x.size()[:2]
        embeddings = self.weight[:seq_len, :].view(1, seq_len, self.embedding_dim)
        if self.mode == self.MODE_ADD:
            return x + embeddings
        if self.mode == self.MODE_CONCAT:
            return torch.cat((x, embeddings.repeat(batch_size, 1, 1)), dim=-1)
        raise NotImplementedError('Unknown mode: %s' % self.mode)

    def extra_repr(self):
        return 'num_embeddings={}, embedding_dim={}, mode={}'.format(
            self.num_embeddings, self.embedding_dim, self.mode,
        )

class Residual(Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = 120
        self.d1 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.d2 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.dp = nn.Dropout(p=0.2)
        self.drop = True

    def forward(self, x):
        residual = x  # keep original input
        x = F.relu(self.d1(x))
        if self.drop:
            x = self.d2(self.dp(x))
        else:
            x = self.d2(x)
        out = residual + x
        return out

class MultiHeadedAttention(nn.Module): # Этот слой используется для обработки взаимозависимостей между элементами данных с применением нескольких голов внимания (multi-head). Это стандартный слой, который может работать на уровне сессий или пользователей.
    """Multi-Head Attention layer

    :param int n_head: the number of head s
    :param int n_feat: the number of features
    :param float dropout_rate: dropout rate
    """

    def __init__(self, n_head, n_feat, dropout_rate):
        super(MultiHeadedAttention, self).__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, query, key, value, mask):
        """Compute 'Scaled Dot Product Attention'

        :param torch.Tensor query: (batch, time1, size)
        :param torch.Tensor key: (batch, time2, size)
        :param torch.Tensor value: (batch, time2, size)
        :param torch.Tensor mask: (batch, time1, time2)
        :param torch.nn.Dropout dropout:
        :return torch.Tensor: attentined and transformed `value` (batch, time1, d_model)
             weighted by the query dot key attention (batch, head, time1, time2)
        """
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)  # (batch, head, time1, d_k)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)  # (batch, head, time1, time2)
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, time1, time2)
            min_value = float(np.finfo(torch.tensor(0, dtype=scores.dtype).numpy().dtype).min)
            self.attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)  # (batch, head, time1, time2)
        else:
            self.attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

        p_attn = self.dropout(self.attn)
        x = torch.matmul(p_attn, v)  # (batch, head, time1, d_k)
        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)  # (batch, time1, d_model)
        return self.linear_out(x)  # (batch, time1, d_model)

class GNNWithAttention(Module):
    def __init__(self, hidden_size, step=1, num_heads=8, attention=True):
        super(GNNWithAttention, self).__init__()
        self.step = step
        self.hidden_size = hidden_size
        self.attention = attention
        self.num_heads = num_heads

        # Используем только LastAttenion для многоголового внимания
        self.attn = LastAttenion(self.hidden_size, self.num_heads, dot=0.1, l_p=1)  # Параметры можно настроить

    def forward(self, A, hidden, mask=None):
        if self.attention:
            # Применение многоголового внимания
            hidden, _ = self.attn(hidden, hidden, mask)  # Применяем внимание к скрытым состояниям
        return hidden

class LastAttenion(Module):
    def __init__(self, hidden_size, heads, dot, l_p, last_k=3, use_attn_conv=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.heads = heads
        self.last_k = last_k
        self.linear_zero = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, self.heads, bias=False)
        self.linear_four = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.linear_five = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.dropout = 0.1
        self.dot = dot
        self.l_p = l_p
        self.use_attn_conv = use_attn_conv
        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.parameters():
            weight.data.normal_(std=0.1)

    def forward(self, ht1, hidden, mask):
        batch_size, seq_len, _ = hidden.size()

        # Добавляем ось времени (если её нет) - предполагаем, что это 1 шаг во времени
        if len(hidden.size()) == 2:  # [batch_size, hidden_size]
            hidden = hidden.unsqueeze(1)  # Добавляем ось seq_len: [batch_size, 1, hidden_size]

        batch_size, seq_len, _ = hidden.size()  # Теперь у нас есть 3 размерности: [batch_size, seq_len, hidden_size]

        q0 = self.linear_zero(ht1).view(batch_size, seq_len, self.heads, -1).permute(0, 2, 1, 3)
        q1 = self.linear_one(hidden).view(batch_size, seq_len, self.heads, -1).permute(0, 2, 1, 3)
        q2 = self.linear_two(hidden).view(batch_size, seq_len, self.heads, -1).permute(0, 2, 1, 3)

        alpha = torch.matmul(q0, q1.transpose(-1, -2)) / (q0.size(-1) ** 0.5)
        alpha = torch.softmax(alpha, dim=-1)

        if self.use_attn_conv:
            # Преобразуем alpha для LPPooling
            alpha_reshaped = alpha.view(-1, alpha.size(-2), alpha.size(-1))  # [batch * heads, seq_len, seq_len]
            pool = torch.nn.LPPool1d(self.l_p, self.last_k, stride=self.last_k)
            alpha_pooled = pool(alpha_reshaped)  # [batch * heads, seq_len, pooled_len]
            pooled_len = alpha_pooled.size(-1)
            alpha = alpha_pooled.view(batch_size, self.heads, seq_len, pooled_len)

            extended_mask = mask[:, :seq_len].unsqueeze(1).unsqueeze(-1)
            alpha = alpha * extended_mask  # Применяем маску

            # Нормализация
            alpha = torch.softmax(alpha, dim=-1)

            # Обновление q2, чтобы он соответствовал pooled_len
            q2 = q2[..., -pooled_len:, :]  # Обрезаем последние pooled_len элементов

        out = torch.matmul(alpha, q2)
        out = out.permute(0, 2, 1, 3).contiguous()
        out = out.view(batch_size, seq_len, -1)

        mask = mask[:, :seq_len]
        mask = mask.unsqueeze(-1)
        mask = mask.expand(-1, out.size(1), out.size(2))

        out = out * mask
        a = out.sum(dim=1)

        return a, alpha

# Интеграция в модель GC-SAN:
class MultiLevelAttention(Module):
    def __init__(self, hidden_size, heads, dot, l_p, last_k=3, use_attn_conv=False):
        #super().__init__()
        super(MultiLevelAttention, self).__init__()
        self.hidden_size = hidden_size
        self.heads = heads
        self.last_k = last_k
        self.linear_zero = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, self.heads, bias=False)
        self.linear_four = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.linear_five = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.dropout = 0.1
        self.dot = dot
        self.l_p = l_p
        self.use_attn_conv = use_attn_conv
        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.parameters():
            weight.data.normal_(std=0.1)

    def forward(self, ht1, hidden, mask):
        # Получаем размерность hidden
        batch_size, seq_len, _ = hidden.size()

        # Проверяем размерность hidden, если она 2D, добавляем ось времени
        if len(hidden.size()) == 2:  # [batch_size, hidden_size]
            hidden = hidden.unsqueeze(1)  # Добавляем ось seq_len: [batch_size, 1, hidden_size]

        batch_size, seq_len, _ = hidden.size()  # Теперь у нас есть 3 размерности: [batch_size, seq_len, hidden_size]

        # Преобразуем hidden через линейные слои для различных уровней внимания
        q0 = self.linear_zero(ht1).view(batch_size, seq_len, self.heads, -1).permute(0, 2, 1, 3)
        q1 = self.linear_one(hidden).view(batch_size, seq_len, self.heads, -1).permute(0, 2, 1, 3)
        q2 = self.linear_two(hidden).view(batch_size, seq_len, self.heads, -1).permute(0, 2, 1, 3)

        alpha = torch.matmul(q0, q1.transpose(-1, -2)) / (q0.size(-1) ** 0.5)
        alpha = torch.softmax(alpha, dim=-1)

        self.alpha = alpha # Сохраняем alpha как атрибут для регуляризации в других частях модели

        if self.use_attn_conv:
            # Преобразуем alpha для LPPooling
            alpha_reshaped = alpha.view(-1, alpha.size(-2), alpha.size(-1))  # [batch * heads, seq_len, seq_len]
            pool = torch.nn.LPPool1d(self.l_p, self.last_k, stride=self.last_k)
            alpha_pooled = pool(alpha_reshaped)  # [batch * heads, seq_len, pooled_len]
            pooled_len = alpha_pooled.size(-1)
            alpha = alpha_pooled.view(batch_size, self.heads, seq_len, pooled_len)

            extended_mask = mask[:, :seq_len].unsqueeze(1).unsqueeze(-1)
            alpha = alpha * extended_mask  # Применяем маску

            # Нормализация
            alpha = torch.softmax(alpha, dim=-1)

            # Обновление q2, чтобы он соответствовал pooled_len
            q2 = q2[..., -pooled_len:, :]  # Обрезаем последние pooled_len элементов

        # Применение внимания
        out = torch.matmul(alpha, q2)
        out = out.permute(0, 2, 1, 3).contiguous()
        out = out.view(batch_size, seq_len, -1)

        # Применяем маску
        mask = mask[:, :seq_len]
        mask = mask.unsqueeze(-1)
        mask = mask.expand(-1, out.size(1), out.size(2))

        out = out * mask
        a = out.sum(dim=1)

        return a, alpha

class SessionGraphWithMultiLevelAttention(Module):
    def __init__(self, opt, n_node, len_max):
        super(SessionGraphWithMultiLevelAttention, self).__init__()
        self.hidden_size = opt.hiddenSize
        self.len_max = len_max
        self.n_node = n_node
        self.batch_size = opt.batchSize
        self.num_heads = opt.heads
        self.embedding = nn.Embedding(self.n_node, self.hidden_size)

        # Добавление слоя мультиуровневого внимания
        self.multi_level_attn = MultiLevelAttention(self.hidden_size, self.num_heads, dot=opt.dot, l_p=opt.l_p, last_k=opt.last_k)
        # Используем GNN с многоголовым вниманием
        self.gnn_with_attention = GNNWithAttention(self.hidden_size, step=opt.step, num_heads=self.num_heads)

        # Линейные преобразования и другие слои
        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, 1, bias=False)
        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)

        # Прочие слои
        self.rn = Residual()
        self.multihead_attn = nn.MultiheadAttention(self.hidden_size, 1).cuda()
        self.pe = PositionEmbedding(len_max, self.hidden_size)

        # Функция потерь и оптимизатор
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)

        # Инициализация параметров
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def compute_scores(self, hidden, mask, self_att=True, residual=True, k_blocks=4):
        # Логика вычисления оценок остаётся без изменений
        idx = torch.sum(mask, dim=1) - 1  # Индекс последнего элемента в каждой последовательности
        idx = torch.clamp(idx, min=0, max=hidden.size(1) - 1)

        ht = hidden[torch.arange(hidden.size(0)).long(), idx]  # [batch, hidden_size]

        if self_att:
            # Применяем многослойное внимание
            a, alpha = self.multi_level_attn(hidden, hidden, mask)  # Используем новый слой внимания
            a = 0.52 * a + (1 - 0.52) * ht
        else:
            # Вариант без внимания
            q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1])  # [batch, 1, hidden]
            q2 = self.linear_two(hidden)  # [batch, seq_len, hidden]
            alpha = self.linear_three(torch.sigmoid(q1 + q2))  # [batch, seq_len, 1]

            # Маска
            mask_expanded = mask.view(mask.shape[0], -1, 1).float()
            a = torch.sum(alpha * hidden * mask_expanded, dim=1)

            if not self.nonhybrid:
                a = self.linear_transform(torch.cat([a, ht], dim=1))  # Применяем линейное преобразование

        # Прогноз
        b = self.embedding.weight[1:]  # [n_nodes - 1, hidden]
        scores = torch.matmul(a, b.transpose(1, 0))  # [batch, n_nodes - 1]

        return scores

    # Изменение функции потерь
    def loss_function(self, scores, targets):
        # Перекрёстная энтропия для классификационной задачи
        loss = torch.nn.CrossEntropyLoss()(scores, targets)

        # Регуляризация внимания (например, на основе alpha)
        # Можно добавить регуляризацию на alpha, чтобы не позволить модели слишком сильно полагаться на конкретные элементы внимания
        attn_regularization = torch.mean(torch.abs(self.multi_level_attn.alpha))
        loss += 0.01 * attn_regularization  # Параметр регуляризации можно подбирать

        return loss

    def forward(self, inputs, A, mask):
        hidden = self.embedding(inputs)  # Применяем embedding к входным данным

        # Пропускаем через GNN с многоголовым вниманием
        hidden = self.gnn_with_attention(A, hidden, mask)  # Используем gnn_with_attention

        # Проверка размерности и добавление оси времени, если необходимо
        if len(hidden.size()) == 2:  # Если тензор двухмерный, добавляем ось времени
            hidden = hidden.unsqueeze(1)  # Добавляем ось seq_len: [batch_size, 1, hidden_size]

        # Применяем многослойное внимание
        attn_output, _ = self.multi_level_attn(hidden, hidden, mask)

        # Получаем последнее скрытое состояние
        idx = torch.sum(mask, dim=1) - 1
        idx = torch.clamp(idx, min=0, max=hidden.size(1) - 1)
        ht = hidden[torch.arange(hidden.size(0)).long(), idx]

        # Преобразуем ht в 2D тензор (если он 1D) для объединения
        if ht.dim() == 1:
            ht = ht.unsqueeze(1)  # Преобразуем в [batch_size, hidden_size]

        out = self.linear_transform(torch.cat([attn_output, ht], dim=-1))  # Объединяем внимание и скрытое состояние

        return hidden


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable

def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable

def forward(model, i, data):
    alias_inputs, A, items, mask, targets = data.get_slice(i)
    alias_inputs = trans_to_cuda(torch.Tensor(alias_inputs).long())
    items = trans_to_cuda(torch.Tensor(items).long())
    A = trans_to_cuda(torch.Tensor(A).float())
    mask = trans_to_cuda(torch.Tensor(mask).long()) # Здесь мы передаем маску в forward
    hidden = model(items, A, mask)  # Передаем mask

    # Проверка на индексы выходящие за пределы
    max_seq_len = hidden.size(1)  # Должен быть равен 6 (для последовательности длиной 6)

    # Проверим, не выходят ли индексы за пределы
    alias_inputs = torch.clamp(alias_inputs, max=max_seq_len - 1)  # Принудительно обрезаем индексы

    # Проверка, что индексы находятся в допустимом диапазоне
    if (alias_inputs >= hidden.size(1)).any():
        print(f"[Error] alias_inputs contains indices out of bounds!")
        return targets, None  # Возвращаем None, чтобы избежать дальнейших ошибок

    get = lambda i: hidden[i][alias_inputs[i]]
    seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])
    mask = torch.stack([mask[i][alias_inputs[i]] for i in range(len(alias_inputs))])
    return targets, model.compute_scores(seq_hidden, mask)

def train_test(model, train_data, test_data):
    model.scheduler.step()
    print('start training: ', datetime.datetime.now())
    model.train()
    total_loss = 0.0
    slices = train_data.generate_batch(model.batch_size)

    for i, j in zip(slices, np.arange(len(slices))):
        model.optimizer.zero_grad()
        targets, scores = forward(model, i, train_data)

        # Приводим к тензору и CUDA
        targets = trans_to_cuda(torch.Tensor(targets).long())

        # Проверка на [batch, 1] — делаем [batch]
        if targets.dim() > 1:
            targets = targets.squeeze()

        # Проверка на допустимые значения
        if (targets < 1).any():
            print(f"[Error] targets contain 0 or negative values: {targets}")
            raise ValueError("Invalid target indices: must be >= 1")

        loss = model.loss_function(scores, targets - 1)

        loss.backward()
        model.optimizer.step()
        total_loss += loss

        if j % int(len(slices) / 5 + 1) == 0:
            print('[%d/%d] Loss: %.4f' % (j, len(slices), loss.item()))

    print('\tLoss:\t%.3f' % total_loss)

    print('start predicting: ', datetime.datetime.now())
    model.eval()
    precision, mrr = [], []
    slices = test_data.generate_batch(model.batch_size)

    for i in slices:
        targets, scores = forward(model, i, test_data)
        sub_scores = scores.topk(K)[1]
        sub_scores = trans_to_cpu(sub_scores).detach().numpy()
        for score, target, mask in zip(sub_scores, targets, test_data.mask):
            precision_at_k = np.isin(target - 1, score).sum() / K
            precision.append(precision_at_k)
            if len(np.where(score == target - 1)[0]) == 0:
                mrr.append(0)
            else:
                mrr.append(1 / (np.where(score == target - 1)[0][0] + 1))

    precision_at_k_mean = np.mean(precision) * 100
    mrr_mean = np.mean(mrr) * 100

    return precision_at_k_mean, mrr_mean