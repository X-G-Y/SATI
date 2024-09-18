import numpy as np
import random

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from transformers import BertModel, BertConfig
from typing import Optional
from utils import to_gpu
from utils import ReverseLayerF
import warnings
from transformers import RobertaTokenizer, RobertaModel

import math
import torch.nn.functional as F
import torch.nn as disable_weight_init

class AdversarialDiscriminator(nn.Module):
    def __init__(self, feature_dim):
        super(AdversarialDiscriminator, self).__init__()
        self.fc1 = nn.Linear(feature_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.sigmoid(self.fc3(x))

class RNNPoolingModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(RNNPoolingModel, self).__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=False)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: [frame, batch, embedding]
        _, (h_n, _) = self.rnn(x)  # h_n: [1, batch, hidden_dim]
        x = h_n.squeeze(0)  # [batch, hidden_dim]
        x = self.fc(x)  # [batch, output_dim]
        return x
    
class MaskedSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MaskedSelfAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)  

    def forward(self, x, attn_mask=None):
        attn_output, _ = self.multihead_attn(x, x, x, attn_mask=attn_mask)
        return attn_output
    
    
def masked_mean(tensor, mask, dim):
    """Finding the mean along dim"""
    masked = torch.mul(tensor, mask)
    return masked.sum(dim=dim) / mask.sum(dim=dim)

def masked_max(tensor, mask, dim):
    """Finding the max along dim"""
    masked = torch.mul(tensor, mask)
    neg_inf = torch.zeros_like(tensor)
    neg_inf[~mask] = -math.inf
    return (masked + neg_inf).max(dim=dim)


class CrossAttention(nn.Module):
    def __init__(self, dim_q, dim_kv, heads=8):
        super().__init__()
        self.dim_q = dim_q
        self.heads = heads
        self.wq = nn.Linear(dim_q, dim_q, bias=False)
        self.wk = nn.Linear(dim_kv, dim_q, bias=False)
        self.wv = nn.Linear(dim_kv, dim_q, bias=False)
        self.out_proj = nn.Linear(dim_q, dim_q)
        self.dropout = nn.Dropout(0.2)
        self.act = nn.ReLU
    def multihead_reshape(self, x):
        b, lens, dim = x.shape
        x = x.reshape(b, lens, self.heads, dim // self.heads)
        x = x.transpose(1, 2)
        x = x.reshape(b * self.heads, lens, dim // self.heads)
        return x

    def multihead_reshape_inverse(self, x):
        b, lens, dim = x.shape
        x = x.reshape(b // self.heads, self.heads, lens, dim)
        x = x.transpose(1, 2)
        x = x.reshape(b // self.heads, lens, dim * self.heads)
        return x
    def forward(self, q, kv):
        q = self.wq(q)
        k = self.wk(kv)
        v = self.wv(kv)

        q = self.multihead_reshape(q)
        k = self.multihead_reshape(k)
        v = self.multihead_reshape(v)

        atten = q.bmm(k.transpose(1, 2)) * (self.dim_q // self.heads)**-0.5
        atten = atten.softmax(dim=-1)
        atten = atten.bmm(v)
        
        atten = self.multihead_reshape_inverse(atten)
        atten = self.out_proj(atten)
        atten = self.dropout(atten)
        return atten
    
class FBP(nn.Module):
    def __init__(self, d_emb_1, d_emb_2, fbp_hid, fbp_k, dropout):
        super(FBP, self).__init__()
        self.fusion_1_matrix = nn.Linear(d_emb_1, fbp_hid*fbp_k, bias=False)
        self.fusion_2_matrix = nn.Linear(d_emb_2, fbp_hid*fbp_k, bias=False)
        self.fusion_dropout = nn.Dropout(dropout)
        self.fusion_pooling = nn.AvgPool1d(kernel_size=fbp_k)
        self.fbp_k = fbp_k

    def forward(self, seq1, seq2):
        seq1 = self.fusion_1_matrix(seq1)
        seq2 = self.fusion_2_matrix(seq2)
        fused_feature = torch.mul(seq1, seq2)
        if len(fused_feature.shape) == 2:
            fused_feature = fused_feature.unsqueeze(0)
        fused_feature = self.fusion_dropout(fused_feature)
        fused_feature = self.fusion_pooling(fused_feature).squeeze(0) * self.fbp_k # (bs, 512)
        fused_feature = F.normalize(fused_feature, dim=-1, p=2)
        return fused_feature
    
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # Add batch dimension
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]
class SelfGating(nn.Module):
    def __init__(self, input_dim):
        super(SelfGating, self).__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid(),
            nn.Dropout(0.6)
        )
    
    def forward(self, x):
        gate_value = self.gate(x)
        return x * gate_value
def compute_frame_attention(self, features_m1, features_m2):
    S_g_list = []
    for frame in range(features_m1.size(0)):
        Q = features_m1[frame]  #  [batch, ebd]
        K = features_m2[frame]  #  [batch, ebd]
        elementwise_mul = Q * K  #  [batch, ebd]

        # SumPooling
        sum_pooled = torch.sum(elementwise_mul, dim=1)  #  [batch]

        # L2 Normalization
        l2_normalized = F.normalize(sum_pooled, p=2, dim=1)  #  [batch]

        # Linear Projection
        S_g = torch.matmul(l2_normalized)  # [batch]
        
        S_g_list.append(S_g)
    S_g_final = torch.stack(S_g_list, dim=0)  #  [frame, batch]
    return S_g_final

class MyRNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(MyRNNModel, self).__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(0.6)

    def forward(self, x):
        x = x.permute(1, 0, 2)
        _, (hn, _) = self.rnn(x)
        hn = hn[-1] 
        return hn
    
# let's define a simple model that can deal with multimodal variable length sequence
class SATI(nn.Module):
    def __init__(self, config):
        super(SATI, self).__init__()

        self.config = config
        self.text_size = config.embedding_size
        self.visual_size = config.visual_size
        self.acoustic_size = config.acoustic_size
        self.input_sizes = input_sizes = [self.text_size, self.visual_size, self.acoustic_size]
        self.hidden_sizes = hidden_sizes = [int(self.text_size), int(self.visual_size), int(self.acoustic_size)]
        self.output_size = output_size = config.num_classes
        self.dropout_rate = dropout_rate = config.dropout
        self.activation = self.config.activation()
        self.softmax = nn.Softmax()
        self.tanh = nn.Tanh()
        self.domain_private = []
        self.domain_shared = []
        
        rnn = nn.LSTM if self.config.rnncell == "lstm" else nn.GRU
        # defining modules - two layer bidirectional LSTM with layer norm in between

        vocab_file = '/home/s22xjq/SATI/model/vocab.json'
        merges_file = '/home/s22xjq/SATI/model/merges.txt'
        print("using roberta")
        self.robertatokenizer = RobertaTokenizer(vocab_file, merges_file)
        self.roberta = RobertaModel.from_pretrained('/home/s22xjq/SATI/model/roberta-base/')

        
        self.vrnn1 = rnn(input_sizes[1], hidden_sizes[1], bidirectional=True)
        self.vrnn2 = rnn(2*hidden_sizes[1], hidden_sizes[1], bidirectional=True)
        
        self.arnn1 = rnn(input_sizes[2], hidden_sizes[2], bidirectional=True)
        self.arnn2 = rnn(2*hidden_sizes[2], hidden_sizes[2], bidirectional=True)

        self.poolingrnn = MyRNNModel(self.config.hidden_size, self.config.hidden_size, self.output_size)

        ##########################################
        # mapping modalities to same sized space
        ##########################################

        self.project_t = nn.Sequential()
        self.project_t.add_module('project_t', nn.Linear(in_features=768, out_features=config.hidden_size))
        self.project_t.add_module('project_t_activation', self.activation)
        self.project_t.add_module('project_t_layer_norm', nn.LayerNorm(config.hidden_size))

        self.project_v = nn.Sequential()
        self.project_v.add_module('project_v', nn.Linear(in_features=hidden_sizes[1], out_features=config.hidden_size))
        self.project_v.add_module('project_v_activation', self.activation)
        self.project_v.add_module('project_v_layer_norm', nn.LayerNorm(config.hidden_size))

        self.project_a = nn.Sequential()
        self.project_a.add_module('project_a', nn.Linear(in_features=hidden_sizes[2], out_features=config.hidden_size))
        self.project_a.add_module('project_a_activation', self.activation)
        self.project_a.add_module('project_a_layer_norm', nn.LayerNorm(config.hidden_size))

        self.project_h = nn.Sequential()
        self.project_h.add_module('project_h', nn.Linear(in_features=hidden_sizes[2]*4, out_features=config.hidden_size))
        self.project_h.add_module('project_h_activation', self.activation)
        self.project_h.add_module('project_h_layer_norm', nn.LayerNorm(config.hidden_size))

        ##########################################
        # private encoders
        ##########################################
        self.private_t = nn.Sequential()
        self.private_t.add_module('private_t_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.private_t.add_module('private_t_activation_1', nn.Sigmoid())
        
        self.private_v = nn.Sequential()
        self.private_v.add_module('private_v_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.private_v.add_module('private_v_activation_1', nn.Sigmoid())
        
        self.private_a = nn.Sequential()
        self.private_a.add_module('private_a_3', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.private_a.add_module('private_a_activation_3', nn.Sigmoid())
        

        ##########################################
        # shared encoder
        ##########################################
        self.shared = nn.Sequential()
        self.shared.add_module('shared_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.shared.add_module('shared_activation_1', nn.Sigmoid())


        ##########################################
        # reconstruct
        ##########################################
        self.recon_t = nn.Sequential()
        self.recon_t.add_module('recon_t_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.recon_v = nn.Sequential()
        self.recon_v.add_module('recon_v_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.recon_a = nn.Sequential()
        self.recon_a.add_module('recon_a_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))



        ##########################################
        # shared space adversarial discriminator
        ##########################################

        self.discriminator = nn.Sequential()
        self.discriminator.add_module('discriminator_layer_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.discriminator.add_module('discriminator_layer_1_activation', self.activation)
        self.discriminator.add_module('discriminator_layer_1_dropout', nn.Dropout(dropout_rate))
        
        self.W = nn.Sequential()
        self.W.add_module('discriminator_layer_2', nn.Linear(in_features=config.hidden_size, out_features=3, bias=False))
        #self.discriminator.add_module('discriminator_layer_2_activation', self.softmax)
        ##########################################
        # shared-private collaborative discriminator
        ##########################################

        self.sp_discriminator = nn.Sequential()
        self.sp_discriminator.add_module('sp_discriminator_layer_1', nn.Linear(in_features=config.hidden_size, out_features=4))



        self.fusion = nn.Sequential()
        self.fusion.add_module('fusion_layer_1', nn.Linear(in_features=self.config.hidden_size*2, out_features=self.config.hidden_size*1))
        self.fusion.add_module('fusion_layer_1_dropout', nn.Dropout(dropout_rate))
        self.fusion.add_module('fusion_layer_1_activation', self.activation)
        self.fusion.add_module('fusion_layer_3', nn.Linear(in_features=self.config.hidden_size*1, out_features= output_size))

        self.my_fusion = nn.Sequential()
        self.my_fusion.add_module('fusion_layer_2', nn.Linear(in_features=self.config.hidden_size*2, out_features= output_size))
        #self.my_fusion.add_module('fusion_layer_2_dropout', nn.Dropout(0.2))

        
        
        self.tlayer_norm = nn.LayerNorm((hidden_sizes[0]*2,))
        self.vlayer_norm = nn.LayerNorm((hidden_sizes[1]*2,))
        self.alayer_norm = nn.LayerNorm((hidden_sizes[2]*2,))
        self.hlayer_norm = nn.LayerNorm(self.config.hidden_size)
        self.player_norm = nn.LayerNorm(self.config.hidden_size)
        self.slayer_norm = nn.LayerNorm(self.config.hidden_size)


        encoder_layer = nn.TransformerEncoderLayer(d_model=self.config.hidden_size*2, nhead=1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        encoder_layer_v = nn.TransformerEncoderLayer(d_model=hidden_sizes[1], nhead=1)
        self.transformer_encoder_v = nn.TransformerEncoder(encoder_layer_v, num_layers=1)    
        
        encoder_layer_a = nn.TransformerEncoderLayer(d_model=hidden_sizes[2], nhead=2)
        self.transformer_encoder_a = nn.TransformerEncoder(encoder_layer_a, num_layers=1)  
        
        
        self.cross_attn_at = CrossAttention(dim_q=self.config.hidden_size, dim_kv=self.config.hidden_size, heads=4)
        self.norm_atten1 = nn.LayerNorm(normalized_shape=self.config.hidden_size, elementwise_affine=True)
        self.cross_attn_vt = CrossAttention(dim_q=self.config.hidden_size, dim_kv=self.config.hidden_size, heads=4)
        self.norm_atten2 = nn.LayerNorm(normalized_shape=self.config.hidden_size, elementwise_affine=True)
        
        #self.cross_encoder_av = TransformerEncoder(self.config.hidden_size, self.config.hidden_size, n_layers=4, d_inner=512, n_head=4, d_k=None, d_v=None, dropout=0.1, n_position=self.config.hidden_size, add_sa=False)
        #self.cross_encoder_at = TransformerEncoder(self.config.hidden_size, self.config.hidden_size, n_layers=4, d_inner=512, n_head=4, d_k=None, d_v=None, dropout=0.1, n_position=self.config.hidden_size, add_sa=False)
        
        
        ##########################################
        # PositionalEncoder
        ##########################################
        self.position_encoding = PositionalEncoding(d_model=self.config.hidden_size, max_len=1024)
        
        self.masked_attn_layer = MaskedSelfAttention(embed_dim = self.config.hidden_size, num_heads=1)
        
        self.pooling = RNNPoolingModel(input_dim=3*self.config.hidden_size, hidden_dim=self.config.hidden_size*3, output_dim=self.config.hidden_size*3)
        
        
        ##########################################
        # FBP Gate
        ##########################################    
        self.fbp_at = FBP(self.config.hidden_size, self.config.hidden_size, fbp_hid=32, fbp_k=2, dropout=0.4)
        self.fc_gate_at = nn.Linear(32, 1)


        
        self.fbp_vt = FBP(self.config.hidden_size, self.config.hidden_size, fbp_hid=32, fbp_k=2, dropout=0.4)
        self.fc_gate_vt = nn.Linear(32, 1)
        self.gate_activate = nn.Tanh()
        
        ##########################################
        # self Gating
        ##########################################        
        #self.self_gating = SelfGating(self.config.hidden_size)

    def extract_features(self, sequence, lengths, rnn1, rnn2, layer_norm):
        lengths = lengths.cpu().long()
        packed_sequence = pack_padded_sequence(sequence, lengths)

        if self.config.rnncell == "lstm":
            packed_h1, (final_h1, _) = rnn1(packed_sequence)
        else:
            packed_h1, final_h1 = rnn1(packed_sequence)

        padded_h1, _ = pad_packed_sequence(packed_h1)
        normed_h1 = layer_norm(padded_h1)
        packed_normed_h1 = pack_padded_sequence(normed_h1, lengths)

        if self.config.rnncell == "lstm":
            packed_h2, (final_h2, _) = rnn2(packed_normed_h1)
        else:
            packed_h2, final_h2 = rnn2(packed_normed_h1)
        padded_h2, _ = pad_packed_sequence(packed_h2)
        return final_h1, final_h2, padded_h1, padded_h2

    def alignment(self, sentences, visual, acoustic, lengths, bert_sent, bert_sent_type, bert_sent_mask):
        
        batch_size = lengths.size(0)
        
        ##########################################
        # extract avt features
        ##########################################
        

        roberta_output = self.roberta(input_ids=bert_sent, 
                                attention_mask=bert_sent_mask)
        roberta_out = roberta_output.last_hidden_state  
        roberta_out = roberta_out.transpose(0,1)  # [46, 64, 768]
        utterance_text = roberta_out




        
        # extract features from visual modality
        utterance_video = self.transformer_encoder_v(visual)
        utterance_audio = self.transformer_encoder_a(acoustic)       
        #print( utterance_audio.shape)



        # Shared-private encoders
        self.shared_private(utterance_text, utterance_video, utterance_audio)
        
        ##########################################
        # discriminator
        ##########################################
        private_t = torch.sum(self.utt_private_t, dim=0)/self.utt_private_t.size(0)
        shared_t = torch.sum(self.utt_shared_t, dim=0)/self.utt_shared_t.size(0)
        private_a = torch.sum(self.utt_private_a, dim=0)/self.utt_private_a.size(0)
        shared_a = torch.sum(self.utt_shared_a, dim=0)/self.utt_shared_a.size(0)
        private_v = torch.sum(self.utt_private_v, dim=0)/self.utt_private_v.size(0)
        shared_v = torch.sum(self.utt_shared_v, dim=0)/self.utt_shared_v.size(0)
        shared = (shared_v+shared_t+shared_a)/3
        private = (private_v+private_t+private_a)/3
        private = self.player_norm(private)
        shared = self.slayer_norm(shared)
        self.domain_private = private
        self.domain_shared = shared
        #print(private)

        reversed_shared_code_t = ReverseLayerF.apply(shared_t, self.config.reverse_grad_weight)
        reversed_shared_code_v = ReverseLayerF.apply(shared_v, self.config.reverse_grad_weight)
        reversed_shared_code_a = ReverseLayerF.apply(shared_a, self.config.reverse_grad_weight)
        reversed_private_code_t = ReverseLayerF.apply(private_t, self.config.reverse_grad_weight)
        reversed_private_code_v = ReverseLayerF.apply(private_v, self.config.reverse_grad_weight)
        reversed_private_code_a = ReverseLayerF.apply(private_a, self.config.reverse_grad_weight)
        self.domain_shared_t = self.discriminator(reversed_shared_code_t)
        self.domain_shared_v = self.discriminator(reversed_shared_code_v)
        self.domain_shared_a = self.discriminator(reversed_shared_code_a)
        self.domain_private_t = self.discriminator(reversed_private_code_t)
        self.domain_private_v = self.discriminator(reversed_private_code_v)
        self.domain_private_a = self.discriminator(reversed_private_code_a)       
        
        # For reconstruction
        self.reconstruct()
          
        A = self.position_encoding(self.utt_private_a)
        V = self.position_encoding(self.utt_private_v)
        T = self.position_encoding(self.utt_private_t)
        

        at = self.cross_attn_at(T.transpose(0, 1), A.transpose(0, 1))
        at = at.transpose(0, 1)
        gate_ = self.fbp_at(self.utt_shared_a, self.utt_shared_t)
        gate_ = self.gate_activate(self.fc_gate_at(gate_))#.double()
        gate_sign = gate_ / torch.abs(gate_)
        gate_ = (gate_sign + torch.abs(gate_sign)) / 2.0
        at = at*gate_ +  T
        at = self.norm_atten1(at)
        
        
        vt = self.cross_attn_vt(T.transpose(0, 1), V.transpose(0, 1)) 
        vt = vt.transpose(0, 1)
        gate_ = self.fbp_vt(self.utt_shared_v, self.utt_shared_t)
        gate_ = self.gate_activate(self.fc_gate_at(gate_))#.double()
        gate_sign = gate_ / torch.abs(gate_)
        gate_ = (gate_sign + torch.abs(gate_sign)) / 2.0
        vt = vt*gate_ + T
        vt = self.norm_atten2(vt)
        h = torch.cat((at , vt), dim=2)
        h = self.transformer_encoder(h)
        h = torch.sum(h, dim=0)/h.size(0)
        o = self.fusion(h)
        #print(h)
        return o
    
    def reconstruct(self,):

        self.utt_t = (self.utt_private_t + self.utt_shared_t)
        self.utt_v = (self.utt_private_v + self.utt_shared_v)
        self.utt_a = (self.utt_private_a + self.utt_shared_a)
        self.utt_t = torch.sum(self.utt_t, dim=0)/self.utt_t.size(0)
        self.utt_v = torch.sum(self.utt_v, dim=0)/self.utt_v.size(0)
        self.utt_a = torch.sum(self.utt_a, dim=0)/self.utt_a.size(0)
        self.utt_t_recon = self.recon_t(self.utt_t)
        self.utt_v_recon = self.recon_v(self.utt_v)
        self.utt_a_recon = self.recon_a(self.utt_a)


    def shared_private(self, utterance_t, utterance_v, utterance_a):
        
        ##########################################
        # for recon_loss
        ##########################################3
        # Projecting to same sized space
        self.utt_t_orig = utterance_t = self.project_t(utterance_t)
        self.utt_v_orig = utterance_v = self.project_v(utterance_v)
        self.utt_a_orig = utterance_a = self.project_a(utterance_a)


        # Private-shared components
        self.utt_private_t = self.private_t(utterance_t)
        self.utt_private_v = self.private_v(utterance_v)
        self.utt_private_a = self.private_a(utterance_a) #[framec, batch, hidden_Size]


        self.utt_shared_t = self.shared(utterance_t)
        self.utt_shared_v = self.shared(utterance_v)
        self.utt_shared_a = self.shared(utterance_a)


    def forward(self, sentences, video, acoustic, lengths, bert_sent, bert_sent_type, bert_sent_mask):
        batch_size = lengths.size(0)
        o = self.alignment(sentences, video, acoustic, lengths, bert_sent, bert_sent_type, bert_sent_mask)
        return o 
