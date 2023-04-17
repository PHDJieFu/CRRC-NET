from numpy.lib.function_base import _parse_input_dimensions
import torch
from torch import random
import torch.nn as nn
import pickle
import torch.nn.functional as F
from torch.nn.modules import padding
import random

from torch.nn.modules.activation import LeakyReLU


class Cls_Module(nn.Module):
    def __init__(self, len_feature, num_classes):
        super(Cls_Module, self).__init__()
        self.len_feature = len_feature
        self.conv_1 = nn.Sequential(
            nn.Conv1d(in_channels=self.len_feature, out_channels=2048, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2)
        )
        self.classifier = nn.Sequential(
            nn.Conv1d(2048, 2048, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.7),
            nn.Conv1d(2048, num_classes+1, 1)
        )
        self.drop_out = nn.Dropout(p=0.7)


    def forward(self, x, proto_feat):
        proto_feat = proto_feat.permute(0, 2, 1)
        proto_feat = self.conv_1(proto_feat)
        proto_feat = proto_feat.permute(0, 2, 1).squeeze(0)
        out = self.conv_1(x)
        feat = out.permute(0, 2, 1)
        out = self.drop_out(out)
        cas = self.classifier(out)

        cas = cas.permute(0, 2, 1)
        return feat, cas, proto_feat

class BWA_fusion_dropout_feat_v2(nn.Module):
    def __init__(self, n_feature):
        super().__init__()
        embed_dim = 1024
        self.bit_wise_attn = nn.Sequential(
            nn.Conv1d(n_feature, embed_dim, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5)
        )
        self.channel_conv = nn.Sequential(
            nn.Conv1d(n_feature, embed_dim, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5)
        )
        self.channel_avg = nn.AdaptiveAvgPool1d(1)
    
    def forward(self, vfeat, ffeat):
        channelfeat = self.channel_avg(vfeat)
        channel_attn = self.channel_conv(channelfeat)
        bit_wise_attn = self.bit_wise_attn(ffeat)
        filter_feat = torch.sigmoid(bit_wise_attn * channel_attn) * vfeat
        return filter_feat

class Model(nn.Module):
    def __init__(self, len_feature, num_classes, r_act):
        super(Model, self).__init__()
        self.len_feature = len_feature
        self.num_classes = num_classes
        self.r_act = r_act

        self.temperature = 1.0
        self.T = 0.07
        self.all_video_feat_path = 'all_vid_feat_dict.pckl'
        self.class_cor_neg_vidname_dict_path = 'class_cor_neg_vidname_dict.pckl'
        self.class_point_anno_feat_path = 'class_point_anno_feat_dict.pckl'

        self.rgb_feat_encoder = nn.Sequential(
            nn.Conv1d(1024, 1024, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
        )

        self.flow_feat_encoder = nn.Sequential(
            nn.Conv1d(1024, 1024, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
        )

        self.rgb_Attn = BWA_fusion_dropout_feat_v2(1024)
        self.flow_Attn = BWA_fusion_dropout_feat_v2(1024)
        self.attention_rgb = nn.Sequential(
            nn.Conv1d(1024, 512, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Conv1d(512, 512, 3, padding=1),
            nn.LeakyReLU(0.2), 
            nn.Conv1d(512, 1, 1),
            nn.Dropout(0.5),
            nn.Sigmoid()
        )

        self.attention_flow = nn.Sequential(
            nn.Conv1d(1024, 512, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Conv1d(512, 512, 3, padding=1),
            nn.LeakyReLU(0.2), 
            nn.Conv1d(512, 1, 1),
            nn.Dropout(0.5),
            nn.Sigmoid()
        )

        self.cls_module = Cls_Module(len_feature, num_classes)
        self.sigmoid = nn.Sigmoid()
        self.lambda_value = 0.9999

        self.all_vid_frame_feat_dict = self.get_all_video_feat(self.all_video_feat_path)
        self.class_cor_neg_vidname_dict = self.get_class_cor_neg_vidname(self.class_cor_neg_vidname_dict_path)
        self.class_point_feat_dict = self.get_class_point_feat_dict(self.class_point_anno_feat_path)

    def get_all_video_feat(self, all_video_feat_path):
        all_video_feat_file = open(all_video_feat_path, 'rb')
        all_vid_frame_feat_dict = pickle.load(all_video_feat_file)
        all_video_feat_file.close()
        return all_vid_frame_feat_dict

    def get_class_cor_neg_vidname(self, class_cor_neg_vidname_dict_path):
        class_cor_neg_vidname_dict_file = open(class_cor_neg_vidname_dict_path, 'rb')
        class_cor_neg_vidname_dict = pickle.load(class_cor_neg_vidname_dict_file)
        class_cor_neg_vidname_dict_file.close()
        return class_cor_neg_vidname_dict

    def get_class_point_feat_dict(self, class_point_anno_feat_path):
        class_point_anno_feat_dict_file = open(class_point_anno_feat_path, 'rb')
        class_point_anno_feat_dict = pickle.load(class_point_anno_feat_dict_file)
        class_point_anno_feat_dict_file.close()
        return class_point_anno_feat_dict

    def cal_similarity_between_feature(self, anno_feat, all_temporal_feat):
        B, t, feat_dim = anno_feat.shape[0], anno_feat.shape[1], anno_feat.shape[2]
        anno_feat = anno_feat.view(-1, feat_dim)
        anno_feat = F.normalize(anno_feat, dim=-1)
        all_temporal_feat = all_temporal_feat.view(-1, feat_dim)
        all_temporal_feat = F.normalize(all_temporal_feat, dim=-1)
        similarity = torch.matmul(anno_feat, all_temporal_feat.T)
        similarity = torch.sum(similarity, dim=0)
        return similarity

    def get_class_neg_frame_feat(self, vid_class_idx, class_cor_neg_vidname_dict, all_vid_frame_feat_dict):
        sample_num = 1000
        class_cor_neg_vidnames_list = class_cor_neg_vidname_dict[vid_class_idx.item()]
        class_neg_frame_rgb_feat_list = []
        class_neg_frame_flow_feat_list = []
        for i in range(sample_num):
            vid_name_i = random.choice(class_cor_neg_vidnames_list)
            vid_feat_i_rgb = all_vid_frame_feat_dict[vid_name_i][0]

            vid_feat_i_flow = all_vid_frame_feat_dict[vid_name_i][1]
            temporal_len = vid_feat_i_rgb.shape[0]
            index = torch.LongTensor(random.sample(range(temporal_len), 1))
            sample_frame_rgb_feat = torch.index_select(vid_feat_i_rgb, dim=0, index=index)
            sample_frame_flow_feat = torch.index_select(vid_feat_i_flow, dim=0, index=index)
            class_neg_frame_rgb_feat_list.append(sample_frame_rgb_feat)
            class_neg_frame_flow_feat_list.append(sample_frame_flow_feat)
        class_neg_frame_rgb_feat = torch.cat(class_neg_frame_rgb_feat_list, dim=0)
        class_neg_frame_rgb_feat = class_neg_frame_rgb_feat.unsqueeze(0)
        class_neg_frame_flow_feat = torch.cat(class_neg_frame_flow_feat_list, dim=0)
        class_neg_frame_flow_feat = class_neg_frame_flow_feat.unsqueeze(0)
        return class_neg_frame_rgb_feat.permute(0, 2, 1), class_neg_frame_flow_feat.permute(0, 2, 1)

    def get_class_pos_frame_feat(self, vid_class_idx, class_point_feat_dict):
        sample_num = 30
        class_all_point_rgb_feat = class_point_feat_dict[vid_class_idx.item()][0]
        class_all_point_flow_feat = class_point_feat_dict[vid_class_idx.item()][1]
        class_pos_anno_point_num = class_all_point_rgb_feat.shape[0]
        index = torch.LongTensor(random.sample(range(class_pos_anno_point_num), sample_num))
        class_sample_pos_point_rgb_feat = torch.index_select(class_all_point_rgb_feat, dim=0, index=index)
        class_sample_pos_point_rgb_feat = class_sample_pos_point_rgb_feat.unsqueeze(0)
        class_sample_pos_point_flow_feat = torch.index_select(class_all_point_flow_feat, dim=0, index=index)
        class_sample_pos_point_flow_feat = class_sample_pos_point_flow_feat.unsqueeze(0)
        return class_sample_pos_point_rgb_feat.permute(0, 2, 1), class_sample_pos_point_flow_feat.permute(0, 2, 1)

    def forward(self, x, proto_feat, _point_anno=None, vid_labels=None):
        num_segments = x.shape[1]
        k_act = num_segments // self.r_act

        single_modality_feat_dim = int(x.shape[2] / 2)
        x_rgb = x[:, :, :single_modality_feat_dim]
        x_rgb = x_rgb.permute(0, 2, 1)
        x_flow = x[:, :, single_modality_feat_dim:]
        x_flow = x_flow.permute(0, 2, 1)

        x_rgb = self.rgb_feat_encoder(x_rgb)
        x_flow = self.flow_feat_encoder(x_flow)
        
        if vid_labels != None and _point_anno != None:
            vid_class_idxs = torch.nonzero(vid_labels)[:, 1]
            vid_class_num = vid_class_idxs.shape[0]
            logits_list = []
            labels_list = []
            for i in range(vid_class_num):
                vid_class_idx = vid_class_idxs[i]
                vid_class_anno_T = _point_anno[:, :, vid_class_idx]
                vid_class_anno_frame_idx = torch.nonzero(vid_class_anno_T)[:, 1]

                class_oth_pos_frame_rgb_feat, class_oth_pos_frame_flow_feat = self.get_class_pos_frame_feat(vid_class_idx, self.class_point_feat_dict)
                class_oth_pos_frame_rgb_feat = class_oth_pos_frame_rgb_feat.cuda()
                class_oth_pos_frame_flow_feat = class_oth_pos_frame_flow_feat.cuda()

                class_oth_pos_frame_rgb_feat = self.rgb_feat_encoder(class_oth_pos_frame_rgb_feat)
                class_oth_pos_frame_flow_feat = self.flow_feat_encoder(class_oth_pos_frame_flow_feat)

                class_oth_neg_frame_rgb_feat, class_oth_neg_frame_flow_feat = self.get_class_neg_frame_feat(vid_class_idx, self.class_cor_neg_vidname_dict, self.all_vid_frame_feat_dict)
                class_oth_neg_frame_rgb_feat = class_oth_neg_frame_rgb_feat.cuda()
                class_oth_neg_frame_flow_feat = class_oth_neg_frame_flow_feat.cuda()
                class_oth_neg_frame_rgb_feat = self.rgb_feat_encoder(class_oth_neg_frame_rgb_feat)
                class_oth_neg_frame_flow_feat = self.flow_feat_encoder(class_oth_neg_frame_flow_feat)

                vid_class_anno_frame_flow_feat = x_flow[:, :, vid_class_anno_frame_idx]
                vid_class_anno_frame_flow_feat = vid_class_anno_frame_flow_feat.permute(0, 2, 1)
                
                flow_temporal_simi = self.cal_similarity_between_feature(vid_class_anno_frame_flow_feat, x_flow.permute(0, 2, 1))
                _, flow_pos_frame_idxs = torch.topk(flow_temporal_simi, k=num_segments//self.r_act)

                class_vid_pos_frame_rgb_feat = x_rgb[:,:,flow_pos_frame_idxs]
                class_vid_pos_frame_rgb_feat = torch.cat((class_vid_pos_frame_rgb_feat, class_oth_pos_frame_rgb_feat), dim=-1)
                class_vid_pos_frame_rgb_feat = F.normalize(class_vid_pos_frame_rgb_feat, dim=1)
                class_vid_pos_frame_rgb_feat = class_vid_pos_frame_rgb_feat.squeeze(0).permute(1, 0)
                flow_neg_frame_idxs = torch.sort(flow_temporal_simi)[1][:num_segments//6]
                
                class_vid_neg_frame_rgb_feat = x_rgb[:,:,flow_neg_frame_idxs]
                l_pos_rgb = torch.einsum('nc,ck->nk', [class_vid_pos_frame_rgb_feat, class_vid_pos_frame_rgb_feat.T])
                self_mask_rgb = torch.eye(l_pos_rgb.shape[0], dtype=torch.bool).to(l_pos_rgb.device)
                l_pos_rgb = l_pos_rgb[~self_mask_rgb].view(l_pos_rgb.shape[0], -1)

                class_all_neg_frame_rgb_feat = torch.cat((class_vid_neg_frame_rgb_feat, class_oth_neg_frame_rgb_feat), dim=-1)
                class_all_neg_frame_rgb_feat = class_all_neg_frame_rgb_feat.squeeze(0).permute(1, 0)
                class_all_neg_frame_rgb_feat = F.normalize(class_all_neg_frame_rgb_feat, dim=-1)

                l_neg_rgb = torch.einsum('nc,ck->nk', [class_vid_pos_frame_rgb_feat, class_all_neg_frame_rgb_feat.T])
                logits_rgb = torch.cat((l_pos_rgb, l_neg_rgb), dim=-1)
                logits_rgb /= self.T
                pos_labels_rgb = torch.ones_like(l_pos_rgb)
                neg_labels_rgb = torch.zeros_like(l_neg_rgb)
                labels_rgb = torch.cat((pos_labels_rgb, neg_labels_rgb), dim=-1)

                vid_class_anno_frame_rgb_feat = x_rgb[:, :, vid_class_anno_frame_idx]
                vid_class_anno_frame_rgb_feat = vid_class_anno_frame_rgb_feat.permute(0, 2, 1)
                rgb_temporal_simi = self.cal_similarity_between_feature(vid_class_anno_frame_rgb_feat, x_rgb.permute(0, 2, 1))
                _, rgb_pos_frame_idxs = torch.topk(rgb_temporal_simi, k=num_segments//self.r_act)
                class_vid_pos_frame_flow_feat = x_flow[:,:,rgb_pos_frame_idxs]
                class_vid_pos_frame_flow_feat = torch.cat((class_vid_pos_frame_flow_feat, class_oth_pos_frame_flow_feat), dim=-1)
                class_vid_pos_frame_flow_feat = F.normalize(class_vid_pos_frame_flow_feat, dim=1)
                class_vid_pos_frame_flow_feat = class_vid_pos_frame_flow_feat.squeeze(0).permute(1, 0)
                rgb_neg_frame_idxs = torch.sort(rgb_temporal_simi)[1][:num_segments//6]
                class_vid_neg_frame_flow_feat = x_flow[:,:,rgb_neg_frame_idxs]
                l_pos_flow = torch.einsum('nc,ck->nk', [class_vid_pos_frame_flow_feat, class_vid_pos_frame_flow_feat.T])
                self_mask_flow = torch.eye(l_pos_flow.shape[0], dtype=torch.bool).to(l_pos_flow.device)
                l_pos_flow = l_pos_flow[~self_mask_flow].view(l_pos_flow.shape[0], -1)

                class_all_neg_frame_flow_feat = torch.cat((class_vid_neg_frame_flow_feat, class_oth_neg_frame_flow_feat), dim=-1)
                class_all_neg_frame_flow_feat = class_all_neg_frame_flow_feat.squeeze(0).permute(1, 0)
                class_all_neg_frame_flow_feat = F.normalize(class_all_neg_frame_flow_feat, dim=-1)
                l_neg_flow = torch.einsum('nc,ck->nk', [class_vid_pos_frame_flow_feat, class_all_neg_frame_flow_feat.T])
                logits_flow = torch.cat((l_pos_flow, l_neg_flow), dim=-1)
                logits_flow /= self.T
                pos_labels_flow = torch.ones_like(l_pos_flow)
                neg_labels_flow = torch.zeros_like(l_neg_flow)
                labels_flow = torch.cat((pos_labels_flow, neg_labels_flow), dim=-1)
                
                logits_list.append((logits_rgb, logits_flow))
                labels_list.append((labels_rgb.detach(), labels_flow.detach()))
        
        rgb_enhanced_feat = self.rgb_Attn(x_rgb, x_flow)
        rgb_atn = self.attention_rgb(rgb_enhanced_feat) 
        flow_enhanced_feat = self.flow_Attn(x_flow, x_rgb)
        flow_atn = self.attention_flow(flow_enhanced_feat)

        x_atn = (rgb_atn + flow_atn) / 2
        x_atn = x_atn.permute(0, 2, 1)
        enhanced_feat = torch.cat((rgb_enhanced_feat, flow_enhanced_feat), dim=1)

        features, cas, proto_feat = self.cls_module(enhanced_feat, proto_feat)
        cas_sigmoid = self.sigmoid(cas)
        if vid_labels != None:
            current_feature = features.permute(0, 2, 1)
            feat_proto_distance = -torch.ones((features.size()[0], self.num_classes, features.size()[1])).cuda()
            for i in range(self.num_classes):
                feat_proto_distance[:, i, :] = torch.norm(proto_feat[i].reshape(-1, 1).expand(-1, current_feature.size()[2]) - current_feature, 2, dim=1)
            
            feat_nearest_proto_distance, _ = feat_proto_distance.min(dim=1, keepdim=True)
            feat_proto_distance = feat_proto_distance - feat_nearest_proto_distance
            weight = F.softmax(-feat_proto_distance * self.temperature, dim=1)
            weight = weight.permute(0, 2, 1)
            weighted_cas = weight * cas_sigmoid[:, :, :-1]

            video_action_label = vid_labels.unsqueeze(1).expand(-1, weighted_cas.size()[1], -1)
            pseudo_label_cas = weighted_cas * video_action_label

            # construct pseudo label based on predictions
            ones = torch.ones_like(pseudo_label_cas)
            zeros = torch.zeros_like(pseudo_label_cas)
            pseudo_label = torch.where(pseudo_label_cas > 0.7, ones, zeros)

            pseudo_label_ = pseudo_label.permute(0, 2, 1)
            class_weighted_sum_feat = torch.matmul(pseudo_label_, features)
            class_snippet_num = torch.sum(pseudo_label_, dim=2)

        cas_sigmoid_fuse = cas_sigmoid[:,:,:-1] * (1 - cas_sigmoid[:,:,-1].unsqueeze(2))
        cas_sigmoid_fuse = torch.cat((cas_sigmoid_fuse, cas_sigmoid[:,:,-1].unsqueeze(2)), dim=-1)
        value, _ = cas_sigmoid.sort(descending=True, dim=1)
        topk_scores = value[:,:k_act,:-1]

        if vid_labels is None:
            vid_score = torch.mean(topk_scores, dim=1)
        else:
            vid_score = (torch.mean(topk_scores, dim=1) * vid_labels) + (torch.mean(cas_sigmoid[:,:,:-1], dim=1) * (1 - vid_labels))

        if vid_labels != None and _point_anno != None:
            return vid_score, cas_sigmoid_fuse, features, pseudo_label, class_weighted_sum_feat, class_snippet_num,\
                   x_atn, rgb_atn.permute(0, 2, 1), flow_atn.permute(0, 2, 1), cas_sigmoid, logits_list, labels_list
        elif vid_labels != None and _point_anno == None:
            return vid_score, cas_sigmoid_fuse, features, pseudo_label, class_weighted_sum_feat, class_snippet_num,\
                   x_atn, rgb_atn.permute(0, 2, 1), flow_atn.permute(0, 2, 1), cas_sigmoid, None, None
        else:
            return vid_score, cas_sigmoid_fuse, features