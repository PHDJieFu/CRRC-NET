from functools import reduce
import torch
import torch.nn as nn
import utils
import pickle
import torch.nn.functional as F
from fvcore.nn import FlopCountAnalysis, parameter_count_table


class Total_loss(nn.Module):
    def __init__(self, lambdas):
        super(Total_loss, self).__init__()
        self.tau = 0.1
        self.sampling_size = 3
        self.lambdas = lambdas
        self.ce_criterion = nn.BCELoss(reduction='none')

    def topkloss(self, element_logits, labels, is_back=True, lab_rand=None, rat=8, reduce=None):
        if is_back:
            labels_with_back = torch.cat((labels, torch.ones_like(labels[:, [0]])), dim=-1)
        else:
            labels_with_back = torch.cat((labels, torch.zeros_like(labels[:, [0]])), dim=-1)
        if lab_rand is not None:
            labels_with_back = torch.cat((labels, lab_rand), dim=-1)
        topk_val, topk_ind = torch.topk(
            element_logits,
            k = max(1, int(element_logits.shape[-2]) // rat),
            dim=1
        )
        instance_logits = torch.mean(topk_val, dim=1)
        labels_with_back = labels_with_back / (torch.sum(labels_with_back, dim=1, keepdim=True) + 1e-4)
        milloss = (-(labels_with_back * F.log_softmax(instance_logits, dim=-1)).sum(dim=1))
        if reduce is not None:
            milloss = milloss.mean()
        return milloss, topk_ind

    def _multiply(self, x, atn, dim=-1, include_min=False):
        if include_min:
            _min = x.min(dim=dim, keepdim=True)[0]
        else:
            _min = 0
        return atn * (x - _min) + _min

    def multi_nce_loss(self, logits, mask):
        loss = - torch.log( (F.softmax(logits, dim=1) * mask).sum(1) )
        return loss.mean()

    def calculate_self_supervised_loss(self, logits_list, labels_list):
        loss_self_supervised = 0
        action_loss_num = len(logits_list)
        for i in range(action_loss_num):
            action_i_rgb_loss = self.multi_nce_loss(logits_list[i][0], labels_list[i][0])
            action_i_flow_loss = self.multi_nce_loss(logits_list[i][1], labels_list[i][1])
            loss_self_supervised += action_i_rgb_loss
            loss_self_supervised += action_i_flow_loss
        loss_self_supervised /= action_loss_num
        return loss_self_supervised

    def forward(self, vid_score, cas_sigmoid_fuse, features, stored_info, label, point_anno, 
                step, x_atn, rgb_atn, flow_atn, cas_sigmoid, logits_list, labels_list):
        loss = {}

        loss_self_supervised = self.calculate_self_supervised_loss(logits_list, labels_list)

        loss_mutual = 0.5 * F.mse_loss(rgb_atn, flow_atn.detach()) + 0.5 * F.mse_loss(flow_atn, rgb_atn.detach())

        element_logits_supp = self._multiply(cas_sigmoid, x_atn, include_min=True)
        loss_mil_orig, _ = self.topkloss(cas_sigmoid, label, is_back=True, rat=8, reduce=None)
        loss_mil_supp, _ = self.topkloss(element_logits_supp, label, is_back=False, rat=8, reduce=None)

        loss_norm = (x_atn.mean() + rgb_atn.mean() + flow_atn.mean()) / 3

        loss_guide = ((1 - x_atn - cas_sigmoid.softmax(-1)[..., [-1]]).abs().mean() + \
                      (1 - rgb_atn - cas_sigmoid.softmax(-1)[..., [-1]]).abs().mean() + \
                      (1 - flow_atn - cas_sigmoid.softmax(-1)[..., [-1]]).abs().mean()) / 3

        loss_vid = self.ce_criterion(vid_score, label)
        loss_vid = loss_vid.mean()
        
        point_anno = torch.cat((point_anno, torch.zeros((point_anno.shape[0], point_anno.shape[1], 1)).cuda()), dim=2)
        
        weighting_seq_act = point_anno.max(dim=2, keepdim=True)[0]
        num_actions = point_anno.max(dim=2)[0].sum(dim=1)

        focal_weight_act = (1 - cas_sigmoid_fuse) * point_anno + cas_sigmoid_fuse * (1 - point_anno)
        focal_weight_act = focal_weight_act ** 2

        loss_frame = (((focal_weight_act * self.ce_criterion(cas_sigmoid_fuse, point_anno) * weighting_seq_act).sum(dim=2)).sum(dim=1) / num_actions).mean()

        _, bkg_seed = utils.select_seed(cas_sigmoid_fuse.detach().cpu(), point_anno.detach().cpu())
            
        bkg_seed = bkg_seed.unsqueeze(-1).cuda()

        point_anno_bkg = torch.zeros_like(point_anno).cuda()
        point_anno_bkg[:,:,-1] = 1

        weighting_seq_bkg = bkg_seed
        num_bkg = bkg_seed.sum(dim=1)

        focal_weight_bkg = (1 - cas_sigmoid_fuse) * point_anno_bkg + cas_sigmoid_fuse * (1 - point_anno_bkg)
        focal_weight_bkg = focal_weight_bkg ** 2

        loss_frame_bkg = (((focal_weight_bkg * self.ce_criterion(cas_sigmoid_fuse, point_anno_bkg) * weighting_seq_bkg).sum(dim=2)).sum(dim=1) / num_bkg).mean()
        
        loss_score_act = 0
        loss_score_bkg = 0
        loss_feat = 0

        if len(stored_info['new_dense_anno'].shape) > 1:
            new_dense_anno = stored_info['new_dense_anno'].cuda()
            new_dense_anno = torch.cat((new_dense_anno, torch.zeros((new_dense_anno.shape[0], new_dense_anno.shape[1], 1)).cuda()), dim=2)
                    
            act_idx_diff = new_dense_anno[:,1:] - new_dense_anno[:,:-1]
            loss_score_act = 0
            loss_feat = 0
            for b in range(new_dense_anno.shape[0]):
                gt_classes = torch.nonzero(label[b]).squeeze(1)
                act_count = 0
                loss_score_act_batch = 0
                loss_feat_batch = 0

                for c in gt_classes:
                    range_idx = torch.nonzero(act_idx_diff[b,:,c]).squeeze(1)
                    range_idx = range_idx.cpu().data.numpy().tolist()
                    if type(range_idx) is not list:
                        range_idx = [range_idx]
                    if len(range_idx) == 0:
                        continue
                    if act_idx_diff[b, range_idx[0], c] != 1:
                        range_idx = [-1] + range_idx 
                    if act_idx_diff[b, range_idx[-1], c] != -1:
                        range_idx = range_idx + [act_idx_diff.shape[1] - 1]
                        
                    label_lst = []
                    feature_lst = []

                    if range_idx[0] > -1:
                        start_bkg = 0
                        end_bkg = range_idx[0]
                        bkg_len = end_bkg - start_bkg + 1

                        label_lst.append(0)
                        feature_lst.append(utils.feature_sampling(features[b], start_bkg, end_bkg + 1, self.sampling_size))

                    for i in range(len(range_idx) // 2):
                        if range_idx[2*i + 1] - range_idx[2*i] < 1:
                            continue

                        label_lst.append(1)
                        feature_lst.append(utils.feature_sampling(features[b], range_idx[2*i] + 1, range_idx[2*i + 1] + 1, self.sampling_size))

                        if range_idx[2*i + 1] != act_idx_diff.shape[1] - 1:
                            start_bkg = range_idx[2*i + 1] + 1

                            if i == (len(range_idx) // 2 - 1):
                                end_bkg = act_idx_diff.shape[1] - 1
                            else:
                                end_bkg = range_idx[2*i + 2]

                            bkg_len = end_bkg - start_bkg + 1

                            label_lst.append(0)
                            feature_lst.append(utils.feature_sampling(features[b], start_bkg, end_bkg + 1, self.sampling_size))

                        start_act = range_idx[2*i] + 1
                        end_act = range_idx[2*i + 1]

                        complete_score_act = utils.get_oic_score(cas_sigmoid_fuse[b,:,c], start=start_act, end=end_act)
                        
                        loss_score_act_batch += 1 - complete_score_act

                        act_count += 1

                    if sum(label_lst) > 1:
                        feature_lst = torch.stack(feature_lst, 0).clone()
                        feature_lst = feature_lst / torch.norm(feature_lst, dim=1, p=2).unsqueeze(1)
                        label_lst = torch.tensor(label_lst).cuda().float()

                        sim_matrix = torch.matmul(feature_lst, torch.transpose(feature_lst, 0, 1)) / self.tau

                        sim_matrix = torch.exp(sim_matrix)
                        
                        sim_matrix = sim_matrix.clone().fill_diagonal_(0)

                        scores = (sim_matrix * label_lst.unsqueeze(1)).sum(dim=0) / sim_matrix.sum(dim=0)

                        loss_feat_batch = (-label_lst * torch.log(scores)).sum() / label_lst.sum()

                if act_count > 0:
                    loss_score_act += loss_score_act_batch / act_count
                    loss_feat += loss_feat_batch

                
            bkg_idx_diff = (1 - new_dense_anno[:,1:]) - (1 - new_dense_anno[:,:-1])
            loss_score_bkg = 0
            for b in range(new_dense_anno.shape[0]):
                gt_classes = torch.nonzero(label[b]).squeeze(1)
                loss_score_bkg_batch = 0
                bkg_count = 0

                for c in gt_classes:
                    range_idx = torch.nonzero(bkg_idx_diff[b,:,c]).squeeze(1)
                    range_idx = range_idx.cpu().data.numpy().tolist()
                    if type(range_idx) is not list:
                        range_idx = [range_idx]
                    if len(range_idx) == 0:
                        continue
                    if bkg_idx_diff[b, range_idx[0], c] != 1:
                        range_idx = [-1] + range_idx 
                    if bkg_idx_diff[b, range_idx[-1], c] != -1:
                        range_idx = range_idx + [bkg_idx_diff.shape[1] - 1]

                    for i in range(len(range_idx) // 2):
                        if range_idx[2*i + 1] - range_idx[2*i] < 1:
                            continue
                        
                        start_bkg = range_idx[2*i] + 1
                        end_bkg = range_idx[2*i + 1]

                        complete_score_bkg = utils.get_oic_score(1 - cas_sigmoid_fuse[b,:,c], start=start_bkg, end=end_bkg)
                        
                        loss_score_bkg_batch += 1 - complete_score_bkg

                        bkg_count += 1

                if bkg_count > 0:
                    loss_score_bkg += loss_score_bkg_batch / bkg_count
                    
            loss_score_act = loss_score_act / new_dense_anno.shape[0]
            loss_score_bkg = loss_score_bkg / new_dense_anno.shape[0]
            
            loss_feat = loss_feat / new_dense_anno.shape[0]

        loss_score = (loss_score_act + loss_score_bkg) ** 2

        loss_total = self.lambdas[0] * loss_vid + self.lambdas[1] * loss_frame + self.lambdas[2] * loss_frame_bkg + \
                     self.lambdas[3] * loss_score + self.lambdas[4] * loss_feat + \
                     loss_mil_orig.mean() + loss_mil_supp.mean() + loss_mutual + 0.8 * loss_guide + 0.8 * loss_norm + loss_self_supervised

        loss["loss_mil_orig"] = loss_mil_orig.mean()
        loss["loss_mil_supp"] = loss_mil_supp.mean()
        loss["loss_mutual"] = loss_mutual
        loss["loss_guide"] = loss_guide
        loss["loss_norm"] = loss_norm
        loss["loss_self_supervised"] = loss_self_supervised
        
        loss["loss_vid"] = loss_vid
        loss["loss_frame"] = loss_frame
        loss["loss_frame_bkg"] = loss_frame_bkg
        loss["loss_score_act"] = loss_score_act
        loss["loss_score_bkg"] = loss_score_bkg
        loss["loss_score"] = loss_score
        loss["loss_feat"] = loss_feat
        loss["loss_total"] = loss_total

        return loss_total, loss


def train(net, config, loader_iter, optimizer, criterion, logger, step, proto_feat):
    net.train()

    torch.autograd.set_detect_anomaly(True)

    total_loss = {}
    total_cost = []
    optimizer.zero_grad()
    batch_pseudo_action_feat_list = []
    batch_pseudo_action_snippet_num_list = []


    for _b in range(config.batch_size):
        _, _data, _label, _point_anno, stored_info, _, _ = next(loader_iter)

        _data = _data.cuda()
        _label = _label.cuda()
        _point_anno = _point_anno.cuda()

        vid_score, cas_sigmoid_fuse, features, pseudo_label, class_weighted_sum_feat, class_snippet_num,\
                                    x_atn, rgb_atn, flow_atn, cas_sigmoid, logits_list, labels_list = net(_data, proto_feat, _point_anno, _label)    # 

        batch_pseudo_action_feat_list.append(class_weighted_sum_feat)
        batch_pseudo_action_snippet_num_list.append(class_snippet_num)

        pseudo_action_label = _point_anno + pseudo_label
        zeros = torch.zeros_like(pseudo_action_label)
        ones = torch.ones_like(pseudo_action_label)
        pseudo_action_label = torch.where(pseudo_action_label > 0, ones, zeros)

        cost, loss = criterion(vid_score, cas_sigmoid_fuse, features, stored_info, _label, pseudo_action_label, step, \
                                x_atn, rgb_atn, flow_atn, cas_sigmoid, logits_list, labels_list)
        total_cost.append(cost)

        for key in loss.keys():
            if not (key in total_loss):
                total_loss[key] = []

            if loss[key] > 0:
                total_loss[key] += [loss[key].detach().cpu().item()]
            else:
                total_loss[key] += [loss[key]]
    
    batch_pseudo_action_feat_tensor = torch.cat(batch_pseudo_action_feat_list, dim=0)
    batch_pseudo_action_snippet_num_tensor = torch.cat(batch_pseudo_action_snippet_num_list, dim=0)
    batch_pseudo_action_feat = torch.sum(batch_pseudo_action_feat_tensor, dim=0)
    batch_pseudo_action_snippet_num = torch.sum(batch_pseudo_action_snippet_num_tensor, dim=0).unsqueeze(-1)
    ones = torch.ones_like(batch_pseudo_action_snippet_num)
    batch_pseudo_action_snippet_num = torch.where(batch_pseudo_action_snippet_num==0, ones, batch_pseudo_action_snippet_num)
    batch_pseudo_action_feat = batch_pseudo_action_feat / batch_pseudo_action_snippet_num
    batch_pseudo_action_feat = batch_pseudo_action_feat.unsqueeze(0)

    total_cost = sum(total_cost) / config.batch_size
    total_cost.backward()
    optimizer.step()

    for key in total_loss.keys():
        logger.log_value("loss/" + key, sum(total_loss[key]) / config.batch_size, step)

    return batch_pseudo_action_feat