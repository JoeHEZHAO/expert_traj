import torch
import numpy as np

from soft_dtw_cuda import *
from sklearn.cluster import KMeans

# from kmeans_pytorch import kmeans


def expert_find(
    data, num_of_obs, expert_set_train, expert_set_val, weighted=False, gamma=2.0
):

    """
    weighted: weather or not weight the input sequence for comparing DWT measurements;
    """
    global mse, criterion
    all_min_end = []
    all_min_end_indices = []
    rest_diff = []

    mse = torch.nn.MSELoss()
    criterion = SoftDTW(use_cuda=True, gamma=gamma, normalize=True)

    for i in range(num_of_obs):
        tmp_traj_norm = data.obs_traj_norm[:, :, i].permute(0, 2, 1)  # [1, 2, 8]
        tmp_traj_v = data.velocity_obs[:, :, i].permute(0, 2, 1)  # [1, 2, 8]

        """ replicate all test data and then do loss"""
        dset_train_num = (
            expert_set_train.obs_traj_norm.shape[0]
            + expert_set_val.obs_traj_norm.shape[0]
        )
        tmp_traj_norm = tmp_traj_norm.repeat(dset_train_num, 1, 1)
        tmp_traj_v = tmp_traj_v.repeat(dset_train_num, 1, 1)

        """ Apply mask that focus on more recent trajectory among 8 coords """
        # seq_weight = torch.range(0.8, 1.6, 0.2)
        traj_weight = torch.range(1, 2.875, step=0.25).cuda()
        traj_weight = traj_weight.flip(0)
        traj_weight = traj_weight.reshape(1, 8, 1).repeat(dset_train_num, 1, 2)

        loss = criterion(
            tmp_traj_norm.permute(0, 2, 1),
            torch.cat(
                [
                    expert_set_train.obs_traj_norm.cuda(),
                    expert_set_val.obs_traj_norm.cuda(),
                ],
                0,
            ).permute(0, 2, 1),
        )
        loss_v = criterion(
            # tmp_traj_v.permute(0, 2, 1) * traj_weight,
            tmp_traj_v.permute(0, 2, 1),
            torch.cat(
                [expert_set_train.obs_traj_v.cuda(), expert_set_val.obs_traj_v.cuda()],
                0,
            ).permute(0, 2, 1)
            # * traj_weight,
        )

        # min_k, min_k_indices = torch.topk(loss, 50, largest=False)
        # min_k_v, min_k_v_indices = torch.topk(loss_v, 50, largest=False)
        # min_k, min_k_indices = min_k.tolist(), min_k_indices.tolist()
        # min_k_v, min_k_v_indices = min_k_v.tolist(), min_k_v_indices.tolist()

        # collection of train and vald
        col_pred_traj = torch.cat(
            [expert_set_train.pred_traj.cuda(), expert_set_val.pred_traj.cuda()], 0
        )
        # col_obs_traj_norm = torch.cat(
        # [
        # expert_set_train.obs_traj_norm.cuda(),
        # expert_set_val.obs_traj_norm.cuda(),
        # ],
        # 0,
        # )

        """Try the clustering fashion"""
        min_k, min_k_indices = torch.topk(loss_v, 100, largest=False)
        # retrieved_expert = expert_set[min_k_indices][:, -1]
        retrieved_expert = col_pred_traj[min_k_indices][:, :, -1]

        # kmeans

        kmeans = KMeans(n_clusters=20, random_state=0).fit(
            retrieved_expert.cpu().numpy()
        )

        iter_target = min_k_indices

        """ Find the common between them? """
        """
        Choose use either coords or velocity as selecting criterion;
        """
        # iter_target = min_k_v_indices[9:]
        # iter_target = min_k_v_indices
        # iter_target = min_k_indices

        min_k_end = []
        end_point_appr = []
        # for k in iter_target:
        for k in kmeans.cluster_centers_:

            # Calculate the absolute end point estimation;
            test_end = data.pred_traj_gt[:, -1, i].cuda()

            exp_end = torch.from_numpy(k).cuda()

            min_k_end.append(torch.norm(exp_end - test_end, 2))

            end_point_appr.append(exp_end)

        all_min_end.append(min(min_k_end))
        print("Min loss of end point estimation is {}".format(all_min_end[-1]))
        # all_min_end_indices.append(min_k_end.index(min(min_k_end)))
        rest_diff.append(end_point_appr[min_k_end.index(min(min_k_end))])

    return all_min_end, rest_diff


if __name__ == "__main__":
    pass
