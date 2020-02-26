import torch

EPS = 1e-7


def sample_point(cps, p_num, alpha=0.5):
    cp_num = cps.size(1)
    p_num = int(p_num / cp_num)

    # Suppose cps is [n_batch, n_cp, 2]
    cps = torch.cat([cps, cps[:, 0, :].unsqueeze(1)], dim=1)
    auxillary_cps = torch.zeros(cps.size(0), cps.size(1) + 2, cps.size(2)).to(cps.device)
    auxillary_cps[:, 1:-1, :] = cps

    l_01 = torch.sqrt(torch.sum(torch.pow(cps[:, 0, :] - cps[:, 1, :], 2), dim=1) + EPS)
    l_last_01 = torch.sqrt(torch.sum(torch.pow(cps[:, -1, :] - cps[:, -2, :], 2), dim=1) + EPS)

    l_01.detach_().unsqueeze_(1)
    l_last_01.detach_().unsqueeze_(1)

    # print(l_last_01, l_01)

    auxillary_cps[:, 0, :] = cps[:, 0, :] - l_01 / l_last_01 * (cps[:, -1, :] - cps[:, -2, :])
    auxillary_cps[:, -1, :] = cps[:, -1, :] + l_last_01 / l_01 * (cps[:, 1, :] - cps[:, 0, :])

    # print(auxillary_cps)

    t = torch.zeros([auxillary_cps.size(0), auxillary_cps.size(1)]).to(cps.device)
    t[:, 1:] = torch.pow(torch.sum(torch.pow(auxillary_cps[:, 1:, :] - auxillary_cps[:, :-1, :], 2), dim=2), alpha/2)
    t = torch.cumsum(t, dim=1)

    # No need to calculate gradient w.r.t t.
    t = t.detach()
    points = torch.zeros([cps.size(0), p_num * cp_num, cps.size(2)]).to(cps.device)
    temp_step = torch.arange(p_num).float().to(cps.device)
    temp_step_len = (t[:, 2:-1] - t[:, 1:-2]) / (p_num - 1)
    v = torch.matmul(temp_step_len.unsqueeze(2), temp_step.unsqueeze(0).repeat([cps.size(0), 1, 1])).reshape([cps.size(0), -1])
    v = torch.matmul(t[:, 1:-2].unsqueeze(2), torch.ones(cps.size(0), 1, p_num).to(cps.device)).reshape([cps.size(0), -1]) + v
    # vuse = v.clone()
    t0 = t[:, 0:-3].unsqueeze(2).repeat([1, 1, p_num]).reshape([cps.size(0), -1])
    t1 = t[:, 1:-2].unsqueeze(2).repeat([1, 1, p_num]).reshape([cps.size(0), -1])
    t2 = t[:, 2:-1].unsqueeze(2).repeat([1, 1, p_num]).reshape([cps.size(0), -1])
    t3 = t[:, 3:].unsqueeze(2).repeat([1, 1, p_num]).reshape([cps.size(0), -1])

    auxillary_cps0 = auxillary_cps[:, 0:-3, :].unsqueeze(2).repeat([1, 1, p_num, 1]).reshape([cps.size(0), -1, 2])
    auxillary_cps1 = auxillary_cps[:, 1:-2, :].unsqueeze(2).repeat([1, 1, p_num, 1]).reshape([cps.size(0), -1, 2])
    auxillary_cps2 = auxillary_cps[:, 2:-1, :].unsqueeze(2).repeat([1, 1, p_num, 1]).reshape([cps.size(0), -1, 2])
    auxillary_cps3 = auxillary_cps[:, 3:, :].unsqueeze(2).repeat([1, 1, p_num, 1]).reshape([cps.size(0), -1, 2])

    mx01 = ((t1 - v) / (t1 - t0)).unsqueeze(2).repeat([1, 1, 2]) * auxillary_cps0 + \
           ((v - t0) / (t1 - t0)).unsqueeze(2).repeat([1, 1, 2]) * auxillary_cps1

    mx12 = ((t2 - v) / (t2 - t1)).unsqueeze(2).repeat([1, 1, 2]) * auxillary_cps1 + \
           ((v - t1) / (t2 - t1)).unsqueeze(2).repeat([1, 1, 2]) * auxillary_cps2

    mx23 = ((t3 - v) / (t3 - t2)).unsqueeze(2).repeat([1, 1, 2]) * auxillary_cps2 + \
           ((v - t2) / (t3 - t2)).unsqueeze(2).repeat([1, 1, 2]) * auxillary_cps3

    mx012 = ((t2 - v) / (t2 - t0)).unsqueeze(2).repeat([1, 1, 2]) * mx01 \
            + ((v - t0) / (t2 - t0)).unsqueeze(2).repeat([1, 1, 2]) * mx12

    mx123 = ((t3 - v) / (t3 - t1)).unsqueeze(2).repeat([1, 1, 2]) * mx12 \
            + ((v - t1) / (t3 - t1)).unsqueeze(2).repeat([1, 1, 2]) * mx23

    points[:, :] = ((t2 - v) / (t2 - t1)).unsqueeze(2).repeat([1, 1, 2]) * mx012 \
                   + ((v - t1) / (t2 - t1)).unsqueeze(2).repeat([1, 1, 2]) * mx123

    return points
