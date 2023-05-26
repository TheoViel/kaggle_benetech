import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def extract_similarities(fts, idx, preds, img, min_sim=0.6, verbose=0):
    sim_img = []
    resize_to = img.shape[:2]

    for ft_idx in [0, 1, 2]:
        feats = fts[ft_idx][idx]
        feats = feats / ((feats ** 2).sum(0, keepdims=True) + 1e-6).sqrt()

        sims = []
        for box in preds[-1][:5]:
        #     print(box)
            y = (box[0] + box[2]) / 2
            y = int(y / img.shape[1] * feats.size(2))
            x = (box[1] + box[3]) / 2
            x = int(x / img.shape[0] * feats.size(1))

            vec = feats[:, x, y][:, None, None]

        #     sim = ((feats - vec) ** 2).mean(0, keepdims=True)
        #     sim = 1 / (sim + 1)

            sim = (feats * vec).sum(0, keepdims=True)

            sim = torch.clamp(sim, torch.quantile(sim, 0.9) * 1.05, 1)
            sim = (sim - sim.min()) / (sim.max() - sim.min())

            sim = torch.where(sim < min_sim, 0, sim)
        #     sims.append(sim)
            sims.append(sim)

#             plt.imshow(sim[0].cpu().numpy())
#             plt.colorbar()
#             plt.show()
#             break
            
        sims = torch.cat(sims).mean(0, keepdims=True)
        sims = F.interpolate(sims.unsqueeze(0), resize_to)[0]

        if verbose:
            plt.imshow(sims[0].cpu().numpy())
            plt.colorbar()
            plt.show()
            
        sim_img.append(sims)
        
    sim_img = torch.cat(sim_img)
    return sim_img.cpu().numpy().transpose(1, 2, 0)
