class HighwayVideoClips(Dataset):
    
    def __init__(self, root, T=16, size=112, train=True, split=0.8, cache_labels=True):
        self.paths = sorted([p for p in glob.glob(os.path.join(root, "**", "*.avi"), recursive=True)])
        assert self.paths
        # split
        k = int(len(self.paths)*split)
        self.paths = self.paths[:k] if train else self.paths[k:]
        self.T, self.size = T, size
        # pseudo-labels
        self.labels = {}
        if cache_labels:
            for p in self.paths:
                self.labels[p] = pseudo_label_day_night(p)

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        y = self.labels[p] if p in self.labels else pseudo_label_day_night(p)
        clip = read_clip_cv2(p, self.T, self.size)      # (T,H,W,C) in [0,1]
        clip = clip.transpose(3,0,1,2)                  # -> (C,T,H,W)
        x = torch.from_numpy(clip)                      # float32
        y = torch.tensor(y, dtype=torch.long)
        return x, y

    def summarize_counts(self, counts, class_names=None, title="Class distribution"):
        ks = sorted(counts.keys())
        vs = np.array([counts[k] for k in ks])
        plt.hist(vs, bins='auto', edgecolor='black')
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.title("Frequency Histogram Distribution")

