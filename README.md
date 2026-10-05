# Robust Backdoor Detection for Deep Learning via Topological Evolution Dynamics

This is the official repository for the paper "[Robust Backdoor Detection for Deep Learning via Topological Evolution Dynamics](https://arxiv.org/abs/2312.02673)" presented at IEEE Symposium on Security and Privacy (S&P) 2024.

![Topology Persistence Diagram](./TopologyPersistenceDiagram_CIAFR10_Layer_DeepColor-1.png)


## Source-Specific and Dynamic-Triggers (SSDT) Attack

To execute the Source-Specific and Dynamic-Triggers (SSDT) attack on the CIFAR-10, MNIST, or GTSRB dataset, use the following configuration:

- **Command**: `python train_SSDT.py`
- **Arguments**:
  - `--dataset [cifar10/mnist/gtsrb]` (replace with the desired dataset)
  - `--attack_mode SSDT`
  - `--n_iters 300`

Example command for CIFAR-10:

```bash
python train_SSDT.py --dataset cifar10 --attack_mode SSDT --n_iters 300
```

## Topological Evolution Dynamics (TED) Defense

To explore the TED defense methodology, use the `TED.ipynb` Jupyter Notebook provided in this repository.

## Citation

If you use this code in your research or project, please cite our paper:

```bibtex
@inproceedings{mo2024robust,
  title     = {Robust Backdoor Detection for Deep Learning via Topological Evolution Dynamics},
  author    = {Mo, Xiaoxing and Zhang, Yechao and Zhang, Leo Yu and Luo, Wei and Sun, Nan and Hu, Shengshan and Gao, Shang and Xiang, Yang},
  booktitle = {2024 IEEE Symposium on Security and Privacy (SP)},
  pages     = {2048--2066},
  year      = {2024},
  url       = {https://arxiv.org/abs/2312.02673}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](./LICENSE) for details.
