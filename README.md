# Trust-But-Verify: Poisoning-Resilient Locally Private Graph Learning Protocols 

Built upon local differential privacy (LDP), locally private graph learning protocols have emerged as an important paradigm for decentralized graph learning, balancing privacy protection and learning utility. Under such protocols, each user locally perturbs their node features and adjacency information before transmission, ensuring formal privacy guarantees without original data leaving the device. However, the inherently open participation nature renders these protocols critically vulnerable to data poisoning attacks, where adversaries inject carefully crafted malicious nodes to corrupt neighborhood aggregation and degrade downstream utility. Despite the severity of this threat, effective defenses in this setting remain largely unexplored. In this paper, we propose VERITAS, a poisoning-resilient locally private graph learning protocol built on a trust-but-verify paradigm. By introducing a verification list encoding graded peer trust levels, VERITAS jointly privatizes node features and graph structure on the user side, while exploiting bilateral attestation asymmetry on the server side to identify and prune malicious nodes. Concretely, VERITAS comprises four synergistic stages: (1) local data perturbation, (2) attestation-driven malicious node pruning, (3) utility restoration via dual denoising, and (4) robust private graph learning. Extensive experiments on four real-world benchmark datasets across multiple LDP mechanisms and GNN architectures demonstrate that VERITAS effectively defends against data poisoning attacks and significantly improves downstream graph learning utility under rigorous privacy guarantees.
<img width="2108" height="596" alt="image" src="https://github.com/user-attachments/assets/16abddfc-ab2b-4ae0-8662-c50615fe2e41" />

### Methodology
<img width="2080" height="924" alt="image" src="https://github.com/user-attachments/assets/bcbc1bdf-7ac7-4823-bc57-fd92ac028cf0" />

### Project Layout
```
ICDM26/
├── veritas/
│   ├── mechanisms.py  
│   ├── structure.py   
│   ├── attack.py       
│   ├── denoise.py     
│   ├── models.py      
│   ├── data.py         
│   ├── pipeline.py     
│   └── main.py         
├── run_fig3.py         
├── summarize_fig3.py 
├── requirements.txt
└── README.md
```
