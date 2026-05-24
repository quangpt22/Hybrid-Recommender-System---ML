Weighted Hybrid
Result:

Item-based CF Baseline:
RMSE: 0.9698
Precision@5: 0.0295
Recall@5: 0.1474
NDCG@5: 0.0940
MRR@5: 0.0767

User-based CF Baseline:
RMSE: 0.9737
Precision@5: 0.0242
Recall@5: 0.1209
NDCG@5: 0.0771
MRR@5: 0.0630

Content-based CF Baseline:
RMSE: 1.0685
Precision@5: 0.0187
Recall@5: 0.0933
NDCG@5: 0.0607
MRR@5: 0.0501

Hybrid Recommender System:

Tuning Hybrid (Item-based + Content-based):
Alpha = 0.00: RMSE = 1.0685, Precision@5 = 0.0187, Recall@5 = 0.0933, NDCG@5: 0.0607, MRR@5: 0.0501
Alpha = 0.05: RMSE = 1.0579, Precision@5 = 0.0182, Recall@5 = 0.0912, NDCG@5: 0.0569, MRR@5: 0.0458
Alpha = 0.10: RMSE = 1.0478, Precision@5 = 0.0195, Recall@5 = 0.0976, NDCG@5: 0.0606, MRR@5: 0.0487
Alpha = 0.15: RMSE = 1.0382, Precision@5 = 0.0206, Recall@5 = 0.1029, NDCG@5: 0.0646, MRR@5: 0.0522
Alpha = 0.20: RMSE = 1.0291, Precision@5 = 0.0221, Recall@5 = 0.1103, NDCG@5: 0.0702, MRR@5: 0.0571
Alpha = 0.25: RMSE = 1.0207, Precision@5 = 0.0233, Recall@5 = 0.1166, NDCG@5: 0.0750, MRR@5: 0.0615
Alpha = 0.30: RMSE = 1.0128, Precision@5 = 0.0233, Recall@5 = 0.1166, NDCG@5: 0.0768, MRR@5: 0.0637
Alpha = 0.35: RMSE = 1.0055, Precision@5 = 0.0244, Recall@5 = 0.1220, NDCG@5: 0.0814, MRR@5: 0.0680
Alpha = 0.40: RMSE = 0.9988, Precision@5 = 0.0255, Recall@5 = 0.1273, NDCG@5: 0.0837, MRR@5: 0.0695
Alpha = 0.45: RMSE = 0.9928, Precision@5 = 0.0269, Recall@5 = 0.1347, NDCG@5: 0.0888, MRR@5: 0.0737
Alpha = 0.50: RMSE = 0.9873, Precision@5 = 0.0282, Recall@5 = 0.1410, NDCG@5: 0.0919, MRR@5: 0.0757
Alpha = 0.55: RMSE = 0.9826, Precision@5 = 0.0295, Recall@5 = 0.1474, NDCG@5: 0.0932, MRR@5: 0.0756
Alpha = 0.60: RMSE = 0.9785, Precision@5 = 0.0301, Recall@5 = 0.1506, NDCG@5: 0.0949, MRR@5: 0.0766
Alpha = 0.65: RMSE = 0.9750, Precision@5 = 0.0308, Recall@5 = 0.1538, NDCG@5: 0.0963, MRR@5: 0.0775
Alpha = 0.70: RMSE = 0.9722, Precision@5 = 0.0312, Recall@5 = 0.1559, NDCG@5: 0.0965, MRR@5: 0.0771
Alpha = 0.75: RMSE = 0.9701, Precision@5 = 0.0322, Recall@5 = 0.1612, NDCG@5: 0.0986, MRR@5: 0.0782
Alpha = 0.80: RMSE = 0.9687, Precision@5 = 0.0327, Recall@5 = 0.1633, NDCG@5: 0.1001, MRR@5: 0.0796
Alpha = 0.85: RMSE = 0.9679, Precision@5 = 0.0318, Recall@5 = 0.1591, NDCG@5: 0.0977, MRR@5: 0.0778
Alpha = 0.90: RMSE = 0.9679, Precision@5 = 0.0310, Recall@5 = 0.1548, NDCG@5: 0.0954, MRR@5: 0.0761
Alpha = 0.95: RMSE = 0.9685, Precision@5 = 0.0295, Recall@5 = 0.1474, NDCG@5: 0.0935, MRR@5: 0.0759
Alpha = 1.00: RMSE = 0.9698, Precision@5 = 0.0295, Recall@5 = 0.1474, NDCG@5: 0.0940, MRR@5: 0.0767

Best alpha: 0.90 with RMSE = 0.9679

Compare Hybrid (Item-based + Content-based) vs Item-based
Item-based CF: RMSE: 0.9698
Hybrid (α=0.90): RMSE: 0.9679
Improvement: 0.0019 (0.2%)

Tuning Hybrid (User-based + Content-based):
Alpha = 0.00: RMSE = 1.0685, Precision@5 = 0.0187, Recall@5 = 0.0933, NDCG@5: 0.0607, MRR@5: 0.0501
Alpha = 0.05: RMSE = 1.0563, Precision@5 = 0.0182, Recall@5 = 0.0912, NDCG@5: 0.0560, MRR@5: 0.0446
Alpha = 0.10: RMSE = 1.0447, Precision@5 = 0.0195, Recall@5 = 0.0976, NDCG@5: 0.0601, MRR@5: 0.0479
Alpha = 0.15: RMSE = 1.0338, Precision@5 = 0.0206, Recall@5 = 0.1029, NDCG@5: 0.0652, MRR@5: 0.0528
Alpha = 0.20: RMSE = 1.0237, Precision@5 = 0.0214, Recall@5 = 0.1071, NDCG@5: 0.0689, MRR@5: 0.0562
Alpha = 0.25: RMSE = 1.0142, Precision@5 = 0.0235, Recall@5 = 0.1177, NDCG@5: 0.0759, MRR@5: 0.0623
Alpha = 0.30: RMSE = 1.0056, Precision@5 = 0.0233, Recall@5 = 0.1166, NDCG@5: 0.0761, MRR@5: 0.0627
Alpha = 0.35: RMSE = 0.9978, Precision@5 = 0.0225, Recall@5 = 0.1124, NDCG@5: 0.0728, MRR@5: 0.0597
Alpha = 0.40: RMSE = 0.9907, Precision@5 = 0.0255, Recall@5 = 0.1273, NDCG@5: 0.0785, MRR@5: 0.0626
Alpha = 0.45: RMSE = 0.9845, Precision@5 = 0.0257, Recall@5 = 0.1283, NDCG@5: 0.0772, MRR@5: 0.0605
Alpha = 0.50: RMSE = 0.9791, Precision@5 = 0.0252, Recall@5 = 0.1262, NDCG@5: 0.0745, MRR@5: 0.0576
Alpha = 0.55: RMSE = 0.9746, Precision@5 = 0.0248, Recall@5 = 0.1241, NDCG@5: 0.0749, MRR@5: 0.0588
Alpha = 0.60: RMSE = 0.9710, Precision@5 = 0.0242, Recall@5 = 0.1209, NDCG@5: 0.0727, MRR@5: 0.0569
Alpha = 0.65: RMSE = 0.9682, Precision@5 = 0.0252, Recall@5 = 0.1262, NDCG@5: 0.0745, MRR@5: 0.0577
Alpha = 0.70: RMSE = 0.9663, Precision@5 = 0.0263, Recall@5 = 0.1315, NDCG@5: 0.0761, MRR@5: 0.0580
Alpha = 0.75: RMSE = 0.9653, Precision@5 = 0.0255, Recall@5 = 0.1273, NDCG@5: 0.0743, MRR@5: 0.0571
Alpha = 0.80: RMSE = 0.9652, Precision@5 = 0.0248, Recall@5 = 0.1241, NDCG@5: 0.0716, MRR@5: 0.0547
Alpha = 0.85: RMSE = 0.9660, Precision@5 = 0.0235, Recall@5 = 0.1177, NDCG@5: 0.0690, MRR@5: 0.0532
Alpha = 0.90: RMSE = 0.9677, Precision@5 = 0.0229, Recall@5 = 0.1145, NDCG@5: 0.0666, MRR@5: 0.0511
Alpha = 0.95: RMSE = 0.9702, Precision@5 = 0.0218, Recall@5 = 0.1092, NDCG@5: 0.0650, MRR@5: 0.0507
Alpha = 1.00: RMSE = 0.9737, Precision@5 = 0.0242, Recall@5 = 0.1209, NDCG@5: 0.0771, MRR@5: 0.0630

Best alpha: 0.80 with RMSE = 0.9652

Compare Hybrid (User-based + Content-based) vs User-based:
User-based CF: RMSE: 0.9737
Hybrid (α=0.80): RMSE: 0.9652
Improvement: 0.0085 (0.9%)
