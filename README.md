# Phase3 Clustering Analysis

## Clustering Grouping

```bash
>>> datasets.groupby(desire_cols[2:]).size().sort_values(ascending=False)
Fatal_Injury  Non-fatal_injury  P.D._only  Traffic_signal  Stop_sign  Yield_sign  School_bus  Traffic_gate  Traffic_controller  No_control  Roundabout
0             0                 1          0               0          0           0           0             0                   1           0             6400
                                           1               0          0           0           0             0                   0           0             4546
              1                 0          1               0          0           0           0             0                   0           0             1183
              0                 1          0               1          0           0           0             0                   0           0             1131
              1                 0          0               0          0           0           0             0                   1           0             1079
                                                           1          0           0           0             0                   0           0              341
              0                 1          0               0          1           0           0             0                   0           0              101
                                                                      0           0           0             0                   0           1               86
              1                 0          0               0          1           0           0             0                   0           0               32
1             0                 0          0               0          0           0           0             0                   1           0               16
0             0                 1          0               0          0           0           1             0                   0           0               10
1             0                 0          1               0          0           0           0             0                   0           0               10
0             0                 1          0               0          0           0           0             1                   0           0                4
              1                 0          0               0          0           0           0             1                   0           0                3
                                                                                              1             0                   0           0                3
                                                                                              0             0                   0           1                3
1             0                 0          0               1          0           0           0             0                   0           0                2
0             0                 1          0               0          0           1           0             0                   0           0                2
```

- We can conclude the clustering should be the most sampled region
  - PD Damage with no control revelances
  - PD Damage with Traffic Signal
  - PD Damage with stop signs
  - Non-Fatal Injuries with Traffic Signal
  - Should see the most of the traffic control have more connection toward property damage and Traffic Signal

