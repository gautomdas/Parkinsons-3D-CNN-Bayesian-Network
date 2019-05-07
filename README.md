# 3D CNN Bayesian Network for Parkinson's Detection

Results:

![PPMI_3052_MR_SAG_3D_T1__br_raw_20110314144145885_132_S101544_I223769](https://github.com/gautomdas/Parkinsons-3D-CNN-Bayesian-Network/blob/master/PPMI_3052_MR_SAG_3D_T1__br_raw_20110314144145885_132_S101544_I223769.jpg)



![PPMI_3006_MR_sag_3D_FSPGR_BRAVO_straight__br_raw_20110705115013407_37_S113519_I243182](https://github.com/gautomdas/Parkinsons-3D-CNN-Bayesian-Network/blob/master/PPMI_3006_MR_sag_3D_FSPGR_BRAVO_straight__br_raw_20110705115013407_37_S113519_I243182.jpg)


This project adapts a regular 3D CNN network seen below:

![model](https://github.com/gautomdas/Parkinsons-3D-CNN-Bayesian-Network/blob/master/model.png)


### Tensorflow 2.0 Adaptions

From the original model the following adaptions were made to take advantage of tensorflow 2.0:

1. Using tensorflow probabilities to build a bayesian network, this allows for researchers to get an additional insight into how much the network believes in its final answer.
2. Eager executions were used to avoid having to go through the trouble of generating a session runtime for tensorflow



### Disclaimers

This is not a working product nor does it claim to be. This is merely a proof of concept. The original data was provided by the Parkinson's Progression Markers Initiative.
