# CBFS-Net
This is the code of our submitted paper CBFS-Net.
# Abstract
Despite the great progress made by deep CNNs in defect segmentation, achieving both real-time performance and high accuracy in detecting tiny defects in ultra-high-resolution images can still be a challenging task. Lightweight network architecture design, attention mechanisms, and multi-scale feature fusion techniques have thus been developed to address this issue. Our proposed CBFS-Net introduces a unique defect segmentation approach that emphasizes feature selection to more effectively leverage information from defective areas. First, we partition the ultra-high-resolution input image into equally sized regions and group them together into batches to generate region proposals. Then, we employ a hierarchical transformer encoder to effectively capture multi-scale features for the proposals in a single forward pass. Second, we attach a feature selection module (FSM), which consists of an MLP classifier and a selector, to the end of the encoder for defective feature selection. With FSM, massive non-defective background features are filtered out, which reduces memory usage and improves computational efficiency. Finally, we develop a lightweight feature fusion decoder that can efficiently combine the multi-scale defective features and produce highly accurate defect segmentation masks. Significantly, our model achieves an impressive mDice score of 91.26\% at a processing speed of 10.64 FPS on the compact camera module (CCM) defect segmentation dataset recently annotated by our team, thereby outperforming the state-of-the-art method in terms of both accuracy and efficiency.
# Network Architecture
![CBFS-Net](https://github.com/Howietzh/CBFS-Net/blob/master/CBFS-Net.png)
# CCM Defect Segmentation Dataset
[Baidu YunPan](https://pan.baidu.com/s/1gHzgIPkfZPxW7bYVwoXQTw "CCM Defect Segmentation Dataset"), retrieval code：hog0
# Train
python tools/train.py config_file.py --gpu-id 0 --seed 0 --work-dir workdirs
# Test
python tools/test.py config_file.py checkpoint_file.pth --eval mDice mFscore
