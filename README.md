# Stress-Detection-ABayes

Tor Finseth, 2023. 

This work was funded by the National Aeronautics and Space Administration (grant number 80NSSC18K1572).

T. T. Finseth, M. C. Dorneich, S. Vardeman, N. Keren and W. D. Franke, "Real-Time Personalized Physiologically Based Stress Detection for Hazardous Operations," in IEEE Access, vol. 11, pp. 25431-25454, 2023, doi: [10.1109/ACCESS.2023.3254134](https://doi.org/10.1109/ACCESS.2023.3254134).
___

## OVERVIEW
When training for hazardous operations, real-time stress detection is an asset for optimizing task performance and reducing stress. Stress detection systems train a machine-learning model with physiological signals to classify stress levels of unseen data. Unfortunately, individual differences and the time-series nature of physiological signals limit the effectiveness of generalized models and hinder both post-hoc stress detection and real-time monitoring. This study evaluated a personalized stress detection system that selects a personalized subset of features for model training. The system was evaluated post-hoc for real-time deployment. Further, traditional classifiers were assessed for error caused by indirect approximations against a benchmark, optimal probability classifier (Approximate Bayes; ABayes). Healthy participants completed a task with three levels of stressors (low, medium, high), either a complex task in virtual reality (responding to spaceflight emergency fires, n =27) or a simple laboratory-based task (N-back, n =14). Heart rate, blood pressure, electrodermal activity, and respiration were assessed. Personalized features and window sizes were compared. Classification performance was compared for ABayes, support vector machine, decision tree, and random forest. The results demonstrate that a personalized model with time series intervals can classify three stress levels with higher accuracy than a generalized model. However, cross-validation and holdout performance varied for traditional classifiers vs. ABayes, suggesting error from indirect approximations. The selected features changed with window size and tasks, but found blood pressure was most prominent. The capability to account for individual difference is an advantage of personalized models and will likely have a growing presence in future detection systems.

This repo contains the the dataset and post-hoc code. The real-time code will be released in another repository pending publication of another paper.

## DATASET INFO
Data were collected for the machine-learning pipeline using four physiological signals that were acquired simultaneously: electrocardiogram (ECG), Electrodermal Activity (EDA), Respiration (RSP), and Noninvasive Blood Pressure (NIBP). Biopac MP150 system (Biopac Systems Inc., Santa Barbara, CA) was used to measure ECG, and was equipped with an ECG100C module [58]. ECG and RSP were sampled using Biopac MP150 (125 Hz) and Bionomadix Bioshirt that uses Bluetooth signal thereby increasing mobility of the participant. Beat-to-beat blood pressure data were collected using an oscillometric NIBP fingercuff placed on the participants’ nondominant hand over the middle phalanx of the long and ring finger (CNAP Monitor 500, CNSystems Medizintechnik AG).

Forty-one healthy participants (34 male, 7 female) performed a complex task in virtual reality (spaceflight emergency fire, N = 27) or a laboratory-based task (N-back, N = 14). The mean age was 20.9±6.5 years, all adults in the age range of 18-41 years.

## Classes
Each participant completed a task consisting three stressor levels (low, medium, and high). These levels were validated in a previous study (see T. Finseth, M. C. Dorneich, N. Keren, W. D. Franke and S. Vardeman, "Designing Training Scenarios for Stressful Spaceflight Emergency Procedures," 2020 AIAA/IEEE 39th Digital Avionics Systems Conference (DASC), San Antonio, TX, USA, 2020, pp. 1-10, doi: [10.1109/DASC50938.2020.9256403](https://doi.org/10.1109/DASC50938.2020.9256403)).
