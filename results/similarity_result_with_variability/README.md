### Condition:
The exposure condition during human experiment

### TrainingTalkerID:
Determined by Condition. Not reordered yet. The Control and Multi-Talker conditions contain five talkers, while the Single and Talker-Specific conditions contain one talker.
Example:"['ENG_M_055', 'ENG_M_066', 'ENG_M_070', 'ENG_M_131', 'ENG_M_133']"

### TestTalkerID:
Selected TestTalker, only 1 CMN Talker.
Example:"['ALL_016_M_CMN']"

### SentenceID:
Currently selected sentence with SentenceSet and Sentence number information combination.
Example: "HT1_S001". HT1,HT2 two SentenceSet and each contain 16 sentences.

### Keyword:
Currently selected keyword.
Example: "shoes"

### similarity:
The similarity between Exposure 3D-tSNE features and Test 3D-tSNE feature after DTW. See more in paper.
Example:"0.825303"

### IsCorrect
Whether the keyword transcription is correct.
Example:"1"

### VariabilityAcrossTime:
Variability calculation method 1: mean deviation from the mean time_steps ('3D-point')
Example:"0.838999"

### VariabilityInSimilarityAcrossWords:
Variability calculation method 2. Contain two aggregation methods: a. 1-mean(similarity). b.SD(similarity)
Example:"[0.8032 , 0.1366]"

### ParticipantID
Subject ID for human experiments.
Example:"f8b17be6a35f51d7f154ff160f33e71c"

### Trial
Human experiments Trial.

### Data_parameters_info:
Similarity_{data}_{asr-model}_{layer}_{aggregation function}_{w/ or w/o control}_{w/ or w/o talker-specific}_{tau}_{best k}

### TrainingTalkerID_sorted
Sorted TrainingTalkerID (Exposure), removed data redundancy.

### UniqueExposureTestCombination
