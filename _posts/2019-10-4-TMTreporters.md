---
layout: post
title: "TMT-labeling"
categories: proteomics
comments: true
---

## TMT-labeling의 원리
TMT-labeling은 isobaric labeling technique의 하나이다. 
Label의 전체적인 mass는 일정하게 유지하면서, <sup>13</sup>C나 <sup>15</sup>N등의 isotope의 위치를 reporter마다 다르게 하여 
multiplex labeling을 가능하게 하는 기술이다. TMT reagent의 chemical structure는 다음과 같다 
(그림 출처: Cheng, L et al. Peptide labeling using isobaric tagging reagents for quantitative phosphoproteomics, 
Methods Mol Biol. 2016; 1355: 53-70).

![TMTreporter](/assets/img/proteomics/tmt_reporter1.png)

위의 그림에서 별표 (*, asterisk)는 isotope가 존재할 수 있는 위치를 나타내며, isotope의 위치에 따라서 reporter group과 balancer group의
mass가 달라진다 (그렇지만 그 합은 항상 일정하다 = isobaric).

아래 그림에서 보듯이 N-terminus와 결합하거나 Lysine에 결합되면 reporter와 balancer group만 modification에 기여하게 되므로, 
약 229 Da의 static modification이 발생하게 된다 (그림 출처: Bachor, R et al. Trends in the design of new isobaric labeling reagents for quantitative proteomics, Molecules. 2019; 24: E701).

![TMTmodification](/assets/img/proteomics/tmt_reporter2.png)


TMT-10 plex의 경우 각각의 reporter는 다음의 mass를 갖는다 (positively charged).  

|Label|<sup>12</sup>C|<sup>13</sup>C|<sup>14</sup>N|<sup>15</sup>N|H|e<sup>-</sup>|Mass| 
|-----|--------------|--------------|--------------|--------------|-|-------------|----|  
|sig126|8||1||16|-1|126.1277259|  
|sig127N|8|||1|16|-1|127.1247608|  
|sig127C|7|1|1||16|-1|127.1310808|
|...|...|...|...|...|...|...|...|  
|sig130N|5|3||1|16|-1|130.1348253|  
|sig130C|4|4|1||16|-1|130.1411453|  
|sig131|4|4||1|16|-1|131.1381802|  
