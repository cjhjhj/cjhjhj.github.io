---
layout: post
title: "What are M, M+1, M+2, M+H, M+2H, ... peaks?"
categories: proteomics
comments: true
---

## M, M+1, M+2 혹은 (M+H)<sup>+</sup>, (M+2H)<sup>2+</sup> peak들은 무엇을 의미할까?

### Charge states
MS spectrum에 나타나는 peak들은 charge (흔히 z로 표시되고 z = 1, 2, .. 등으로 나타낸다) 를 갖고 있는 peptide들을 측정한 것으로서 mass가 아닌 m/z (mass-to-charge ratio) 의 차원을 갖는다. Singly-charged (z = 1) peptide의 경우에는 측정된 peak이 monoisotopic mass를 가리키게 되며, z = 2, 3, ... 등으로 높은 charge state를 갖는 경우에는 아래 그림과 같이 doubly (혹은 triply)-charged된 m/z값을 갖는 peak로 표시된다 (그림 출처: https://www.waters.com/webassets/cms/events/docs/us_events/2015_us_events/QDa_QRC_Mass_Data_Terminology_Considerations_Interpretation.pdf)

![Image of peaks](/assets/img/proteomics/isotopicPeaks.png)

어떤 특정한 peptide (species) 가 여러 charge state를 갖으면서 측정되었다면 당연히 (M+H)+, 즉 singly-charged species, 가 가장 큰 m/z값을 갖게 되어 그림상에서는 가장 왼쪽에 위치하며 charge state가 증가할 수록 peak에서의 m/z값이 작아지게 된다.

### Isotopic peaks
Isotopic peaks들은 자연적으로 존재하는 동위원소들 (C, H, N, O 등 대부분의 원소들이 동위원소를 갖지만 mass spectrometry에 가장 큰 영향을 주는 동위원소는 <sup>13</sup>C이다) 때문에 생성된다. M을 monoisotopic peak의 mass라고 했을 때 (z = 1로 가정), M+1은 한 개의 <sup>13</sup>C의 존재때문에 발생하는 약 1 Da 정도 (정확하게는 <sup>13</sup>C - <sup>12</sup>C) 의 mass-shift를 갖는 peak를 가리킨다. 위의 그림에서 m/z = 1046.54의 peak이 monoisotopic peak (= M)이고, m/z = 1047.55에서의 peak이 M+1 이다. 마찬가지로 m/z = 1048.55, 1049.55는 각각 2, 3개의 <sup>13</sup>C 때문에 발생하는 M+2, M+3 peak을 가리킨다.  
많은 경우 isotopic peak들은 monoisotopic peak에 비하면 intensity가 낮지만, mass에 따라서 혹은 peptide (species) 를 구성하는 원소들에 따라서 꽤 높은 intensity를 갖거나 monoisotopic peak보다 높은 intensity를 갖기도 한다.  
Isotopic peaks가 중요한 이유중의 하나는 isotope spacing을 이용해서 해당 peptide의 charge state를 유추할 수 있기 때문이다. 예를 들어, M과 M+1 peak의 m/z 값 차이가 1에 가깝다면 z = 1이라는 의미이며, M과 M+1 peak의 m/z 값 차이가 0.5에 가깝다면 z = 2라는 의미이다. (그림 참조)

![Image of peaks](/assets/img/proteomics/isotopicPeaks_charge_20190903.png)
